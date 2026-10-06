//! Run one Kimi-K3 pipeline stage (`scripts/gen_stage.py --layers A:B --chunk N`)
//! over one prefill chunk of a fresh sequence, its boundary tensors from and
//! to files: the gate of a stage against SGLang's PP proxy tensors, and the
//! link that chains kern's own stages.
//!
//!   cargo run --release --manifest-path stage/Cargo.toml -- \
//!       --manifest k3-pp-l23-46.json --weights <HF checkpoint dir> [--gpu 0] \
//!       (--tokens ids.i64 | --hidden-in hidden.bf16 --blocks-in blocks.bf16) \
//!       --out <dir> [--iters 5]
//!
//! The first stage takes the chunk's token ids (i64), a later one the stream
//! it was left: `hidden` [T, H] and the attention-residual bank [T, nb, H]
//! with its nb valid snapshots (SGLang's `residual`, unpadded; padded here to
//! the manifest's bank). A stage with a head writes `next_token.i64` and
//! `logits.f32` of the chunk's last row; one without writes `hidden.bf16`
//! [T, H] and `blocks.bf16` [T, nb_out, H], nb_out = ceil(B / 12), the files
//! the next stage takes. `--iters` reruns the chunk for its median wall time
//! (the states it rewrites are not read back).
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use kern_manifest::types::Dim;
use kern_runtime::{Capacity, Runtime, Topology};

const H: usize = 7168;
const BLOCK: usize = 12;

fn le_i64(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn le_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn dim(d: &Dim) -> usize {
    match d {
        Dim::Const(n) => *n as usize,
        d => panic!("expected a constant dimension, got {d:?}"),
    }
}

/// `rows` rows of `src` ([rows, nb_src, H] bf16) re-laid as [rows, nb_dst, H],
/// the snapshots past `min(nb_src, nb_dst)` zero.
fn rebank(src: &[u8], rows: usize, nb_src: usize, nb_dst: usize) -> Vec<u8> {
    let row = H * 2;
    let n = nb_src.min(nb_dst) * row;
    (0..rows)
        .flat_map(|r| {
            let s = &src[r * nb_src * row..][..n];
            s.iter().copied().chain(std::iter::repeat_n(0, nb_dst * row - n))
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let mut manifest = PathBuf::new();
    let mut weights = PathBuf::new();
    let mut gpu = 0usize;
    let mut tokens = None;
    let mut hidden_in = None;
    let mut blocks_in = None;
    let mut out = PathBuf::new();
    let mut iters = 0usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut v = || args.next().expect("value");
        match a.as_str() {
            "--manifest" => manifest = PathBuf::from(v()),
            "--weights" => weights = PathBuf::from(v()),
            "--gpu" => gpu = v().parse()?,
            "--tokens" => tokens = Some(PathBuf::from(v())),
            "--hidden-in" => hidden_in = Some(PathBuf::from(v())),
            "--blocks-in" => blocks_in = Some(PathBuf::from(v())),
            "--out" => out = PathBuf::from(v()),
            "--iters" => iters = v().parse()?,
            _ => anyhow::bail!("unknown arg {a}"),
        }
    }
    let m = kern_manifest::Verified::from_json(&std::fs::read_to_string(&manifest)?)?;
    let (first, end) = {
        let name = m.model.rsplit('/').nth(1).unwrap_or_default();
        let (a, b) = name.trim_start_matches('l').split_once('-').ok_or_else(|| {
            anyhow::anyhow!("model `{}`: not a pipeline stage (gen_stage.py --layers A:B)", m.model)
        })?;
        (a.parse::<usize>()?, b.parse::<usize>()?)
    };
    let nb_max = dim(&m.buffers["blocks"].shape[1]);
    let (nb_in, nb_out) = (first.div_ceil(BLOCK), end.div_ceil(BLOCK));
    let head = m.buffers.contains_key("next_token");

    let table = &m.buffers["block_table"];
    let per_row = dim(&table.shape[1]) * table.domain.as_ref().map(|d| d.stride as usize).unwrap_or(1);
    let capacity = Capacity { tokens: Some(per_row as u64), seqs: 1 };
    let topo = Topology::one("ep", 0, 1);
    let t0 = Instant::now();
    let mut rt = Runtime::load(&m, None, gpu, Some(capacity), Some(&topo))?;
    rt.load_weights(&kern_runtime::Safetensors::open(&[&weights])?)?;
    let mine = rt.export_handles()?;
    rt.import_peers("ep", &[mine])?;
    let once = BTreeMap::from([("tokens".to_string(), 1u64), ("seqs".to_string(), 1), ("rows".to_string(), 1)]);
    for (name, p) in &m.programs {
        if p.once {
            rt.run(name, &once)?;
        }
    }
    eprintln!("loaded `{}` in {:.1} s", m.model, t0.elapsed().as_secs_f64());

    let (ids, hidden, blocks) = match (&tokens, &hidden_in, &blocks_in) {
        (Some(t), None, None) => (Some(std::fs::read(t)?), None, None),
        (None, Some(h), Some(b)) => (None, Some(std::fs::read(h)?), Some(std::fs::read(b)?)),
        _ => anyhow::bail!("give --tokens, or --hidden-in with --blocks-in"),
    };
    let t = match (&ids, &hidden) {
        (Some(ids), _) => ids.len() / 8,
        (_, Some(h)) => h.len() / (H * 2),
        _ => unreachable!(),
    };
    anyhow::ensure!(ids.is_some() == (first == 0), "stage [{first}, {end}) takes {}", if first == 0 { "--tokens" } else { "--hidden-in/--blocks-in" });

    let vars = BTreeMap::from([
        ("tokens".to_string(), t as u64),
        ("seqs".to_string(), 1),
        ("rows".to_string(), t as u64),
        ("ctx".to_string(), t as u64),
    ]);
    let lease = rt.lease(per_row)?;
    for name in rt.page_tables().map(str::to_string).collect::<Vec<_>>() {
        let mut row = Vec::new();
        lease.extend_row(&name, &mut row)?;
        rt.write_input(&name, &le_i32(&row))?;
    }
    let rows_max = m.vars["rows"].max as usize;
    for name in rt.seq_tables().map(str::to_string).collect::<Vec<_>>() {
        // one column per row of the batch; the chunk's sequence is row 0
        let mut col = Vec::new();
        for r in 0..lease.seq_lines(&name)? {
            col.push(lease.seq_line(&name, r)?);
            col.extend(std::iter::repeat_n(0, rows_max - 1));
        }
        rt.write_input(&name, &le_i32(&col))?;
    }
    rt.write_input("span_at", &le_i32(&[0]))?;
    rt.write_input_at("slot_mapping", &le_i64(&(0..t).map(|j| lease.slot(j)).collect::<Vec<_>>()), &vars)?;
    rt.write_input_at("seq_lens", &le_i32(&[t as i32]), &vars)?;
    if let Some(ids) = &ids {
        rt.write_input_at("token_ids", ids, &vars)?;
    } else {
        let blocks = blocks.unwrap();
        let nb_src = blocks.len() / (t * H * 2);
        anyhow::ensure!(nb_src >= nb_in, "--blocks-in has {nb_src} snapshots, layer {first} needs {nb_in}");
        rt.write_input_at("hidden_in", &hidden.unwrap(), &vars)?;
        rt.write_input_at("blocks_in", &rebank(&blocks, t, nb_src, nb_max), &vars)?;
    }

    let t1 = Instant::now();
    rt.run("prefill", &vars)?;
    let first_ms = t1.elapsed().as_secs_f64() * 1e3;
    std::fs::create_dir_all(&out)?;
    if head {
        let next = rt.read_output("next_token")?;
        std::fs::write(out.join("next_token.i64"), &next[..8])?;
        let v = dim(&m.buffers["logits"].shape[1]);
        std::fs::write(out.join("logits.f32"), rt.read_buffer_prefix("logits", v * 4)?)?;
        println!("next_token={}", i64::from_le_bytes(next[..8].try_into()?));
    } else {
        let hidden = rt.read_output("hidden")?;
        std::fs::write(out.join("hidden.bf16"), &hidden[..t * H * 2])?;
        let blocks = rt.read_output("blocks")?;
        std::fs::write(out.join("blocks.bf16"), rebank(&blocks, t, nb_max, nb_out))?;
    }
    let mut ms: Vec<f64> = (0..iters)
        .map(|_| {
            let s = Instant::now();
            rt.run("prefill", &vars).map(|_| s.elapsed().as_secs_f64() * 1e3)
        })
        .collect::<Result<_, _>>()?;
    ms.sort_by(f64::total_cmp);
    let median = ms.get(ms.len() / 2).copied();
    println!(
        "stage [{first}, {end}) T={t}: first run {first_ms:.1} ms{}",
        median.map(|x| format!(", median of {iters} {x:.1} ms")).unwrap_or_default()
    );
    Ok(())
}
