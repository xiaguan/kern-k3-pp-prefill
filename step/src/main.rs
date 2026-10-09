//! Run a Kimi-K3 decode manifest (`scripts/gen_stage.py --dcp`) teacher-forced
//! on one rank: B sequences fed their given tokens one decode step at a
//! time, every step's logits kept. One process per rank; a group's ranks
//! meet through an NCCL id in a file on a shared disk, so they may sit on
//! different nodes. A manifest whose paged states are dealt by position
//! over the group runs one replicated batch: every member takes the same
//! rows, and only what addresses a member's own positions differs (its
//! slots, the pad's for the positions it does not hold, its local lengths,
//! its page and line tables).
//!
//!   k3_step --manifest dcp-l4-tp8.json --weights <HF checkpoint dir> --gpu 0 \
//!       --rank 0 --world 8 --nccl-id <shared dir>/id --tokens ids.i64 --rows 4 --out <dir>
//!
//! `ids.i64` is the rows' tokens, [rows, steps] i64; step s feeds every row
//! its token s at position s. Every rank writes `next.r<rank>.i64` [steps,
//! rows]; rank 0 also writes `top.f64` [steps, rows, 3] (the top-1 token, its
//! logit and the runner-up's) and `logits.f32` [kept, rows, vocab] of every
//! `--every`-th step (s + 1 a multiple of it; default 1, every step). Without
//! `--world` (a group of one, the manifest's own oracle) no NCCL is joined.
//! `--dump <buffer>` (repeatable) has rank 0 also write `<buffer>.bin`, the
//! buffer's live rows after every step: a workspace the last layer leaves
//! behind, such as its routing (`topk_idx`).
//! A manifest of two layers (`--layers 2`: one KDA layer, one MoE layer) has
//! no paged state and takes no slots or lengths.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use kern_manifest::types::Dim;
use kern_pool::Interleave;
use kern_runtime::{Capacity, NcclId, Runtime, Topology};

struct Args {
    manifest: PathBuf,
    weights: PathBuf,
    gpu: usize,
    rank: u64,
    world: u64,
    nccl_id: Option<PathBuf>,
    tokens: PathBuf,
    rows: usize,
    every: usize,
    dump: Vec<String>,
    out: PathBuf,
}

fn args() -> anyhow::Result<Args> {
    let mut a = Args {
        manifest: PathBuf::new(),
        weights: PathBuf::new(),
        gpu: 0,
        rank: 0,
        world: 1,
        nccl_id: None,
        tokens: PathBuf::new(),
        rows: 1,
        every: 1,
        dump: Vec::new(),
        out: PathBuf::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| anyhow::anyhow!("{k} takes a value"));
        match k.as_str() {
            "--manifest" => a.manifest = v()?.into(),
            "--weights" => a.weights = v()?.into(),
            "--gpu" => a.gpu = v()?.parse()?,
            "--rank" => a.rank = v()?.parse()?,
            "--world" => a.world = v()?.parse()?,
            "--nccl-id" => a.nccl_id = Some(v()?.into()),
            "--tokens" => a.tokens = v()?.into(),
            "--rows" => a.rows = v()?.parse()?,
            "--every" => a.every = v()?.parse()?,
            "--dump" => a.dump.push(v()?),
            "--out" => a.out = v()?.into(),
            _ => anyhow::bail!("unknown arg {k}"),
        }
    }
    Ok(a)
}

fn le<T: Copy, const N: usize>(v: &[T], f: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    v.iter().flat_map(|x| f(*x)).collect()
}

/// A row's top-1 token, its logit and the runner-up's logit.
fn top2(row: &[f32]) -> [f64; 3] {
    let (i, v) = row.iter().enumerate().fold((0, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m });
    let second = row.iter().enumerate().filter(|&(j, _)| j != i).fold(f32::MIN, |m, (_, &x)| m.max(x));
    [i as f64, v as f64, second as f64]
}

/// Rank 0 writes a fresh id; the others wait for it to appear.
fn nccl_id(path: &Path, rank: u64) -> anyhow::Result<NcclId> {
    if rank == 0 {
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, NcclId::new()?.to_bytes())?;
        std::fs::rename(&tmp, path)?;
    }
    let t = Instant::now();
    loop {
        if let Ok(b) = std::fs::read(path) {
            if let Ok(b) = <[u8; 128]>::try_from(b.as_slice()) {
                return Ok(NcclId::from_bytes(b));
            }
        }
        anyhow::ensure!(t.elapsed() < Duration::from_secs(3600), "no nccl id at {} after an hour", path.display());
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn main() -> anyhow::Result<()> {
    let a = args()?;
    let m = kern_manifest::Verified::from_json(&std::fs::read_to_string(&a.manifest)?)?;
    let ids: Vec<i64> =
        std::fs::read(&a.tokens)?.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
    let (b, steps) = (a.rows, ids.len() / a.rows);
    anyhow::ensure!(b * steps == ids.len() && steps > 0, "{} tokens are not {b} rows", ids.len());
    let seqs_max = m.vars["seqs"].max as usize;
    anyhow::ensure!(b <= seqs_max, "{b} rows, the manifest takes at most {seqs_max}");
    let dealt = m.states.values().any(|s| s.shard.is_some());
    // a manifest with no MLA layer (one MoE layer) has no paged state to address
    let paged = m.buffers.contains_key("slot_mapping");
    anyhow::ensure!(dealt == (a.world > 1), "a dealt manifest runs over --world > 1, an undealt one alone");
    let il = Interleave::new(a.world)?;
    let local = steps.div_ceil(a.world as usize);

    let t0 = Instant::now();
    let topo = Topology::one("tp", a.rank, a.world);
    // every row's local positions, plus the pad's page
    let capacity = Capacity { tokens: Some(((local + 64) * (b + 1)) as u64), seqs: (b + 1) as u64 };
    let mut rt = Runtime::load(&m, None, a.gpu, Some(capacity), Some(&topo))?;
    rt.load_weights(&kern_runtime::Safetensors::open(&[&a.weights])?)?;
    if a.world > 1 {
        let path = a.nccl_id.as_ref().ok_or_else(|| anyhow::anyhow!("--world > 1 needs --nccl-id"))?;
        rt.join_nccl("tp", &nccl_id(path, a.rank)?)?;
    }
    let once = BTreeMap::from([("tokens".to_string(), 1u64), ("seqs".to_string(), 1)]);
    for (name, p) in &m.programs {
        if p.once {
            rt.run(name, &once)?;
        }
    }
    eprintln!("rank {} loaded `{}` in {:.1} s", a.rank, m.model, t0.elapsed().as_secs_f64());

    let leases = (0..b).map(|_| if paged { rt.lease(local) } else { rt.lease_slot() }).collect::<Result<Vec<_>, _>>()?;
    // Positions this member does not hold write into a page nobody reads.
    let pad = rt.lease(1)?;
    let me = a.rank;
    let vars = BTreeMap::from([("tokens".to_string(), b as u64), ("seqs".to_string(), b as u64)]);
    for name in rt.page_tables().map(str::to_string).collect::<Vec<_>>() {
        let mut t = Vec::new();
        for l in &leases {
            l.extend_row(&name, &mut t)?;
        }
        rt.write_input_at(&name, &le(&t, i32::to_le_bytes), &vars)?;
    }
    for name in rt.seq_tables().map(str::to_string).collect::<Vec<_>>() {
        let lines = leases[0].seq_lines(&name)?;
        let mut t = Vec::with_capacity(lines * seqs_max);
        for r in 0..lines {
            for l in &leases {
                t.push(l.seq_line(&name, r)?);
            }
            t.resize((r + 1) * seqs_max, 0);
        }
        rt.write_input(&name, &le(&t, i32::to_le_bytes))?;
    }
    rt.write_input("span_at", &le(&[0i32], i32::to_le_bytes))?;

    let vocab = match m.buffers["logits"].shape[1] {
        Dim::Const(n) => n as usize,
        ref d => anyhow::bail!("logits width {d:?}"),
    };
    let row_bytes = |name: &str| -> anyhow::Result<usize> {
        let buf = m.buffers.get(name).ok_or_else(|| anyhow::anyhow!("no buffer `{name}` to dump"))?;
        let consts: Vec<u64> = buf.shape[1..].iter().map(|d| match d { Dim::Const(n) => Ok(*n), d => Err(anyhow::anyhow!("`{name}` dim {d:?}")) }).collect::<anyhow::Result<_>>()?;
        Ok((consts.iter().product::<u64>() * buf.dtype.bytes()) as usize)
    };
    let dumps: Vec<(String, usize)> = a.dump.iter().map(|n| Ok((n.clone(), row_bytes(n)? * b))).collect::<anyhow::Result<_>>()?;
    let mut dumped = vec![Vec::new(); dumps.len()];
    let (mut next, mut logits, mut top) = (Vec::new(), Vec::new(), Vec::<f64>::new());
    let t1 = Instant::now();
    for s in 0..steps {
        let p = s as u64;
        let tok: Vec<i64> = (0..b).map(|r| ids[r * steps + s]).collect();
        rt.write_input_at("token_ids", &le(&tok, i64::to_le_bytes), &vars)?;
        if paged {
            let slot: Vec<i64> = leases
                .iter()
                .map(|l| if il.owner(p) == me { l.slot(il.local(p) as usize) } else { pad.slot(0) })
                .collect();
            let len = vec![il.len(p + 1, me) as i32; b];
            rt.write_input_at("slot_mapping", &le(&slot, i64::to_le_bytes), &vars)?;
            rt.write_input_at("seq_lens", &le(&len, i32::to_le_bytes), &vars)?;
        }
        rt.run("decode", &vars)?;
        next.extend_from_slice(&rt.read_output("next_token")?[..b * 8]);
        if me == 0 {
            let l = rt.read_buffer_prefix("logits", b * vocab * 4)?;
            let f: Vec<f32> = l.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            top.extend(f.chunks_exact(vocab).flat_map(top2));
            if (s + 1) % a.every == 0 {
                logits.extend(l);
            }
            for ((name, n), d) in dumps.iter().zip(&mut dumped) {
                d.extend(rt.read_buffer_prefix(name, *n)?);
            }
        }
    }
    let ms = t1.elapsed().as_secs_f64() * 1e3 / steps as f64;
    std::fs::create_dir_all(&a.out)?;
    std::fs::write(a.out.join(format!("next.r{me}.i64")), &next)?;
    if me == 0 {
        std::fs::write(a.out.join("logits.f32"), &logits)?;
        std::fs::write(a.out.join("top.f64"), le(&top, f64::to_le_bytes))?;
        for ((name, _), d) in dumps.iter().zip(&dumped) {
            std::fs::write(a.out.join(format!("{name}.bin")), d)?;
        }
    }
    println!("rank {me}: {steps} steps x {b} rows, {ms:.1} ms per step (eager, with readback)");
    Ok(())
}
