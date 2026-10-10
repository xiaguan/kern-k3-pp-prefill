#pragma once
// The packed prefill's FMHA tables (kern_k3_fmha_lens_varlen in k3_prefill.cu, and block 0 of
// kern_k3g_embed_rms in k3_prefill_glue.cu where a lone chunk's stage 0 makes them in its first launch).
//
// fmha_plan: the FMHA's tables for a packed call of `n`
// sequences, three rows of `ns` (>= n + 3) words: seq_lens_kv, cum_seq_lens_q
// and cum_seq_lens_kv over the call's n + 2 FMHA sequences, then the split
// record at word 3 ns. The kernel aligns its causal mask to cum_kv's spans, so
// a padded span would show a sequence zero keys; the tile past a sequence's end
// reads the next one's real rows, masked, and past the last the gather's zeros.
//
// A call of short_max < T <= SPLIT_ROWS rows is one or two Q tiles a head, 96
// or 192 CTAs of one context's length on 152 SMs. Its heaviest sequence s
// (Lq rows over P cached ones, row i sees keys [0, P + i]) is split at key c
// into three causal pieces of about a third of the work each:
//   z = s       N: keys [c + 1, P + Lq), q as is: row i sees [c + 1, P + i]
//   z = n       R: keys c .. 1 (reversed), q rows reversed (at T + Lq - 1 - i): row i sees [i + 1, c]
//   z = n + 1   A: keys [0, Lq), q rows again (at T + Lq + i): row i sees [0, i]
// merged by the gate through the FMHA's softmax stats. Unsplit, z = n and
// n + 1 are empty. Split record: {s or -1, c, cum_q[s], Lq, P}.
#define SPLIT_ROWS 512
__device__ __forceinline__ void fmha_plan(const int* __restrict__ seq_lens, const long long* __restrict__ cu_q,
                                          int* __restrict__ lens, int n, int ns, int short_max) {
  const int T = (int)cu_q[n];
  int s = -1;
  long long best = 0;
  if (T > short_max && T <= SPLIT_ROWS)
    for (int j = 0; j < n; ++j) {
      const long long lq = cu_q[j + 1] - cu_q[j], work = lq * seq_lens[j];
      if (lq > 0 && seq_lens[j] - lq >= 2 * lq + 2 && work > best) best = work, s = j;
    }
  const int lq = s < 0 ? 0 : (int)(cu_q[s + 1] - cu_q[s]), p = s < 0 ? 0 : seq_lens[s] - lq;
  const int c = s < 0 ? 0 : min(max(seq_lens[s] / 3, lq), p - 1);
  int* const kl = lens;
  int* const cq = lens + ns;
  int* const ck = lens + 2 * ns;
  int kv = 0;
  for (int j = 0; j < n; ++j) {
    kl[j] = j == s ? seq_lens[j] - c - 1 : seq_lens[j];
    cq[j] = (int)cu_q[j];
    ck[j] = kv;
    kv += kl[j];
  }
  kl[n] = c;
  kl[n + 1] = lq;
  cq[n] = T, cq[n + 1] = T + lq, cq[n + 2] = T + 2 * lq;
  ck[n] = kv, ck[n + 1] = kv + c, ck[n + 2] = kv + c + lq;
  int* const rec = lens + 3 * ns;
  rec[0] = s, rec[1] = c, rec[2] = s < 0 ? 0 : (int)cu_q[s], rec[3] = lq, rec[4] = p;
}
