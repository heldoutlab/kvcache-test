# Is q8_0 KV-cache quantization free? And should you quantize V, not K?

Pre-registered test by [Held Out Lab](https://heldoutlab.substack.com): independent tests of AI model claims, with the rules fixed before the run.

**Write-up with charts:** [I tested the "q8_0 KV cache is free" tip](https://heldoutlab.substack.com/p/i-tested-the-q8_0-kv-cache-is-free) · [Subscribe](https://heldoutlab.substack.com/subscribe) to get the next test (LFM3-1.7B, Liquid's claim on my agent benchmark).

**Status: complete (main run 2026-10-25/26).** [`PREREG.md`](PREREG.md) holds the claim, the pass/fail rule, the author's prediction (pushed before any model ran), the pilot log and the sha256 of every task and harness file. Public tasks are in `tasks/`, the harness in `harness/`, every raw output and server log in `results/`. The private held-out set is not published; only its hash is.

- **Claim:** "q8_0 KV cache is free; if you must go lower, quantize V to q4_0 but leave K alone."
- **Models:** Muse Glimmer 30B and Gemma 4 26B-A4B, Q4_K_M, stock llama.cpp, on a 32 GB M1 Pro MacBook.
- **Tasks:** fresh long-context questions (about 4k, 16k and 48k tokens) generated from scratch, each with an exact answer key.

## Results

- **Muse Glimmer 30B: q8_0 — no loss found.** 83 vs 81 of 96 questions correct (q8_0 vs f16); identical answers at 4k and 16k. Generation 2–6% slower.
- **Gemma 4 26B-A4B: q8_0 — not resolved.** 63 vs 66 of 96 (gap at 16k). Generation 29–47% slower with q8_0.
- **"Quantize V, not K": no difference found.** q8 K / q4 V and q4 K / q8 V both 82 of 96 on Glimmer. On Apple Metal, both mixed settings ran flash attention on the CPU: prompt reading about 4x slower at 48k.
- **Prediction:** not proved wrong by its pre-registered thresholds, but mostly off (details in the [write-up](https://heldoutlab.substack.com/p/i-tested-the-q8_0-kv-cache-is-free)).

Reports: [`results/report_glimmer.txt`](results/report_glimmer.txt), [`results/report_gemma4.txt`](results/report_gemma4.txt) (and `report.json` per model). Re-run the analysis with `python3 harness/analyze.py`.

Questions or corrections: hello@heldoutlab.com. [Subscribe](https://heldoutlab.substack.com/subscribe) for the next test.

Code is MIT-licensed (see `LICENSE`); task texts and results are CC BY 4.0. Tasks, graders and harness were written with help from Claude (Anthropic).
