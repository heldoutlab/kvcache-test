# Is q8_0 KV-cache quantization free? And should you quantize V, not K?

Pre-registered test by [Held Out Lab](https://heldoutlab.substack.com): independent tests of AI model claims, with the rules fixed before the run.

**Status: pre-registered, not yet run.** [`PREREG.md`](PREREG.md) holds the claim, the pass/fail rule, the author's prediction and the settings, pushed before any model ran. The task files, harness and their hashes are added at the freeze (after a calibration pilot, before the main run); results, raw outputs and the write-up follow.

- **Claim:** "q8_0 KV cache is free; if you must go lower, quantize V to q4_0 but leave K alone."
- **Models:** Muse Glimmer 30B and Gemma 4 26B-A4B, Q4_K_M, stock llama.cpp, on a 32 GB M1 Pro MacBook.
- **Tasks:** fresh long-context questions (about 4k, 16k and 48k tokens) generated from scratch, each with an exact answer key.

Questions or corrections: hello@heldoutlab.com. [Subscribe](https://heldoutlab.substack.com/subscribe) for the results.

Code is MIT-licensed (see `LICENSE`); task texts and results are CC BY 4.0. Tasks, graders and harness were written with help from Claude (Anthropic).
