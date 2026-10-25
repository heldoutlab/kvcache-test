# Pre-registration: "q8_0 KV cache is free; below that, quantize V, not K"

Drafted 2026-10-22 (with Claude's help); rule, second model and the author's prediction fixed 2026-10-25. **Pushed to github.com/heldoutlab/kvcache-test before any model ran, including the pilot.** Task files and harness hashes are added at the freeze, after the pilot and before the main run, under the rules written below.

## Claim under test
"q8_0 KV cache is free (no measurable quality loss); if you must go lower, quantize V to q4_0 but leave K alone."

Where it is said (all read 2026-10-22):
- PaulKeeble, HN comment, 2026-10-17. Use `--cache-type-k q8_0 --cache-type-v q8_0`; avoid going below 8 bits; if you must, drop V to q4_0 and leave K alone. Thread: "Show HN: Llmspeed", 329 points, 2026-10-17. https://news.ycombinator.com/item?id=50129807
- Same thread, trucks-refinish, 2026-10-17: runs keys at q8 and values at q4. https://news.ycombinator.com/item?id=50129769
- The opposite view in the largest recent thread. Aurornis, 2026-08-14, on "Qwen 3.8 27B" (1,438 points): even q8 KV "comes with notable drops in performance for longer tasks". https://news.ycombinator.com/item?id=49301538
- Reader requests for this test. alentred, 2026-09-08: uses q8_0 to fit 100k tokens and wants to know what it costs. Ey7NFZ3P0nzAe, 2026-09-09, asks to quantize K differently from V. Both on the Quesma thread "Benchmarking Qwen3.8 27B quantizations" (286 points, 2026-09-08). https://news.ycombinator.com/item?id=49613840, https://news.ycombinator.com/item?id=49622547
- A citation that does not support the claim. A reply in that thread (skolos, 2026-09-08) says arXiv 2609.04098 shows KV quantization has "almost no impact … all the way to q4". The paper is about NVFP4 W4A4 weights and FP8 KV scales. Its abstract does not test llama.cpp q8_0/q4_0 caches or a K/V split. https://arxiv.org/abs/2609.04098

## What is already known (prior work)
| Source | What it measured | What it found | Why it does not settle the claim |
|---|---|---|---|
| localbench (oobabooga), 2026-04-24, https://localbench.substack.com/p/kv-cache-quantization-benchmark | KL divergence vs f16 cache, ~250k tokens, 4 models, K and V set to the same type | q8_0 KL: Qwen 3.6 27B 0.024, Qwen 3.6 35B-A3B 0.039, Gemma 4 31B 0.108, Gemma 4 26B-A4B 0.377. q4_0 on Gemma 26B-A4B: KL 1.088, top-1 68.0%. States that "q8_0 is practically lossless" is wrong for Gemma | No task accuracy. No K/V split. Context lengths and backend not stated |
| llama.cpp Discussion #23470 (opened by sanmai 2026-05-21; replies to 2026-09-17), https://github.com/ggml-org/llama.cpp/discussions/23470 | KLD/PPL on wikitext, plus answer churn on 500 ARC-Challenge questions (chambejp, 2026-08-16 and 09-12) | Qwen2.5-7B at 512 context: q4_0 K collapses (KLD 5.49), f16 K / q4_0 V 0.0040, q8_0/q8_0 0.0018. ARC churn out of 500: q8_0/q8_0 changes at most 4 answers; q8 K / q4 V changes 1/10/1 on three models; q4_0 K alone collapses Qwen2.5-7B (375/500). Qwen3.8-27B at 65,536 context, 99% KLD: q8/q8 0.0064, K q8 / V q4 0.0123, K q4 / V q8 0.0168, q4/q4 0.0221 | Short multiple-choice answers, or KL on wikitext. No long-context task accuracy. No Metal results; backend often unstated |
| llama.cpp PR #21038 (ggerganov, merged 2026-04-01), https://github.com/ggml-org/llama.cpp/pull/21038 | Adds a Hadamard rotation of Q/K/V for quantized caches. AIME25 ×8, gpt-oss-20b, low reasoning | F16 cache 37.9%; q8_0 31.7% before the PR and 37.1% after; q4_0 2.0% before and 21.7% after | Same type for K and V. One model, short context. Shows that older "KV quant is bad" reports predate the rotation |
| KIVI (arXiv 2402.02750; v1 2024-02-05, v2 2024-07-25), https://arxiv.org/abs/2402.02750 | Key and value distributions | Keys have fixed outlier channels and should be quantized per channel; values per token. Per-token quantization hurts keys far more (Table 2) | This is the source of the "leave K alone" intuition. llama.cpp's q4_0/q8_0 are per-token blocks of 32. The automatic rotation (above) targets the same outlier problem. Not tested on llama.cpp |
| InventiveHQ, 2026-06-26, https://inventivehq.com/blog/kv-cache-quantization-quality-benchmark | difflib similarity of greedy output vs f16, Qwen2.5-Coder-7B, 8k context, 12 prompts | q8_0 81.6%, q4_0 8.3% similarity. Also M3 Max speeds | Similarity is not accuracy. n = 12. K and V set to the same type |
| Level1Techs, "Why your local LLM feels dumber than it is" (2026-08-16; HN 511 points 2026-08-22) | vLLM int8/int4 KV cache, one ~100k-token agentic prompt | int8 "eventually" recovered; int4 did not | vLLM, not llama.cpp. One prompt |
| Memoriant DGX Spark benchmark (2026-03/04), https://github.com/Memoriant/dgx-spark-kv-cache-benchmark | Speed and memory, CUDA | q8_0/q4_0 cut the KV buffer from 768 MiB to 408/216 MiB. Decode at ~110k: 38.0 → 25.0/24.0 tok/s | No quality measure |

**Verdict: still open.** Nobody has published long-context task accuracy for llama.cpp's K/V split, and nobody has published anything on Apple Metal. The evidence for "V before K" is KL/PPL on wikitext and answer churn on short multiple-choice questions. The 2026-09-12 KLD matrix on Qwen3.8-27B points the same way as the claim (K q8/V q4 0.0123 < K q4/V q8 0.0168). The gap is much smaller than in older reports, which predate the automatic rotation.

## Runtime facts that shape the design (stock Homebrew llama.cpp 0.4.1, build 10964, commit b29c606e2; not upgraded, `brew` offers 0.7.0)
Read from the source at that commit in a local clone (`~/dev/company/llama.cpp-fork`) unless a link is given.
1. **A quantized V cache requires flash attention.** With `-fa auto`, a quantized V forces it on. With `-fa off`, startup fails ("quantized V cache requires flash_attn"; `src/llama-context.cpp`). Issue #21450 (M1 Pro 32 GB, 2026-04-04, closed as stale 2026-05-19) shows the older failure mode for q8_0 K / q4_0 V. https://github.com/ggml-org/llama.cpp/issues/21450. All arms here run with `-fa on`.
2. **On Metal, flash attention needs K and V of the same type.** `ggml-metal-device.m`, supports_op for `FLASH_ATTN_EXT`: `if (op->src[1]->type != op->src[2]->type) return false;`. This is unchanged on upstream master (2026-09-26) and in the fork's head (2026-10-20). Uniform types run on the GPU: f32, f16, q8_0, q4_0, q4_1, q5_0, q5_1, and bf16 only where the GPU reports bfloat support. **The two arms the claim is about (q8_0/q4_0 and q4_0/q8_0) are therefore expected to run attention on the CPU.** The scheduler sends ops the GPU does not support to the CPU. That path is the vector kernel with no tiling. It also converts Q to K's dot-product type (q8_0) before the K·Q product (`ggml-cpu/ops.cpp`, `q_to_vec_dot`), which is an extra rounding step the Metal path does not have. This is expected from the code. It has not been observed, and the pilot confirms or refutes it from prompt-processing speed and the server log. A Mac user who follows the advice gets this path, so it is what we test. The optional arm P (below) separates "CPU path" from "mixed types". The CUDA equivalent is that mixed types need `GGML_CUDA_FA_ALL_QUANTS`, otherwise attention silently falls back to the CPU (issue #24485, 2026-06-11). https://github.com/ggml-org/llama.cpp/issues/24485
3. **Metal prefill with a quantized cache dequantizes to F16 first** and then runs the F16 flash-attention kernel (PR #27390, 2026-08-20; "only for large batches" #27438). Decode uses the quantized vector kernels.
4. **A Hadamard rotation is applied automatically** to K (and V) whenever that cache is quantized and the head size is a multiple of 64 (Glimmer: 128). This comes from PR #21038 (`src/llama-kv-cache.cpp`); `LLAMA_ATTN_ROT_DISABLE=1` turns it off. All runs keep the default (on).
5. A Metal fix for quantized flash attention landed after 0.4.1: a043d38a6, "fix excess threadgroup memory in quantized flash attention", #29340, 2026-10-06. Whether it affects this model is UNVERIFIED. The pilot catches crashes or garbage output.
6. **Sliding-window model.** For Glimmer, `-ctk/-ctv` apply to both the global and the sliding-window cache (`llama-kv-cache-iswa.cpp`). Only the 13 global layers hold the long-range context.
7. **Prompt reuse.** llama-server makes a context checkpoint at the start of each user message (Muse Glimmer delimiter `<|start|>user`; 32 checkpoints by default). The document and the question are sent as two separate user messages, so the 7 later questions on a document should reuse the document instead of re-reading it. `timings.prompt_n` is logged for every request to check this. If reuse fails, the fallback is `--swa-full`, which changes memory and speed but not outputs.

## Model
**Muse Glimmer 30B, `meta-models/Muse-Glimmer-30B-GGUF`, file `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`.** 16,756,683,904 bytes; sha256 `4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e` (HF LFS id, read 2026-10-22). Already on disk at `/Users/admin/models/`. The runner hashes the local file and refuses to start on a mismatch. It has not been hashed yet.
- Popular: downloads in the last 30 days, read 2026-10-22 from the HF API: meta-models GGUF 211,483; unsloth GGUF 562,637; lmstudio-community GGUF 114,660. Card context 131,072.
- Architecture from GGUF metadata: `muse-glimmer`, 52 layers, 39 sliding-window (2048) and 13 global ([L, L, L, G] repeating), 32 query and 2 KV heads, head size 128, gated attention, logit softcap 20. The card says RoPE is used on local layers only. The card also contradicts itself: one table says "sliding-window, all layers, 32/8 heads". The GGUF metadata is used here.
- Fits easily. Analytic KV cache at `-c 65536` (`harness/config.py: kv_bytes`): f16/f16 930 MiB, q8_0/q8_0 494 MiB, mixed 378 MiB, q4_0/q4_0 261 MiB. **For this model, q8_0 saves only about 0.44 GB.** People quantize the KV cache to save memory, and that motive is weak here. The post must say so.
- Same file and binary as the Bonsai round-2 calibration (results there: 55–60 tok/s prefill, 5.4–6.1 tok/s decode on short prompts, M1 Pro).

**Second model (author's decision, 2026-10-25): Gemma 4 26B-A4B**, the most KV-sensitive model in the localbench test. File `unsloth/gemma-4-26B-A4B-it-GGUF` / `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` (16.95 GB, sha256 f2c28b3d…3f293f). Arms A and B only (part 1 of the rule); thinking off (the card's default, passed explicitly as `enable_thinking: false` because llama-server defaults to on); output cap 8,192 (pilot); card sampler temperature 1.0, top_p 0.95, top_k 64. Arms C and D are run on Gemma only if the time rule leaves room, as a descriptive addition.

## Arms (one server start per arm; everything else identical)
| Arm | `-ctk` | `-ctv` | Role |
|---|---|---|---|
| A | f16 | f16 | reference (llama.cpp default) |
| B | q8_0 | q8_0 | "q8_0 is free" |
| C | q8_0 | q4_0 | the advice: quantize V, not K |
| D | q4_0 | q8_0 | mirror of the advice |
| E (optional) | q4_0 | q4_0 | how bad uniform 4-bit is |
| P (optional) | q8_0 | q8_0 + `-nkvo` | path control: same types as B, cache and attention on the CPU like C/D |

Server: `llama-server -m <file> -ngl 99 -fa on -c 65536 -np 1 -b 2048 -ub 512 --jinja -ctk X -ctv Y`. Port 8793.

## Contexts and task sets
Documents are generated from scratch by `gen_tasks.py` (Python 3.12, stdlib only, seeded and deterministic). Each document is a fictional "operations archive": reference tables (people, assets, projects) and a dated daily journal (work orders, expense claims, contract amendments). There are 8 questions per document, each with an exact answer key:

| Family | Per document | What it needs | Grading |
|---|---|---|---|
| R: multi-needle retrieval | 2 | the recorded code of 3 named entries spread over the journal's depth (one per third), each with a near-miss ID decoy | all 3 in order (also reported: share of needles right) |
| M3: 3-hop lookup | 1 | asset → owning project → project lead → lead's building; decoy asset with a near-miss ID | exact option (lenient containment if no other option named) |
| M2: 2-hop lookup | 1 | journal entry → person → unit; decoy person with the same surname | as M3 |
| AS: aggregation, sum | 1 | total of 4–7 claims matching unit + category + status, plus 6 near-match decoys | integer |
| AC: aggregation, count | 1 | count of 3–6 entries matching person + priority + status, plus 3 decoys | integer |
| L: latest version | 2 | value of a clause after a chain of 3–5 amendments; some WITHDRAWN (the last one in about half of chains); decoys on the same contract and same clause | number |

Target lengths, approximate (chars ÷ 4): **4k ≈ 3,600, 16k ≈ 15,000, 48k ≈ 46,000** tokens per document, within ±3%. The difficulty settings are identical at every length; only the amount of filler changes.

| Set | Template | Documents | Questions | Use |
|---|---|---|---|---|
| public | pipe tables, terse lines (facilities department) | 6 per context = 18 | 144 | the test; published after the run |
| private | prose log lines, different domain, IDs, surnames and seed (freight cooperative) | 6 per context = 18 | 144 | held out, never published (hash only) |
| calibration | key=value rows (equipment office) | 1 per context = 3 | 18 (6 per context) | difficulty and timing pilot; never scored |

The private template and seed live in `private_theme.py` (git-ignored). `python3.12 gen_tasks.py --check` regenerates everything twice in memory. It checks byte-identical output and solves every question again from the rendered text alone (regex parse, not the generator's state). It also checks that every line parses, that IDs and names are unique, that every reference resolves, that sizes are on target, that the sets share no IDs, names or question text, and that the files on disk match. It currently prints OK. The real token counts come from the model's tokenizer at run time (`run.py --preflight`). IDs and numbers usually tokenize worse than 4 characters per token, so the 48k documents may come out too long. Rule: if any prompt falls outside 0.80–1.00 × its nominal size (4,096 / 16,384 / 49,152), all sets are regenerated once with `--cpt` set to the measured characters per token. This happens before freezing.

## Settings
- System prompt: "You answer questions about the document the user provides. Use only the information in that document." Messages: system, user (document), user (question + "Reply with your answer on a final line in the form: ANSWER: <answer>").
- Sampler: the model card's, temperature 1.0, top_p 0.95, top_k 64, plus llama.cpp's default min_p 0.05. **Same seed for a question in every arm** (sha256 of run seed + question id; main run seed base 1001), so arms differ only through the cache.
- Reasoning cannot be switched off (card). `chat_template_kwargs: {"reasoning_strength": "low"}` for every arm, for cost. Output cap 4,096 tokens per question. A cut-off answer counts as wrong and is reported per arm. The cap is raised once to 8,192, before freezing, if more than 5% of arm-A calibration answers hit it.
- Order: all of arm A, then B, C, D, then the noise run. Within an arm: 4k, 16k, 48k; documents in ID order; questions in fixed order.
- Logged per request: answer, reasoning length, finish reason, prompt tokens processed (cache reuse), prefill and decode speed, wall time. Server RSS is logged after each document, plus the analytic KV size per arm.

## Pilot (before freezing; the test model never sees a public or private question before the freeze)
1. `run.py --preflight` (arm A server, tokenizer only) → token-count rule above.
2. Arm A on the 18 calibration questions. **Difficulty band: arm A must score 40–90% (8–16 of 18).** If above 90%, difficulty is raised once: needles 3→5, AS 8–12 records, AC 6–10, L chains 5–8. If below 40%, it is lowered once: needles 2, AS 2–4, AC 2–4, L 2–3. Then every set is regenerated and re-checked. The output-cap rule is applied here.
3. Timing: arms B, C and D on the calibration 16k document, and arm C on the calibration 48k document. This confirms or refutes the CPU fallback for C/D and gives seconds per question for each arm.
4. **Time rule:** project the machine hours for A–D plus the noise run. ≤ 40 h: 6 documents per context. 40–60 h: the first 4 documents per context, for every arm (`queue.sh 4`). Above 60 h at 4 documents: stop, and the author redesigns before freezing.
5. Freeze: record the hashes below and push.

**Pilot log (Glimmer)**
- 2026-10-25, step 1: the first preflight measured 3.153 characters per token, so all 39 prompts were too long (48k prompts about 57.5k tokens). All sets were regenerated once with `--cpt 3.15`, and `gen_tasks.py --check --cpt 3.15` printed OK. On the second preflight, all 36 public and private prompts were inside 0.80–1.00 × nominal. The 3 calibration prompts were 1.4%, 3.6% and 5.9% over nominal (4,153 / 16,973 / 52,030 tokens), because their theme tokenizes worse (2.8 characters per token). The rule allows one regeneration, so they were left as they are. They are used only for the difficulty band and timing, where a longer prompt makes the time estimate conservative. All of them fit the 65,536-token server context.
- 2026-10-25, step 2: arm A scored 15 of 18 (83%) on the calibration questions, inside the 40–90% band, so difficulty was not changed. No answer reached the 4,096-token cap (longest 1,314). Wall time was 45 min for the 3 calibration documents. Prefill ran at 56–62 tok/s and decode at 5.5–6.1 tok/s.
- 2026-10-25, step 3 (timing, calibration documents, Glimmer). The CPU-attention fallback for mixed K/V types is **confirmed**. Arm B (q8_0/q8_0) at 16k ran at the same speed as A (prefill 59 tok/s, decode 5.5 tok/s; 346 s for the first question). Arm D (K q4 / V q8) at 16k: prefill 18 tok/s, decode 2.5–3.0 tok/s, 1,010 s for the first question. Arm C (K q8 / V q4) at 16k: prefill 21 tok/s, decode 3.0 tok/s, 897 s. At 48k: prefill 11 tok/s, decode 1.9 tok/s, and 4,763 s (79 min) for the first question. Calibration correctness for these arms is not part of any analysis.
- 2026-10-25, Gemma 4 preflight: Gemma's tokenizer gives 2.65–2.75 characters per token on these documents (calibration theme 2.37). All public and private prompts are 4–11% over nominal on Gemma (4k ≈ 4.3k, 16k ≈ 17.8k, 48k ≈ 54.5k tokens). Decision: **the same documents are kept for both models, and nothing is regenerated.** The q8_0 comparison is within each model, on identical documents, so a longer real prompt makes the test harder for the claim, not easier. The 0.80–1.00 band was written before Gemma was added. The analysis reports each model's real token counts next to the nominal labels. Every prompt fits the 65,536-token context with the 4,096-token output cap (largest public prompt 54,701 tokens).
- 2026-10-25, Gemma 4 step 2, first attempt (set aside): llama-server turns thinking on by default, and Gemma's chat template follows that flag. So the first Gemma calibration ran **with thinking on**, against this document's "thinking off". It scored 8 of 18, with 7 replies cut off at 4,096 tokens while still reasoning (0 of 6 at 48k). Those records are kept in `results/gemma4/pilot_thinking_on/` and are not used. Fix: Gemma now sends `chat_template_kwargs: {"enable_thinking": false}` explicitly (`config.py`). The first replies after the fix have empty reasoning.
- 2026-10-25, Gemma 4 step 2, thinking off: arm A scored 9 of 18 (50%): 5 of 6 at 4k, 3 of 6 at 16k, **0 of 6 at 48k** (calibration 48k is 61,384 tokens on Gemma). The 48k answers are confident and wrong (wrong tags, wrong totals, the wrong person), not refusals. 1 of 18 replies hit the 4,096 cap (a repeating list of names), which is 5.6%, above the 5% trigger, so **Gemma's output cap is raised to 8,192** (`MODELS["gemma4"]["max_tokens"]`). Glimmer had 0 of 18 at the cap and keeps 4,096. The difficulty band is not applied to Gemma: tasks were already regenerated once, and both models must see the same documents.
- 2026-10-25, Gemma 4 timing (arm B, calibration 16k and 48k): prefill is the same as A (≈181 tok/s at 16k, ≈82 at 48k), but **decode is slower under q8_0**: 17 vs 27 tok/s at 16k and 10 vs 20 tok/s at 48k. Reported as a speed result.
- 2026-10-25, **floor safeguard (added before freezing, because of Gemma's 0 of 6 at 48k)**: a context where arm A scores below 25% on the public main run cannot show a 10-point loss and must not count as "no loss found". Such a context is reported in full but left out of the verdict for that model (both parts, and the pooled figures used by the rule). If every context is below the floor, the verdict is "NOT TESTABLE". `config.FLOOR_ACC = 25.0`; `analyze.py` lists `floor_contexts` and `contexts_used`. It applies to both models alike and was set before any model saw a public or private question. Consequence, stated now: if Gemma's 48k is below the floor, the author's "about 6 points on Gemma at 48k" prediction is scored as not testable, not as right or wrong.
- 2026-10-25, analysis fix: `analyze.py` printed a verdict only when arms C and D existed, so Gemma (A and B only) would have had none. Part 1 is now computed whenever A and B exist; part 2 says "NOT TESTED" for a model without C and D. Selftest updated and passes.
- 2026-10-25, **step 4, time rule**. Measured seconds per document (first question, which includes prefill, plus 7 follow-ups at the calibration mean). Glimmer: A 486 / 879 / 1,874 s (4k / 16k / 48k); B ≈ 1.09 × A (measured 962 s at 16k); C 1,095 / 1,982 / 7,815 s (4k scaled); D 1,464 / 2,648 / 10,441 s (4k and 48k scaled from 16k). Per set of three documents: A 0.9 h, B 1.0 h, C 3.0 h, D 4.0 h. Projection for A–D plus noise: **55 h at 6 documents per context, 37 h at 4**. 55 h falls in the 40–60 h band, so **the first 4 documents per context are used for every arm and both models** (`queue.sh 4`). Gemma A + B at 4 documents ≈ 4–5 h more. Expected total ≈ 41–42 h.

## Scoring and analysis (`harness/analyze.py`)
- Accuracy by arm × context, by family, and by needle depth (descriptive).
- **Paired differences vs arm A** (A−B, A−C, A−D, A−E) and **C−D**, per context and pooled. 95% intervals come from a stratified cluster bootstrap (resample whole documents within each context, 10,000 draws), because the 8 questions on one document are correlated. Discordant counts (A right/X wrong, and the reverse) are reported too.
- **Noise floor:** arm A run again with seed base 2002 on the 16k public documents (48 questions). The floor is |accuracy(seed 1001) − accuracy(seed 2002)|, with its discordance. This shows how much two runs of the same configuration disagree by chance at temperature 1.0.
- Speed: median full-document prefill tok/s and decode tok/s per arm × context, generated tokens per question, total hours. A prefill below 0.5× arm A's at the same context is flagged as "attention left the GPU". Memory: analytic KV MiB and peak server RSS.
- Broken items: an item found ambiguous or mis-keyed after the run is dropped for every arm and listed. It is never re-keyed. Items that every arm fails are inspected before reporting.
- The private set is not part of the verdict (see OPEN-QUESTIONS for an optional check).

## Pass/fail rule: CANDIDATE OPTIONS (the author picks one before freezing; `config.RULE`; analyze.py prints no verdict until it is set)
"Loss" means arm A minus the other arm, in percentage points, public set, main run.

**Chosen rule (author's decision, 2026-10-25): Option 1.** Applied to each model separately: part 1 (q8_0) for Muse Glimmer and Gemma 4; part 2 (V before K) for Muse Glimmer only.

**Option 1. Per-context thresholds (the same shape as the Bonsai 2 rule).**
- Part 1, q8_0: "NOT FREE" if the loss at any context is ≥ 10 points and its interval excludes 0. "NO LOSS FOUND" if the loss is ≤ 5 at every context and ≤ 3 pooled. Otherwise "NOT RESOLVED".
- Part 2, V before K: let g = accuracy C − accuracy D, pooled. "SUPPORTED" if g ≥ 5 and the interval excludes 0. "CONTRADICTED" if g ≤ −5 and the interval excludes 0. "NO DIFFERENCE FOUND" if |g| ≤ 3. Otherwise "NOT RESOLVED".

**Option 2. Relative to the noise floor** (floor = arm A seed-to-seed change, minimum 2 points).
- Part 1: "NOT FREE" if the loss at any context is > 2 × floor and its interval excludes 0. "WITHIN NOISE" if |loss| ≤ floor at every context. Otherwise "NOT RESOLVED".
- Part 2: "SUPPORTED" if g > floor and the interval excludes 0. "CONTRADICTED" if −g > floor and the interval excludes 0. "NO DIFFERENCE FOUND" if |g| ≤ floor. Otherwise "NOT RESOLVED".

**Option 3. Pooled, with a 48k guard and an equivalence-style "free".**
- Part 1: "NOT FREE" if the pooled loss is ≥ 5 with the interval excluding 0, or the 48k loss is ≥ 10 with the interval excluding 0. "FREE WITHIN 5 POINTS" if the whole pooled 95% interval lies inside [−5, +5]. Otherwise "NOT RESOLVED".
- Part 2: "SUPPORTED" if g ≥ 5, the interval excludes 0, and C's pooled loss vs A is ≤ 5. "CONTRADICTED" if g ≤ −5 and the interval excludes 0. Otherwise "NOT RESOLVED".

Whichever is chosen: the 48k column is the one the claim is about, so it is reported first. **Floor (pilot amendment, before freezing):** a context where arm A scores below 25% on the public main run is reported but left out of that model's verdict; see the pilot log. Optional arms E and P are descriptive only.

## Noise and sample size, in plain words
- Each arm answers 144 public questions: 48 at each length, on 6 documents. With the 4-document fallback it is 96 (32 per length).
- At temperature 1.0, two runs of the *same* setup will disagree on some answers. The noise run measures how many. If 15% of answers flip between any two arms by chance, the 95% interval on a difference is roughly **±11 points at one context length and ±6–7 points pooled**. At 25% flips it is about ±14 and ±8. Questions on one document move together, so the real intervals are somewhat wider. These are estimates; the pilot and the noise run give the real figure.
- So the test can catch a loss of about 10 points at 48k, or about 6–8 points pooled. **It cannot show that q8_0 is "free" to within 1–3 points.** "No loss found" must not be written up as "lossless". KL-style measurements (localbench, #23470) are far more sensitive. This test asks a different question: does the loss show up in answers?
- One model, one machine, one build. Different models disagree a lot (localbench: q8_0 KL 0.024 on Qwen 3.6 27B against 0.377 on Gemma 4 26B-A4B). The result applies to Muse Glimmer on Apple Metal and nothing wider.

## Time and cost (estimates, UNVERIFIED until the pilot)
Assumptions: prefill 55 tok/s and decode 5.5 tok/s on Metal (measured on this model and machine in the Bonsai calibration); about 600 generated tokens per question at "low" (a guess, not measured); for C/D, attention on the CPU at roughly 100 GFLOP/s effective (a guess).

| Arm(s) | Per set of 3 documents (4k + 16k + 48k) | 6 documents per context | 4 documents per context |
|---|---|---|---|
| A, B (GPU) | ~65 min (prefill ~22 min, 24 answers ~43 min) | ~6.5 h each (range 4–9) | ~4.3 h each |
| C, D (CPU attention expected) | ~140 min (48k prefill alone ~60 min) | ~14 h each (range 8–25) | ~9.3 h each |
| noise (A, 16k only) | — | ~2 h | ~1.3 h |
| pilot (preflight, calibration, timing) | — | ~4 h | ~4 h |
| **Total A–D + noise + pilot** | | **~47 h** | **~33 h** |
| optional E / P | | +6.5 h / +14 h | +4.3 h / +9.3 h |

On these estimates the 6-document plan is near the top of the 1–2 day budget, and the time rule will probably pick 4 documents. Cost: $0 (local), apart from electricity. Run under `caffeinate` (the Mac hibernated during the Bonsai run). The laptop must be free: the LFM2.5 test is also in preparation.

## Risks and answers
- **"Wrong model: nobody quantizes Glimmer's tiny cache."** True for memory: it saves 0.44 GB. But the claim is stated for any model, and sensitivity varies by model. The post says this is one model. Option: add Gemma 4 26B-A4B (arms A/B only, about 3–5 h), the model where KL says q8_0 is not free.
- **"Synthetic tasks aren't real use."** They copy the shapes of document QA: lookup, multi-hop, aggregation and finding the current version, with distractors and exact keys. Real documents could not be graded exactly and might be in training data. They do not cover coding agents at 64k–200k, where the HN reports come from. The post says so.
- **"Metal kernels differ from CUDA."** Yes. Metal dequantizes for prefill and uses quantized kernels for decode, and mixed K/V types run on the CPU. Results describe a Mac with stock llama.cpp 0.4.1. CUDA users with `GGML_CUDA_FA_ALL_QUANTS` run the mixed types on the GPU.
- **"C and D lost because of the CPU path, not because of the types."** Arm P (q8_0/q8_0 on the CPU) answers this if C/D lose more than B. Otherwise P is skipped.
- **"The rotation changes everything."** Rotation is on by default in this build and in what users run now. Older reports without rotation are not comparable. This is said in the post. Option (not planned): repeat D with `LLAMA_ATTN_ROT_DISABLE=1`.
- **"48k isn't long."** Correct; users run 64k–200k. 48k is what the budget allows with a 28B dense model on an M1 Pro. The trend across 4k → 16k → 48k is reported.
- **"Low reasoning hides or exaggerates the loss."** It is the same for every arm. Reasoning length per arm is reported (a damaged cache may also make the model think longer).
- **"Temperature 1.0 is noise."** Seeds are paired across arms and there is a noise run. Greedy decoding is an alternative (OPEN-QUESTIONS).
- **"Prompt reuse makes later questions different."** Every arm uses the same mechanism; `prompt_n` shows which requests re-read the document.
- **"Token counts are off."** They are measured by the real tokenizer in the preflight, with a fixed regeneration rule.
- **"Answer keys are wrong."** Every key is re-derived from the rendered text by an independent parser (`--check`). Items found broken are dropped for every arm, never re-keyed.
- **"A silent fallback or crash in one arm."** The build string and model hash are checked. Log warnings are captured. A prefill below 0.5× arm A is flagged. Server errors are counted.

## Prediction (author writes this before anything runs)

**What do you expect to happen?** (written by the author, 2026-10-25. The same day, before anything ran, the author added the last sentence and reworded the reason; the first wording is in this repository's history, commit 8d6a3d0.)

I expect q8_0 to lose about 2 points at 48k on Glimmer and about 6 points on Gemma 4 because Gemma 4's global attention layers use very large heads, and earlier divergence measurements found its cache far more sensitive to quantization than other models'. I expect compressing V to be better than compressing K.

**What result would prove you wrong?** (written by the author, 2026-10-25)

If q8_0 loses more than 8 points at 48k, or if the V-vs-K result goes the other way by more than 2 points.

## Frozen hashes

Frozen 2026-10-25 01:53 UTC, after the pilot and before any model saw a public or private question. The main run is `nohup caffeinate -is ./queue.sh 4` (time rule: 4 documents per context). Glimmer runs A, B, C, D and the noise run, then Gemma runs A and B. A change to any file below after this point would be a dated amendment, pushed before the affected run.

| File | sha256 |
|---|---|
| `tasks/public.jsonl` | `7d7274f658054674c0a1347a2918999c6333465cc8b832d9a9ea23e88df248d1` |
| `tasks/calibration.jsonl` | `e8f928f44b0a2acddd3517760b25418fdd9783c85ae5b8ea6170c62bd43f60c8` |
| `tasks/private.jsonl` (private, not published) | `5d4a3f455bbb0a32a033ea6c5279c71cc875c1626728e5a9bdb7e90e56e711a6` |
| `tasks/MANIFEST.json` | `9d5b4f3b40d5c5bcacf2c4de150cb1751953b9fe1948d8084963cfc530084ec3` |
| `gen_tasks.py` | `4d606bbebffdface4cca041f83d99680f5d907269af7e0120b0cfad511660e9d` |
| `private_theme.py` (private, not published) | `7ae8bd54dce56f45b697da2a6e580fa128dcb3e5ac5608e324405b6fbd34e271` |
| `harness/config.py` | `aa717682298bdd15395358465806975378d60398feacfe917faf54425e25a17b` |
| `harness/run.py` | `05e7c31c6313e37f125fecea2ae609a2067e2c395e8b7418cbb0183b9b3abbdb` |
| `harness/grade.py` | `fde27b377e1a07439b2d156f7d51cd1824a325697339757362d49faa4b3d564a` |
| `harness/analyze.py` | `b3f1cf4dfb37777472fb15416f802462aa9f7e8092907762cfe93fbf3ca7b05f` |
| `queue.sh` | `06ec7fb612c61ba081d18d26586c3800e8f0411997508527cd142f6fd5ec190b` |
