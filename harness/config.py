"""Fixed settings for the KV-cache test. Frozen by PREREG.md before the main run.

Nothing here has been run yet (2026-10-22). Values marked PILOT are confirmed or adjusted by the
calibration/timing pilot BEFORE freezing, under the rules written in PREREG.md.
"""
import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS_DIR = os.path.join(ROOT, "tasks")
RESULTS_DIR = os.path.join(ROOT, "results")
PYTHON = "/opt/homebrew/bin/python3.12"

# Stock Homebrew llama.cpp, not upgraded for this test.
SERVER_BIN = "/opt/homebrew/bin/llama-server"
EXPECTED_BUILD = "b10964-b29c606e2"  # `llama-server --version`: 0.4.1 (build 10964, commit b29c606e2)

PORT = 8793
CTX_SIZE = 65536        # one size for every arm and every context length; the longest prompt is <= 49,152
N_BATCH = 2048          # llama.cpp defaults, pinned explicitly
N_UBATCH = 512
MAX_TOKENS = 4096       # output cap; a model gets 8192 ("max_tokens" in MODELS) if > 5% of its arm-A calibration replies hit 4096
REQUEST_TIMEOUT_S = 2 * 3600

SYSTEM_PROMPT = ("You answer questions about the document the user provides. "
                 "Use only the information in that document.")

MODELS = {
    "glimmer": {
        "label": "Muse Glimmer 30B KQuant-17GB Q4_K_M (16.76 GB)",
        "path": "/Users/admin/models/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
        "repo": "meta-models/Muse-Glimmer-30B-GGUF",
        "size": 16756683904,
        "sha256": "4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e",  # HF LFS oid, 2026-10-22
        # Model card "Best Practices": temperature 1.0, top_p 0.95, top_k 64 (no min_p given; llama.cpp default kept).
        "sampler": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.05},
        # Reasoning cannot be switched off (card); "low" keeps the run affordable. Same for every arm.
        "extra_body": {"chat_template_kwargs": {"reasoning_strength": "low"}},
        # From GGUF metadata (read 2026-10-22): 52 layers, 39 sliding-window (2048) + 13 global, 2 KV heads, head 128.
        "kv_layout": [{"layers": 13, "window": None, "n_head_kv": 2, "dk": 128, "dv": 128},
                      {"layers": 39, "window": 2048, "n_head_kv": 2, "dk": 128, "dv": 128}],
    },
    # Second model, added by the author 2026-10-25 (arms A and B only). KV layout from google/gemma-4-26B-A4B-it
    # config.json (read 2026-10-22). Sampler from its model card and generation_config.json (read 2026-10-25):
    # temperature 1.0, top_p 0.95, top_k 64; min_p not given, llama.cpp default 0.05 kept (as for Glimmer).
    # Thinking: off, the card's default (it is enabled only by a <|think|> token in the system prompt).
    "gemma4": {
        "label": "Gemma 4 26B-A4B it UD-Q4_K_M (16.95 GB)",
        "path": "/Users/admin/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
        "repo": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "size": 16947541728,
        "sha256": "f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f",
        "sampler": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.05},
        # llama-server turns thinking on by default; the card's default (and PREREG) is off
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "max_tokens": 8192,     # pilot: 1 of 18 arm-A calibration replies hit 4096 (5.6% > 5%)
        "kv_layout": [{"layers": 5, "window": None, "n_head_kv": 2, "dk": 512, "dv": 512},
                      {"layers": 25, "window": 1024, "n_head_kv": 8, "dk": 256, "dv": 256}],
    },
}

ARMS = {
    "A": {"label": "f16 K / f16 V (reference)", "ctk": "f16", "ctv": "f16"},
    "B": {"label": "q8_0 K / q8_0 V", "ctk": "q8_0", "ctv": "q8_0"},
    "C": {"label": "q8_0 K / q4_0 V (the advice)", "ctk": "q8_0", "ctv": "q4_0"},
    "D": {"label": "q4_0 K / q8_0 V (mirror of the advice)", "ctk": "q4_0", "ctv": "q8_0"},
    "E": {"label": "q4_0 K / q4_0 V (optional)", "ctk": "q4_0", "ctv": "q4_0"},
    # Optional path control: uniform q8_0 with the KV cache and attention kept on the CPU. On this build, arms
    # C and D (K type != V type) cannot use the Metal flash-attention kernel and are expected to fall back to
    # the CPU; P separates "CPU attention path" from "mixed types" if C/D lose accuracy.
    "P": {"label": "q8_0 K / q8_0 V, attention on CPU (-nkvo; optional control)", "ctk": "q8_0", "ctv": "q8_0",
          "extra_args": ["-nkvo"]},
}
MAIN_ARMS = ("A", "B", "C", "D")
OPTIONAL_ARMS = ("E", "P")

CONTEXTS = ("4k", "16k", "48k")
NOMINAL_TOKENS = {"4k": 4096, "16k": 16384, "48k": 49152}
DOCS_PER_CONTEXT = 6           # main run; the generator makes 6 per context per set
FALLBACK_DOCS_PER_CONTEXT = 4  # time rule in PREREG.md: first 4 documents per context, all arms
NOISE_CONTEXTS = ("16k",)      # arm A re-run with a different seed on these contexts

RUN_SEEDS = {"main": 1001, "noise": 2002, "calibration": 7}

# Pass/fail rule chosen by the author in PREREG.md ("1", "2" or "3"). analyze.py prints no verdict until set.
RULE = "1"  # author's choice, 2026-10-25
FLOOR_ACC = 25.0       # arm A accuracy (%) below which a context cannot show a loss; excluded from the verdict
BOOT = 10000

# bytes per element of each cache type (ggml block sizes)
TYPE_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32, "q4_1": 20 / 32,
              "q5_0": 22 / 32, "q5_1": 24 / 32, "iq4_nl": 18 / 32}


def question_seed(run, qid):
    """Same seed for a question in every arm (common random numbers); different per run."""
    h = hashlib.sha256(("%d|%s" % (RUN_SEEDS[run], qid)).encode()).digest()
    return int.from_bytes(h[:4], "big") % (2 ** 31 - 1)


def kv_bytes(model, ctk, ctv, n_ctx=CTX_SIZE, n_ubatch=N_UBATCH):
    """Analytic KV-cache size for one sequence, as llama.cpp allocates it (SWA cache = window + ubatch, padded)."""
    total = 0.0
    for part in MODELS[model]["kv_layout"]:
        cells = n_ctx if part["window"] is None else min(n_ctx, -(-(part["window"] + n_ubatch) // 256) * 256)
        per_cell = part["n_head_kv"] * (part["dk"] * TYPE_BYTES[ctk] + part["dv"] * TYPE_BYTES[ctv])
        total += part["layers"] * cells * per_cell
    return total


def max_tokens(model):
    return MODELS[model].get("max_tokens", MAX_TOKENS)
