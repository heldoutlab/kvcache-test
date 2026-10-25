#!/opt/homebrew/bin/python3.12
"""Run one KV-cache arm on one task set against a local llama-server. Resumable: finished questions are skipped.
One server start per arm (the cache types are server flags); every arm sees identical prompts and seeds.

  run.py --arm A --set public --dry-run              # print the server command and first prompt; starts nothing
  run.py --preflight                                  # arm A server; real token counts of every document
  run.py --arm A --set calibration --run calibration  # difficulty + timing pilot (never scored)
  run.py --arm B --set public                         # main run, seed base 1001
  run.py --arm A --set public --run noise             # noise floor: arm A, other seed, NOISE_CONTEXTS only
  run.py --arm C --set public --docs 4                # time-rule fallback: first 4 documents per context
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
import grade


# ---------- tasks ----------

def load_set(name):
    path = os.path.join(cfg.TASKS_DIR, "%s.jsonl" % name)
    with open(path, "rb") as f:
        raw = f.read()
    manifest = json.load(open(os.path.join(cfg.TASKS_DIR, "MANIFEST.json")))
    want = manifest["files"]["%s.jsonl" % name]["sha256"]
    have = hashlib.sha256(raw).hexdigest()
    if have != want:
        raise SystemExit("tasks/%s.jsonl does not match tasks/MANIFEST.json" % name)
    return [json.loads(line) for line in raw.decode().splitlines() if line.strip()], have


def select_docs(docs, contexts, per_context):
    out = []
    for ctx in contexts:
        these = sorted((d for d in docs if d["ctx"] == ctx), key=lambda d: d["doc_id"])
        out += these[:per_context] if per_context else these
    return out


def messages_for(doc, question):
    # Document and question are separate user messages, so llama-server makes a context checkpoint
    # just before the question (Muse Glimmer has sliding-window layers) and the 7 later questions
    # on the same document reuse the cached document instead of re-reading it.
    return [{"role": "system", "content": cfg.SYSTEM_PROMPT},
            {"role": "user", "content": doc["text"]},
            {"role": "user", "content": question["question"]}]


# ---------- server ----------

def http(path, body=None, timeout=60):
    url = "http://127.0.0.1:%d%s" % (cfg.PORT, path)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def server_cmd(model, arm):
    m, a = cfg.MODELS[model], cfg.ARMS[arm]
    cmd = [cfg.SERVER_BIN, "-m", m["path"], "--host", "127.0.0.1", "--port", str(cfg.PORT),
           "-ngl", "99", "-fa", "on", "-c", str(cfg.CTX_SIZE), "-np", "1",
           "-b", str(cfg.N_BATCH), "-ub", str(cfg.N_UBATCH), "--jinja",
           "-ctk", a["ctk"], "-ctv", a["ctv"]]
    return cmd + list(a.get("extra_args", []))


def start_server(cmd, log_path):
    log = open(log_path, "a")
    log.write("\n# %s\n# %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(cmd)))
    log.flush()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit("llama-server exited during load; see " + log_path)
        try:
            if http("/health", timeout=5).get("status") == "ok":
                return proc
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(3)
    proc.terminate()
    raise SystemExit("llama-server did not become healthy in 15 minutes")


def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def rss_mib(pid):
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=10).stdout
        return round(int(out.strip()) / 1024, 1)
    except (ValueError, subprocess.SubprocessError):
        return None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def warnings_in_log(log_path, since_bytes):
    """Lines that suggest an op (e.g. flash attention) left the GPU, or that FA was turned off."""
    keys = ("assigned to device", "not supported", "flash_attn", "Flash Attention", "falling back", "CPU")
    with open(log_path, errors="replace") as f:
        f.seek(since_bytes)
        return [l.rstrip() for l in f if any(k in l for k in keys)][:50]


# ---------- one question ----------

def ask(model, msgs, seed):
    m = cfg.MODELS[model]
    body = {"model": "local", "messages": msgs, "seed": seed, "max_tokens": cfg.max_tokens(model), "stream": False,
            "cache_prompt": True}
    body.update(m["sampler"])
    body.update(json.loads(json.dumps(m.get("extra_body", {}))))
    t0 = time.time()
    try:
        resp = http("/v1/chat/completions", body, timeout=cfg.REQUEST_TIMEOUT_S)
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "error_body": e.read().decode(errors="replace")[:2000],
                "wall_s": round(time.time() - t0, 1)}
    except (urllib.error.URLError, OSError) as e:
        return {"http_error": -1, "error_body": repr(e)[:2000], "wall_s": round(time.time() - t0, 1)}
    choice = resp["choices"][0]
    msg = choice["message"]
    return {"content": msg.get("content") or "", "reasoning_chars": len(msg.get("reasoning_content") or ""),
            "reasoning": msg.get("reasoning_content") or "", "finish_reason": choice.get("finish_reason"),
            "usage": resp.get("usage", {}), "timings": resp.get("timings", {}), "wall_s": round(time.time() - t0, 1)}


# ---------- modes ----------

def preflight(model):
    """Real token counts (model tokenizer, via the server) for every document in every set."""
    out_dir = os.path.join(cfg.RESULTS_DIR, model, "preflight")
    os.makedirs(out_dir, exist_ok=True)
    proc = start_server(server_cmd(model, "A"), os.path.join(out_dir, "server.log"))
    rows = []
    try:
        for name in ("calibration", "public", "private"):
            docs, _ = load_set(name)
            for d in docs:
                q = d["questions"][0]
                prompt = http("/apply-template", {"messages": messages_for(d, q),
                                                  **cfg.MODELS[model].get("extra_body", {})})["prompt"]
                n_prompt = len(http("/tokenize", {"content": prompt, "add_special": False})["tokens"])
                n_doc = len(http("/tokenize", {"content": d["text"], "add_special": False})["tokens"])
                rows.append({"doc_id": d["doc_id"], "set": name, "ctx": d["ctx"], "chars": d["chars"],
                             "approx_tokens": d["approx_tokens"], "doc_tokens": n_doc, "prompt_tokens": n_prompt,
                             "chars_per_token": round(d["chars"] / n_doc, 3)})
                print("%s %s chars=%d approx=%d real=%d prompt=%d cpt=%.2f" % (
                    d["doc_id"], d["ctx"], d["chars"], d["approx_tokens"], n_doc, n_prompt, d["chars"] / n_doc), flush=True)
    finally:
        stop_server(proc)
    with open(os.path.join(out_dir, "tokens.json"), "w") as f:
        json.dump(rows, f, indent=2)
    bad = [r for r in rows if not (0.80 * cfg.NOMINAL_TOKENS[r["ctx"]] <= r["prompt_tokens"] <= cfg.NOMINAL_TOKENS[r["ctx"]])]
    cpt = sum(r["chars"] for r in rows) / sum(r["doc_tokens"] for r in rows)
    print("overall chars/token %.3f; %d of %d prompts outside [0.80, 1.00] x nominal" % (cpt, len(bad), len(rows)))
    if bad:
        print("PREREG rule: regenerate with  python3.12 gen_tasks.py --cpt %.2f  and re-run --preflight" % cpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="glimmer", choices=sorted(cfg.MODELS))
    ap.add_argument("--arm", choices=sorted(cfg.ARMS))
    ap.add_argument("--set", default="public", choices=["public", "private", "calibration"])
    ap.add_argument("--run", default="main", choices=sorted(cfg.RUN_SEEDS))
    ap.add_argument("--docs", type=int, default=0, help="documents per context (default: config)")
    ap.add_argument("--contexts", default="", help="comma list, e.g. 16k,48k (default: all; noise: NOISE_CONTEXTS)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N questions (smoke tests only)")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the server command and first prompt; run nothing")
    ap.add_argument("--skip-hash", action="store_true")
    ap.add_argument("--allow-other-build", action="store_true")
    args = ap.parse_args()

    if args.preflight:
        return preflight(args.model)
    if not args.arm:
        ap.error("--arm is required")

    docs, tasks_sha = load_set(args.set)
    if args.contexts:
        contexts = tuple(args.contexts.split(","))
    elif args.run == "noise":
        contexts = cfg.NOISE_CONTEXTS
    else:
        contexts = cfg.CONTEXTS
    per_ctx = args.docs or (0 if args.set == "calibration" else cfg.DOCS_PER_CONTEXT)
    docs = select_docs(docs, contexts, per_ctx)
    cmd = server_cmd(args.model, args.arm)

    if args.dry_run:
        print(" ".join(cmd))
        d = docs[0]
        q = d["questions"][0]
        body = {"messages": [dict(m, content=(m["content"][:300] + " ...[%d chars]" % len(m["content"]))
                                  if len(m["content"]) > 300 else m["content"]) for m in messages_for(d, q)],
                "seed": cfg.question_seed(args.run, q["qid"]), "max_tokens": cfg.max_tokens(args.model)}
        body.update(cfg.MODELS[args.model]["sampler"])
        body.update(cfg.MODELS[args.model].get("extra_body", {}))
        print(json.dumps(body, indent=2))
        print("%d documents, %d questions; analytic KV cache %.1f MiB" % (
            len(docs), sum(len(x["questions"]) for x in docs),
            cfg.kv_bytes(args.model, cfg.ARMS[args.arm]["ctk"], cfg.ARMS[args.arm]["ctv"]) / 2 ** 20))
        return

    out_dir = os.path.join(cfg.RESULTS_DIR, args.model, args.set, args.run)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "%s.jsonl" % args.arm)
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {json.loads(l)["qid"] for l in f if l.strip()}
    todo = [(d, q) for d in docs for q in d["questions"] if q["qid"] not in done]
    total_q = sum(len(d["questions"]) for d in docs)
    print("%s %s %s/%s: %d questions, %d already done" % (args.model, args.arm, args.set, args.run, total_q,
                                                         total_q - len(todo)), flush=True)
    if not todo:
        return

    m = cfg.MODELS[args.model]
    manifest = {"model": args.model, "arm": args.arm, "arm_label": cfg.ARMS[args.arm]["label"], "set": args.set,
                "run": args.run, "contexts": contexts, "docs_per_context": per_ctx, "tasks_sha256": tasks_sha,
                "server_cmd": cmd, "kv_bytes_analytic": cfg.kv_bytes(args.model, cfg.ARMS[args.arm]["ctk"],
                                                                     cfg.ARMS[args.arm]["ctv"]),
                "max_tokens": cfg.max_tokens(args.model), "sampler": m["sampler"], "extra_body": m.get("extra_body", {}),
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if not args.skip_hash:
        manifest["model_sha256"] = sha256_file(m["path"])
        if manifest["model_sha256"] != m["sha256"]:
            raise SystemExit("model file hash does not match the pre-registered hash: " + m["path"])
        manifest["server_sha256"] = sha256_file(os.path.realpath(cfg.SERVER_BIN))
    log_path = os.path.join(out_dir, "%s.server.log" % args.arm)
    log_start = os.path.getsize(log_path) if os.path.exists(log_path) else 0
    proc = start_server(cmd, log_path)
    try:
        props = http("/props")
        manifest["build_info"] = props.get("build_info")
        manifest["n_ctx"] = props.get("default_generation_settings", {}).get("n_ctx")
        if manifest["build_info"] != cfg.EXPECTED_BUILD and not args.allow_other_build:
            raise SystemExit("unexpected llama.cpp build %r (expected %s)" % (manifest["build_info"], cfg.EXPECTED_BUILD))
        manifest["log_warnings"] = warnings_in_log(log_path, log_start)
        manifest["rss_mib_after_load"] = rss_mib(proc.pid)
        with open(os.path.join(out_dir, "%s.manifest.jsonl" % args.arm), "a") as f:
            f.write(json.dumps(manifest) + "\n")

        k = 0
        for d, q in todo:
            k += 1
            seed = cfg.question_seed(args.run, q["qid"])
            turn = ask(args.model, messages_for(d, q), seed)
            rec = {"qid": q["qid"], "doc_id": d["doc_id"], "ctx": d["ctx"], "family": q["family"], "arm": args.arm,
                   "model": args.model, "set": args.set, "run": args.run, "seed": seed,
                   "first_on_doc": q is d["questions"][0], "turn": turn,
                   "server_error": "http_error" in turn,
                   "truncated": turn.get("finish_reason") == "length"}
            if rec["server_error"]:
                rec.update(correct=False, grader={"detail": "server error"})
            else:
                rec["correct"], rec["grader"] = grade.grade(q, turn["content"])
            if q is d["questions"][-1]:
                rec["rss_mib"] = rss_mib(proc.pid)
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            t = turn.get("timings", {})
            print("[%d/%d] %s correct=%s gen=%s prompt_n=%s pp=%.1f t/s tg=%.1f t/s %.0fs%s" % (
                k, len(todo), q["qid"], rec["correct"], t.get("predicted_n"), t.get("prompt_n"),
                t.get("prompt_per_second") or 0, t.get("predicted_per_second") or 0, turn.get("wall_s", 0),
                " TRUNCATED" if rec["truncated"] else ""), flush=True)
            if args.limit and k >= args.limit:
                break
    finally:
        stop_server(proc)


if __name__ == "__main__":
    main()
