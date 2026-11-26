#!/usr/bin/env python3
"""Noise-controlled bracketed A/B benchmark for PolyStrata (strata-poly) on Inkling 2 Mini.

Measures on identical UD-Q4_K_XL weights with isolated per-arm --expert-profile:
  - tg_steady:  6 fresh coding prompts x 256 tok (mean of last 4 steady probes)
  - edit_tg:    2 code-edit prompts x 256 tok (exercises --window 6 --lookup 3)
  - pp2k/8k/20k: unique-prefix prefill speeds (tok/s)
  - post20k_tg: 128-token decode immediately after 20K prefill
  - load_s, vram_peak_mib, experts_gpu

Example:
  python3 .agents/skills/inkling-refresh/scripts/bench-inkling-ab.py \
    --arm base_start:vendor/polystrata/build/strata-poly \
    --arm v5:/tmp/strata-poly-v5 \
    --arm v5_pr320:/tmp/strata-poly-v5-pr320:--prompt-batch=auto \
    --arm base_end:vendor/polystrata/build/strata-poly
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL = Path("/mnt/dev/models/unsloth/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf")
DEFAULT_TOKENIZER = Path("/mnt/dev/models/unsloth/Inkling-2-Mini-GGUF/poly-pack/tokenizer")
RESULTS_JSONL = ROOT / "logs" / "results" / "inkling-refresh-ab.jsonl"
LOG_DIR = ROOT / "logs" / "inkling-refresh-ab"
TOK_CACHE = Path("/tmp/inkling2_refresh_ab_tokens.json")

DEFAULT_ARGS = [
    "--max-context", "262144",
    "--kv", "fp8",
    "--vram-pool",
    "--loop-bf16",
    "--prompt-batch", "2048",
    "--lock-experts",
    "--threads", "12",
    "--window", "6",
    "--lookup", "3",
]

DEFAULT_ENV = {
    "STRATA_PREFILL_NVFP4": "w4a8",
    "STRATA_INK_FAST_CTRL": "1",
    "STRATA_INK_PRESTAGE": "0",
}

CODE_CHUNK = (
    "def process_record(idx: int, payload: bytes, state: dict[int, int]) -> int:\n"
    "    acc = state.get(idx, 0) + len(payload)\n"
    "    if acc % 7 == 0:\n"
    "        state[idx] = acc ^ 0x5A3C\n"
    "    return acc\n\n"
)

EDIT_SOURCE = """```python
class RingBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._buf = [None] * capacity
        self._cap = capacity
        self._head = 0
        self._tail = 0
        self._size = 0

    def push(self, item) -> bool:
        if self._size == self._cap:
            return False
        self._buf[self._tail] = item
        self._tail = (self._tail + 1) % self._cap
        self._size += 1
        return True

    def pop(self):
        if self._size == 0:
            raise IndexError("pop from empty buffer")
        val = self._buf[self._head]
        self._buf[self._head] = None
        self._head = (self._head + 1) % self._cap
        self._size -= 1
        return val
```"""


def load_or_build_tokens(tokenizer_dir: Path) -> dict[str, list[list[int]]]:
    if TOK_CACHE.exists():
        data = json.loads(TOK_CACHE.read_text(encoding="utf-8"))
        if all(k in data for k in ("tg", "edit", "pp2k", "pp8k", "pp20k")):
            return data

    # Use Strata venv's tokenizers / transformers package
    strata_py = ROOT / "vendor" / "strata" / ".venv" / "bin" / "python"
    py = str(strata_py) if strata_py.exists() else sys.executable

    prompts = {
        "tg": [
            f"Write a clean, well-typed Python implementation of an async rate limiter and LRU cache with unit tests (variant {i}, seed {i * 41})."
            for i in range(6)
        ],
        "edit": [
            f"Refactor the RingBuffer class below to add `peek()`, `__len__()`, and an optional `overwrite=False` parameter on `push()`, keeping the rest verbatim (edit pass {i}):\n\n{EDIT_SOURCE}"
            for i in range(2)
        ],
        "pp2k": [
            f"# pp2k-run-{k}\n" + CODE_CHUNK * 32 + "\nSummarize the behavior of process_record in one concise sentence."
            for k in range(2)
        ],
        "pp8k": [
            f"# pp8k-run-{k}\n" + CODE_CHUNK * 128 + "\nSummarize the behavior of process_record in one concise sentence."
            for k in range(2)
        ],
        "pp20k": [
            f"# pp20k-run-0\n" + CODE_CHUNK * 320 + "\nSummarize the behavior of process_record in one concise sentence."
        ],
    }

    tools_dir = ROOT / "vendor" / "polystrata" / "terminal" / "tools"
    helper = (
        "import json, pathlib, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import strata_tokenizer as ST\n"
        "tok = ST.Tokenizer.from_dir(pathlib.Path(sys.argv[2]))\n"
        "prompts = json.loads(sys.stdin.read())\n"
        "print(json.dumps({k: [list(tok.encode(t)) for t in v] for k, v in prompts.items()}))\n"
    )
    raw = subprocess.check_output(
        [py, "-c", helper, str(tools_dir), str(tokenizer_dir)],
        input=json.dumps(prompts),
        text=True,
    )
    data = json.loads(raw)
    TOK_CACHE.write_text(json.dumps(data), encoding="utf-8")
    print(
        f"Prepared tokens: tg={len(data['tg'][0])}, edit={len(data['edit'][0])}, "
        f"pp2k={len(data['pp2k'][0])}, pp8k={len(data['pp8k'][0])}, pp20k={len(data['pp20k'][0])}",
        flush=True,
    )
    return data


def get_vram_mib() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(out.strip().splitlines()[0])


def capture_sys_frontmatter() -> dict[str, str]:
    env_sh = ROOT / "bench-models" / "bench-env.sh"
    if not env_sh.exists():
        return {}
    try:
        out = subprocess.check_output(["bash", str(env_sh), "--kv"], text=True, stderr=subprocess.DEVNULL)
        meta: dict[str, str] = {}
        for line in out.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
        return meta
    except Exception:
        return {}


def parse_arm_spec(spec: str) -> dict[str, object]:
    """Parse name:binary[:flag=val,...][:ENV=VAL,...]."""
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid --arm spec (expected name:binary[:flags][:env]): {spec}")
    name = parts[0]
    binary = Path(parts[1])
    if not binary.is_absolute():
        binary = (ROOT / binary).resolve()
    overrides: list[str] = []
    env_overrides: dict[str, str] = {}
    if len(parts) >= 3 and parts[2]:
        for item in parts[2].split(","):
            if "=" in item:
                k, v = item.split("=", 1)
                overrides.extend([k, v])
            else:
                overrides.append(item)
    if len(parts) >= 4 and parts[3]:
        for item in parts[3].split(","):
            k, v = item.split("=", 1)
            env_overrides[k] = v
    return {"name": name, "binary": binary, "overrides": overrides, "env": env_overrides}


def merge_cli_args(base: list[str], overrides: list[str]) -> list[str]:
    out = list(base)
    i = 0
    while i < len(overrides):
        flag = overrides[i]
        has_val = i + 1 < len(overrides) and not overrides[i + 1].startswith("--")
        val = overrides[i + 1] if has_val else None
        if flag in out:
            idx = out.index(flag)
            if has_val and idx + 1 < len(out) and not out[idx + 1].startswith("--"):
                out[idx + 1] = str(val)
            elif not has_val:
                pass
        else:
            out.append(flag)
            if has_val:
                out.append(str(val))
        i += 2 if has_val else 1
    return out


def parse_done_metrics(done_line: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for pat, key in (
        (r"prefill[= ]([0-9.]+)\s*tok/s", "pp_tps"),
        (r"prompt=([0-9.]+)\s*tok/s", "pp_tps"),
        (r"read=([0-9.]+)\s*t/s", "pp_tps"),
        (r"decode[= ]([0-9.]+)\s*tok/s", "tg_tps"),
        (r"gen=([0-9.]+)\s*tok/s", "tg_tps"),
        (r"write=([0-9.]+)\s*t/s", "tg_tps"),
        (r"p_ms=([0-9.]+)", "p_ms"),
        (r"g_ms=([0-9.]+)", "g_ms"),
        (r"prompt_ms=([0-9.]+)", "p_ms"),
        (r"gen_ms=([0-9.]+)", "g_ms"),
        (r"gen_tok=(\d+)", "gen_tok"),
        (r"prompt_tok=(\d+)", "prompt_tok"),
    ):
        m = re.search(pat, done_line)
        if m and key not in out:
            out[key] = float(m.group(1))
    return out


def run_arm(
    arm: dict[str, object],
    model: Path,
    tokens: dict[str, list[list[int]]],
    stamp: str,
) -> dict[str, object]:
    name = str(arm["name"])
    binary = Path(arm["binary"])
    profile_path = Path(f"/tmp/inkling-ab-prof-{stamp}-{name}.bin")
    if profile_path.exists():
        profile_path.unlink()

    cli_args = merge_cli_args(DEFAULT_ARGS, list(arm["overrides"]))
    cli_args.extend(["--expert-profile", str(profile_path)])

    env = os.environ.copy()
    env.update(DEFAULT_ENV)
    env.update(dict(arm["env"]))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{stamp}-{name}.log"
    cmd = ["taskset", "-c", "0-11", str(binary), "--model", str(model), "--serve"] + cli_args

    print(f"\n=== ARM: {name} ===", flush=True)
    print(f"  bin:  {binary}", flush=True)
    print(f"  args: {' '.join(cli_args)}", flush=True)

    errf = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=errf,
        text=True,
        bufsize=1,
        env=env,
    )
    res: dict[str, object] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "arm": name,
        "binary": str(binary),
        "args": cli_args,
        "env": {k: env[k] for k in set(DEFAULT_ENV) | set(arm["env"])},
    }
    try:
        t0 = time.time()
        ready_line = None
        while time.time() - t0 < 180:
            line = proc.stdout.readline()
            if not line:
                break
            if line.startswith("READY"):
                ready_line = line.strip()
                break
        if not ready_line:
            errf.close()
            tail = log_path.read_text(encoding="utf-8")[-800:]
            print(f"  FAILED START:\n{tail}", flush=True)
            res["error"] = "start_failed"
            return res

        load_s = round(time.time() - t0, 2)
        v_load = get_vram_mib()
        v_peak = v_load
        print(f"  {ready_line} ({load_s}s, vram_load={v_load} MiB)", flush=True)

        def run_gen(tok_ids: list[int], max_new: int) -> dict[str, float]:
            nonlocal v_peak
            t_req = time.time()
            proc.stdin.write(f"GEN {max_new} temp=0.0 " + ",".join(str(x) for x in tok_ids) + "\n")
            proc.stdin.flush()
            tok_count = 0
            t_first = None
            done_line = ""
            while True:
                l = proc.stdout.readline()
                if not l:
                    raise RuntimeError("EOF from strata-poly")
                if l.startswith("T "):
                    if t_first is None:
                        t_first = time.time()
                    tok_count += 1
                elif l.startswith("DONE"):
                    done_line = l.strip()
                    break
                elif l.startswith("ERR"):
                    raise RuntimeError(l.strip())
            t_end = time.time()
            v_now = get_vram_mib()
            v_peak = max(v_peak, v_now)
            parsed = parse_done_metrics(done_line)
            # Fall back to wall-clock measurements when DONE omits fields
            if "pp_tps" not in parsed:
                if "p_ms" in parsed and parsed["p_ms"] > 0:
                    parsed["pp_tps"] = len(tok_ids) / (parsed["p_ms"] / 1000.0)
                elif t_first is not None and t_first > t_req:
                    parsed["pp_tps"] = len(tok_ids) / (t_first - t_req)
            if "tg_tps" not in parsed:
                if "g_ms" in parsed and parsed["g_ms"] > 0:
                    n_g = parsed.get("gen_tok", float(tok_count))
                    parsed["tg_tps"] = n_g / (parsed["g_ms"] / 1000.0)
                elif t_first is not None and t_end > t_first and tok_count > 1:
                    parsed["tg_tps"] = (tok_count - 1) / (t_end - t_first)
            parsed["n_tok"] = float(tok_count)
            parsed["done_raw"] = done_line
            return parsed

        # 1. Fresh coding decode (6 probes, steady = last 4)
        tg_vals: list[float] = []
        for idx, pt in enumerate(tokens["tg"]):
            m = run_gen(pt, 256)
            tg_vals.append(round(m.get("tg_tps", 0.0), 2))
            if idx == 0:
                print(f"  [sample DONE] {m['done_raw']}", flush=True)
        tg_steady = round(statistics.mean(tg_vals[2:]), 2)
        print(f"  tg256 all={tg_vals} -> steady(last4)={tg_steady} tok/s", flush=True)

        # 2. Code-edit decode (2 probes, exercises lookup)
        edit_vals: list[float] = []
        for pt in tokens["edit"]:
            m = run_gen(pt, 256)
            edit_vals.append(round(m.get("tg_tps", 0.0), 2))
        edit_mean = round(statistics.mean(edit_vals), 2)
        print(f"  edit256 all={edit_vals} -> mean={edit_mean} tok/s", flush=True)

        # 3. Prefill pp2k, pp8k, pp20k + post20k decode
        pp2k_vals = [round(run_gen(pt, 16).get("pp_tps", 0.0), 1) for pt in tokens["pp2k"]]
        pp8k_vals = [round(run_gen(pt, 16).get("pp_tps", 0.0), 1) for pt in tokens["pp8k"]]
        m20 = run_gen(tokens["pp20k"][0], 128)
        pp20k = round(m20.get("pp_tps", 0.0), 1)
        post20k_tg = round(m20.get("tg_tps", 0.0), 2)

        pp2k = round(statistics.mean(pp2k_vals), 1)
        pp8k = round(statistics.mean(pp8k_vals), 1)
        print(
            f"  pp2k={pp2k} ({pp2k_vals}) | pp8k={pp8k} ({pp8k_vals}) | "
            f"pp20k={pp20k} | post20k_tg={post20k_tg} | vram_peak={v_peak} MiB",
            flush=True,
        )

        res.update(
            {
                "load_s": load_s,
                "tg_all": tg_vals,
                "tg_steady": tg_steady,
                "edit_all": edit_vals,
                "edit_tg": edit_mean,
                "pp2k_all": pp2k_vals,
                "pp2k": pp2k,
                "pp8k_all": pp8k_vals,
                "pp8k": pp8k,
                "pp20k": pp20k,
                "post20k_tg": post20k_tg,
                "vram_load_mib": v_load,
                "vram_peak_mib": v_peak,
            }
        )
    finally:
        try:
            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
            proc.wait(timeout=5)
        errf.close()
        if profile_path.exists():
            profile_path.unlink()

    # Extract resident GPU experts from stderr log
    log_txt = log_path.read_text(encoding="utf-8")
    m_exp = re.findall(r"(\d+)\s+of\s+(\d+)\s+(?:routed\s+)?experts|expert cache[: ]+(\d+)\s+slots", log_txt)
    if m_exp:
        last = m_exp[-1]
        res["experts_gpu"] = int(last[0] or last[2])
    return res


def main() -> int:
    parser = argparse.ArgumentParser(description="Bracketed A/B benchmark for PolyStrata Inkling 2 Mini")
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="Arm spec: name:binary[:--flag=val,...][:ENV=VAL,...]",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    args = parser.parse_args()

    arms = [parse_arm_spec(s) for s in args.arm]
    tokens = load_or_build_tokens(args.tokenizer)
    sys_meta = capture_sys_frontmatter()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    rows: list[dict[str, object]] = []
    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        row = run_arm(arm, args.model, tokens, stamp)
        row["sys"] = sys_meta
        rows.append(row)
        with open(RESULTS_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    print("\n=== SUMMARY TABLE ===", flush=True)
    print(f"{'Arm':18s} | {'tg_std':>7s} | {'edit_tg':>7s} | {'pp2k':>7s} | {'pp8k':>7s} | {'pp20k':>7s} | {'p20_tg':>6s} | {'VRAM':>6s} | {'Exp':>5s}")
    print("-" * 92)
    for r in rows:
        if "error" in r:
            print(f"{r['arm']:18s} | ERROR: {r['error']}")
            continue
        print(
            f"{r['arm']:18s} | {r['tg_steady']:7.2f} | {r['edit_tg']:7.2f} | "
            f"{r['pp2k']:7.1f} | {r['pp8k']:7.1f} | {r['pp20k']:7.1f} | "
            f"{r['post20k_tg']:6.2f} | {r['vram_peak_mib']:6d} | {str(r.get('experts_gpu', '-')):>5s}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
