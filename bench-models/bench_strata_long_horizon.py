#!/usr/bin/env python3
"""
bench_strata_long_horizon.py
Evaluates Strata (Qwen3.8-Flash-Next Plat) across context horizons:
1k, 4k, 8k, 16k, 32k, 48k, 60k tokens.
Measures:
- Prompt Processing (PP) tok/s & TTFT
- Generation (TG) tok/s
- Draft Acceptance Rate (MTP)
- Context needle recall & coherence
- System memory & VRAM delta
"""

import sys
import os
import time
import json
import urllib.request
import subprocess

API_URL = "http://127.0.0.1:8080/v1/chat/completions"
API_KEY = "l3ms-615d796a0853c66da3d5e59be648e4a0203f5d8267f8a0b4"
MODEL = "strata"

def get_vram_used_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return 0

# Sample technical text filler to construct realistic prompts
FILLER_PARAGRAPHS = [
    """The memory hierarchy of modern computing architectures enforces strict trade-offs between latency, bandwidth, and capacity. High-bandwidth memory (HBM) and GDDR6X offer terabytes per second of throughput at the expense of capacity and cost. Standard DDR5 SDRAM provides hundreds of gigabytes of capacity but tops out around 60 to 90 GB/s on dual-channel consumer platforms. Meanwhile, PCIe Gen 4 and Gen 5 NVMe SSDs provide massive multi-terabyte persistent capacity with access latencies measured in tens of microseconds.""",
    """In Mixture of Experts (MoE) neural network inference, the parameter footprint often scales well into hundreds of billions of weights, yet only a sparse subset of expert feed-forward layers are activated per token. When these models exceed physical GPU VRAM, conventional runtimes partition the network across discrete layers, executing host-device DMA transfers token by token. If expert routing is non-uniform, this layer-by-layer offloading introduces high bus contention and stalls the compute pipeline while waiting on system DRAM bandwidth.""",
    """Speculative decoding leverages a smaller draft model or specialized auxiliary heads to predict multiple subsequent tokens concurrently. These draft tokens are evaluated in parallel by the primary target model in a single forward verification pass. If the draft accuracy remains sufficiently high, the effective token generation rate increases substantially, overcoming the memory bandwidth bottleneck that traditionally caps single-batch autoregressive generation.""",
    """Kernel fusion represents another critical optimization vector in GPU execution graphs. By fusing elementwise operations such as Root Mean Square Normalization (RMSNorm) and tensor scaling directly with subsequent matrix multiplications or rotary position embeddings (RoPE), runtime overhead from kernel dispatch latency and intermediate global memory round-trips is dramatically minimized.""",
    """Paged attention and virtual memory paging abstractions permit dynamic allocation of Key-Value (KV) cache pages without requiring contiguous physical memory reservations. Quantizing the KV cache to 8-bit (k8) or 4-bit (v4) precision reduces memory footprint by up to 75% while preserving attention fidelity over context windows exceeding sixty-four thousand tokens."""
]

def make_context(target_tokens, needle_passcode):
    # Base text expansion
    words_per_para = sum(len(p.split()) for p in FILLER_PARAGRAPHS)
    # Approx 1.33 tokens per word
    paras_needed = int(target_tokens / (words_per_para * 1.33)) + 1
    
    body = []
    midpoint = paras_needed // 2
    for i in range(paras_needed):
        idx = i % len(FILLER_PARAGRAPHS)
        para = f"[Section {i+1}] " + FILLER_PARAGRAPHS[idx]
        if i == midpoint:
            para += f"\n\nCRITICAL SYSTEM NOTIFICATION: The verified security token for this deployment is {needle_passcode}. Remember this exact security token.\n\n"
        body.append(para)
    
    text = "\n\n".join(body)
    
    prompt = (
        f"You are evaluating a long-context architecture document. Read the following text carefully:\n\n"
        f"{text}\n\n"
        f"Based on the text above, answer two questions:\n"
        f"1. What is the verified security token mentioned in the text? State it exactly.\n"
        f"2. Write a concise 2-paragraph technical explanation describing how speculative decoding overcomes DRAM memory bandwidth bottlenecks in offloaded MoE models."
    )
    return prompt

def run_bench():
    # Targets in tokens (approximate prompt sizes)
    test_sizes = [1024, 4096, 8192, 16384, 32768, 48000, 60000]
    results = []

    print(f"{'Target Ctx':>10} | {'Prompt Tok':>10} | {'Prefill t/s':>11} | {'Gen Tok':>8} | {'Decode t/s':>10} | {'MTP Draft %':>11} | {'Needle Found':>12} | {'VRAM (MB)':>9}")
    print("-" * 105)

    for target in test_sizes:
        needle = f"OBSIDIAN_PHOENIX_{target}_SECURE"
        prompt = make_context(target, needle)
        vram_before = get_vram_used_mb()

        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.0
        }
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            data=data
        )

        t_start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                res = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"Error running context size {target}: {e}")
            continue
        
        t_total = time.time() - t_start
        vram_after = get_vram_used_mb()

        usage = res.get("usage", {})
        timings = res.get("timings", {})
        choices = res.get("choices", [{}])
        msg = choices[0].get("message", {})
        content = (msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")

        prompt_tok = usage.get("prompt_tokens", 0)
        comp_tok = usage.get("completion_tokens", 0)
        
        pp_speed = timings.get("prompt_per_second", 0.0)
        tg_speed = timings.get("predicted_per_second", 0.0)
        
        draft_n = timings.get("draft_n", 0)
        draft_acc = timings.get("draft_n_accepted", 0)
        draft_pct = (draft_acc / draft_n * 100.0) if draft_n > 0 else 0.0

        needle_found = needle in content

        res_item = {
            "target_ctx": target,
            "prompt_tokens": prompt_tok,
            "completion_tokens": comp_tok,
            "prefill_ts": pp_speed,
            "prompt_ms": timings.get("prompt_ms", 0),
            "decode_ts": tg_speed,
            "decode_ms": timings.get("predicted_ms", 0),
            "draft_n": draft_n,
            "draft_accepted": draft_acc,
            "draft_acc_pct": draft_pct,
            "needle_found": needle_found,
            "vram_mb": vram_after,
            "wall_sec": t_total,
            "sample_content": content[:160].replace("\n", " ")
        }
        results.append(res_item)

        print(f"{target:>10} | {prompt_tok:>10} | {pp_speed:>11.1f} | {comp_tok:>8} | {tg_speed:>10.1f} | {draft_pct:>10.1f}% | {str(needle_found):>12} | {vram_after:>9}")

    print("\nBenchmark Complete. Summary JSON:")
    print(json.dumps(results, indent=2))
    
    with open("/home/kchauhan/repos/l3ms/vendor/strata/long_horizon_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_bench()
