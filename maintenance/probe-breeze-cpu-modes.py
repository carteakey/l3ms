#!/usr/bin/env python3
"""One explicit cached-weight CPU run; no services, downloads or retries.

Retains inputs, executed script/spec, process receipt and WAVs in a private
fixture. Streaming acceptance is chunk retrieval after completion, not playback.
"""
import argparse
import array
import json
import os
from pathlib import Path
import runpy
import shutil
import time
import wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("clon", "streaming"))
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--audio-cpp-root", required=True, type=Path)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument("--voice-reference", type=Path)
    parser.add_argument("--reference-text")
    args = parser.parse_args()
    if args.mode == "clon" and (not args.voice_reference or not args.reference_text):
        parser.error("cloning requires reference audio and its nonempty transcript")
    repo = Path(__file__).resolve().parents[1]
    fixture = args.fixture.resolve()
    metadata = fixture.stat()
    assert metadata.st_uid == os.geteuid() and metadata.st_mode & 0o777 == 0o700
    case = fixture / args.mode
    case.mkdir(mode=0o700)  # Refuse repeat dispatch into an existing case.
    shutil.copyfile(__file__, case / "executed-probe.py")
    output = case / "output"
    output.mkdir(mode=0o700)
    payload = {"text": "Hello from the local model stack."}
    if args.mode == "clon":
        shutil.copyfile(args.voice_reference, case / "reference.wav")
        payload.update(voice_reference=str(case / "reference.wav"),
                       reference_text=args.reference_text)
    else:
        payload["mode"] = "streaming"
    (case / "input.json").write_text(json.dumps(payload))
    shutil.copyfile(repo / "media-models/generate-breeze-tts-2.sh",
                    case / "executed-wrapper.sh")
    shutil.copyfile(args.audio_cpp_root / "model_specs/breeze_tts.json",
                    case / "executed-model-spec.json")
    environment = {
        "PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "LANG": "C.UTF-8",
        "L3MS_ROOT": str(repo), "L3MS_DATA_DIR": str(case / "data"),
        "L3MS_HOST_SETTINGS_DIR": str(case / "settings"),
        "L3MS_AUDIO_CPP_ROOT": str(args.audio_cpp_root.resolve()),
        "L3MS_MEDIA_ROOT": str(args.media_root.resolve()),
        "L3MS_BREEZE_TTS_THREADS": "2", "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }
    arguments = ["--run-profile", "media:breeze-tts-2-cpu", "--task",
                 "clon" if args.mode == "clon" else "tts", "--input",
                 str(case / "input.json"), "--output-dir", str(output),
                 "--task-timeout-ms", "120000"]
    binary = repo / "target/debug/l3ms"
    (case / "invocation.json").write_text(json.dumps({
        "argv": [str(binary), *arguments], "environment": environment,
        "seed": 0, "cpu_threads": 2, "deadline_ms": 120000,
    }, indent=2))
    probe = runpy.run_path(str(repo / "maintenance/probe-nagi-coordinator.py"))
    started = time.monotonic()
    result = probe["run_cli"](binary, arguments, repo, environment, 140)
    elapsed = time.monotonic() - started
    assert result is not None, "Inference/cleanup unverified; inspect retained case; do not replay"
    assert result["mode"] == ("voice_clone" if args.mode == "clon" else "streaming")
    artifacts = result["artifacts"]
    assert any(item["native_kind"] == "final_wave" for item in artifacts)
    if args.mode == "streaming":
        assert any(item["native_kind"] == "streaming_chunk" for item in artifacts)
    waves = []
    for item in artifacts:
        path = Path(item["path"])
        assert path.resolve().is_relative_to(output)
        with wave.open(str(path)) as audio:
            assert audio.getnchannels() == 1 and audio.getsampwidth() == 2
            assert audio.getframerate() == 24000 and audio.getnframes() > 0
            samples = array.array("h", audio.readframes(audio.getnframes()))
            waves.append({"path": str(path), "frames": len(samples),
                          "peak_pcm16": max(map(abs, samples))})
    assert any(item["peak_pcm16"] > 0 for item in waves)
    receipt = {"ok": True, "result": result, "wall_seconds": elapsed,
               "waves": waves, "process_identities_released": True,
               "quality_evaluation": False, "live_audio_playback": False}
    (case / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
