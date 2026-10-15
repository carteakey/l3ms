#!/usr/bin/env python3
"""
l3ds Unified Decision Runner & Benchmark Engine.
Supports:
  - Nagi (Smol 480M / Big 4B / HUGE 12B) via nagi-decisions
  - Laya (ModernBERT-large) via laya
  - SemIf (OpenJev) via semif-phase1 direct logit extraction
  - Jev 1.13 via typesafe-sdk (or WAN emulation fallback)
  - llama.cpp native System One via local llama-swap model routing
"""

import argparse
import ipaddress
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Tuple
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class BaseDecisionEngine:
    def decide(self, state: Dict[str, Any], questions: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class InferenceUnavailable(RuntimeError):
    """A requested native or hosted decision backend cannot run."""


MAX_INPUT_BYTES = 1024 * 1024
MAX_DECISION_IMAGES = 8
LLAMACPP_IMAGE_MODELS = {"vllm-d3-flash", "cloudflare-clef-flash"}
MAX_HTTP_RESPONSE_BYTES = 2 * 1024 * 1024
LLAMACPP_TIMEOUT_SECONDS = 30
LLAMACPP_DEFAULT_BASE_URL = "http://localhost:8080"
_ROUTER_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_IMAGE_DATA_URL = re.compile(
    r"data:image/[A-Za-z0-9][A-Za-z0-9.+-]*;base64,"
    r"(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{4})"
)


def reject_nonfinite_json(_value):
    raise ValueError("non-finite JSON number")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _llamacpp_endpoint(base_url: str) -> str:
    """Build the llama-swap native System One URL."""
    if not isinstance(base_url, str):
        raise ValueError("invalid endpoint")

    parsed = urlsplit(base_url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("endpoint must be local HTTP")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("endpoint credentials are not allowed")
    if parsed.query or parsed.fragment:
        raise ValueError("endpoint cannot include a query or fragment")
    try:
        port = parsed.port
        host = parsed.hostname.lower()
        is_loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = False
        port = None
    if not is_loopback:
        raise ValueError("endpoint must use a localhost address")

    netloc = host if port is None else f"{host}:{port}"
    if ":" in host:
        netloc = f"[{host}]" if port is None else f"[{host}]:{port}"
    base_path = parsed.path.rstrip("/")
    path = f"{base_path}/v1/systemone"
    return urlunsplit(("http", netloc, path, "", ""))


class LlamaCppSystemOneEngine(BaseDecisionEngine):
    """One-shot HTTP bridge to llama.cpp's native System One endpoint."""

    def __init__(self, model: str, api_key: str = None, base_url: str = None):
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("LLAMA_SWAP_API_KEY")
        configured_url = base_url if base_url is not None else os.environ.get("LLAMA_SWAP_URL")
        if not configured_url:
            configured_url = LLAMACPP_DEFAULT_BASE_URL
        self.initialization_error = None
        self.endpoint = None
        self.opener = None
        try:
            if not isinstance(model, str) or not _ROUTER_MODEL_ID.fullmatch(model):
                raise ValueError("invalid router model id")
            self.endpoint = _llamacpp_endpoint(configured_url)
            if self.api_key is not None and any(char in self.api_key for char in "\r\n"):
                raise ValueError("invalid credential")
            # Do not route local credentials through environment-configured proxies, and do not
            # follow redirects that could send the bearer token or decision payload elsewhere.
            self.opener = build_opener(ProxyHandler({}), _NoRedirect())
        except (TypeError, ValueError):
            self.initialization_error = "llama.cpp endpoint configuration is invalid"

    def decide(
        self,
        state: Dict[str, Any],
        questions: Dict[str, Any],
        images: List[str] = None,
    ) -> Dict[str, Any]:
        if self.initialization_error:
            raise InferenceUnavailable(self.initialization_error) from None
        try:
            images = [] if images is None else images
            _validate_decision_images(images)
            _validate_llamacpp_questions(self.model, questions)
            if images and self.model not in LLAMACPP_IMAGE_MODELS:
                raise ValueError("selected model does not support decision images")
            request_body = {"model": self.model, "state": state, "questions": questions}
            if images:
                request_body["images"] = images
            body = json.dumps(
                request_body,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            if len(body) > MAX_INPUT_BYTES:
                raise ValueError("request too large")
            request = Request(
                self.endpoint,
                data=body,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            if self.api_key:
                request.add_header("Authorization", f"Bearer {self.api_key}")

            started = time.perf_counter()
            with self.opener.open(request, timeout=LLAMACPP_TIMEOUT_SECONDS) as response:
                if response.getcode() != 200:
                    raise ValueError("unexpected HTTP status")
                response_body = response.read(MAX_HTTP_RESPONSE_BYTES + 1)
            if len(response_body) > MAX_HTTP_RESPONSE_BYTES:
                raise ValueError("response too large")
            payload = json.loads(
                response_body.decode("utf-8"),
                parse_constant=reject_nonfinite_json,
            )
            if not isinstance(payload, dict) or "error" in payload:
                raise ValueError("invalid response")
            json.dumps(payload, allow_nan=False)

            for result_field in ("choices", "scores", "nouls"):
                if result_field in payload and not isinstance(payload[result_field], dict):
                    raise ValueError("malformed native answers")

            if "answers" in payload:
                answers = payload["answers"]
                if not isinstance(answers, dict) or not answers:
                    raise ValueError("invalid native answers")
            else:
                answers = {}
                for result_field in ("choices", "scores", "nouls"):
                    field_answers = payload.get(result_field)
                    if field_answers is not None:
                        answers.update(field_answers)
                if not answers:
                    raise ValueError("missing native answers")
                payload["answers"] = answers
            if set(answers) != set(questions) or any(
                not isinstance(answer, dict) or not answer for answer in answers.values()
            ):
                raise ValueError("malformed native answers")
            for name, answer in answers.items():
                kind = questions[name].get("type")
                if kind in ("noul", "score"):
                    value = answer.get(kind)
                    if type(value) not in (int, float) or (kind == "noul" and not 0 <= value <= 1):
                        raise ValueError("invalid typed answer")
                if kind == "choice":
                    allowed = questions[name].get("criteria", questions[name].get("options", []))
                    if answer.get("choice") not in allowed:
                        raise ValueError("undeclared choice")
                probabilities = answer.get("probabilities", {})
                if not isinstance(probabilities, dict):
                    raise ValueError("invalid probabilities")
                values = list(probabilities.values())
                if "confidence" in answer:
                    values.append(answer["confidence"])
                if any(type(value) not in (int, float) or not 0 <= value <= 1 for value in values):
                    raise ValueError("invalid probability")

            payload.setdefault("model", self.model)
            payload["client_latency_ms"] = (time.perf_counter() - started) * 1000
            payload["engine"] = "llamacpp-native"
            return payload
        except Exception:
            # Transport, HTTP, and parse errors may include headers, credentials, request data,
            # or response bodies. Keep the failure fixed and do not synthesize an answer.
            raise InferenceUnavailable("llama.cpp System One request failed") from None


def one_shot_inputs(args):
    """Read bounded decision input without exposing state/questions in argv."""
    if args.input_stdin:
        payload = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if not payload or len(payload) > MAX_INPUT_BYTES:
            raise ValueError("invalid input size")
        envelope = json.loads(payload.decode("utf-8"), parse_constant=reject_nonfinite_json)
        if (
            not isinstance(envelope, dict)
            or not {"state", "questions"}.issubset(envelope)
            or set(envelope) - {"state", "questions", "images"}
        ):
            raise ValueError("invalid input envelope")
        state, questions = envelope["state"], envelope["questions"]
        images = envelope.get("images", [])
        _validate_decision_images(images)
        if not isinstance(questions, dict) or not 1 <= len(questions) <= 128:
            raise ValueError("invalid question count")
        if not all(isinstance(key, str) and key.strip() and len(key.encode("utf-8")) <= 128
                   and isinstance(value, dict) and value for key, value in questions.items()):
            raise ValueError("invalid questions")
        return state, questions, images
    state = json.loads(args.state) if args.state else {"message": "Server 500 error encountered on checkout"}
    questions = json.loads(args.question) if args.question else {
        "routing": {
            "type": "choice",
            "instructions": "Route request to the responsible on-call squad.",
            "criteria": {
                "payments": "Billing, checkout, checkout pipelines",
                "infrastructure": "Kubernetes, hardware, network nodes",
                "support": "General customer support tickets"
            }
        }
    }
    return state, questions, []


def _validate_decision_images(images):
    if (
        not isinstance(images, list)
        or len(images) > MAX_DECISION_IMAGES
        or not all(isinstance(image, str) and _IMAGE_DATA_URL.fullmatch(image) for image in images)
    ):
        raise ValueError("invalid decision images")


def _validate_llamacpp_questions(model, questions):
    if model == "cloudflare-clef-flash" and (
        not isinstance(questions, dict)
        or any(not isinstance(question, dict) or question.get("instructions") is None
               for question in questions.values())
    ):
        raise ValueError("Clef decision questions require instructions")


class MockOrApiJevEngine(BaseDecisionEngine):
    def __init__(
        self,
        model: str = "jev-latest",
        api_key: str = None,
        base_url: str = None,
        allow_simulation: bool = False,
        single_attempt: bool = False,
    ):
        self.model = model
        self.allow_simulation = allow_simulation
        self.single_attempt = single_attempt
        self.single_attempt_consumed = False
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = base_url or os.environ.get("TYPESAFE_BASE_URL")
        self.client = None
        self.initialization_error = None
        if self.single_attempt and self.allow_simulation:
            self.initialization_error = "single-attempt hosted mode cannot use simulation"
        elif not self.allow_simulation and not self.api_key:
            self.initialization_error = "hosted decision API credentials are unavailable"
        elif self.single_attempt:
            # The SDK documents TYPESAFE_LOG_LEVEL=off; its DEBUG wire logger includes request and
            # response bodies. Keep that logger disabled for this one-shot process before import.
            os.environ["TYPESAFE_LOG_LEVEL"] = "off"
            logging.getLogger("typesafe_sdk").disabled = True

        if self.api_key and not self.allow_simulation and self.initialization_error is None:
            try:
                if self.single_attempt:
                    from typesafe_sdk import RetryPolicy, TypeSafeClient

                    self.client = TypeSafeClient(
                        api_key=self.api_key,
                        base_url=self.base_url,
                        retry=RetryPolicy(max_retries=0),
                    )
                else:
                    from typesafe_sdk import TypeSafeClient

                    self.client = TypeSafeClient(api_key=self.api_key, base_url=self.base_url)
            except Exception:
                self.initialization_error = "hosted decision client is unavailable"

    def decide(self, state: Dict[str, Any], questions: Dict[str, Any]) -> Dict[str, Any]:
        if self.single_attempt and self.single_attempt_consumed:
            raise InferenceUnavailable("hosted decision request is already complete") from None
        if self.client:
            t0 = time.perf_counter()
            try:
                resp = self.client.system_one(state=state, questions=questions, model=self.model)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                dump = resp.model_dump() if hasattr(resp, "model_dump") else {}
                answers = dump.get("answers", {})
                if not answers:
                    answers = {}
                    if "choices" in dump and dump["choices"]:
                        answers.update(dump["choices"])
                    if "scores" in dump and dump["scores"]:
                        answers.update(dump["scores"])
                    if "nouls" in dump and dump["nouls"]:
                        answers.update(dump["nouls"])
                return {
                    "model": dump.get("model", self.model),
                    "usage": dump.get("usage", {}),
                    "answers": answers,
                    "latency_ms": elapsed_ms,
                    "engine": "systemone-official-api"
                }
            except Exception:
                raise InferenceUnavailable("hosted decision request failed") from None
            finally:
                if self.single_attempt:
                    self.single_attempt_consumed = True
                    client, self.client = self.client, None
                    try:
                        client.close()
                    except Exception:
                        pass
        else:
            if self.initialization_error:
                raise InferenceUnavailable(self.initialization_error) from None
            if not self.allow_simulation:
                raise InferenceUnavailable("hosted decision client is unavailable") from None
            # Emulate WAN network latency of hosted Jev API (P50 ~250ms, jitter up to 500ms)
            state_str = json.dumps(state) if isinstance(state, dict) else str(state)
            simulated_network_delay = 0.22 + (hash(state_str) % 100) / 1000.0
            time.sleep(simulated_network_delay)
            answers = {}
            for q_id, q_data in questions.items():
                opts = list(q_data.get("criteria", {}).keys()) or list(q_data.get("options", []))
                chosen = opts[0] if opts else "default"
                answers[q_id] = {
                    "choice": chosen,
                    "confidence": 0.94,
                    "probabilities": {opt: (0.94 if opt == chosen else 0.06 / max(1, len(opts) - 1)) for opt in opts}
                }
            return {
                "answers": answers,
                "latency_ms": simulated_network_delay * 1000,
                "engine": "jev-api (simulated-wan, set TYPESAFE_API_KEY for live calls)"
            }


class NagiEngine(BaseDecisionEngine):
    def __init__(self, model_name: str, device: str = "cuda", allow_simulation: bool = False):
        self.model_name = model_name
        self.device = device
        self.allow_simulation = allow_simulation
        self.model = None
        self.load_err = "model did not load"
        if self.allow_simulation:
            return
        try:
            import nagi
            if "smol" in model_name.lower():
                try:
                    self.model = nagi.load_smol(repo=model_name, device=device)
                except Exception as cuda_err:
                    if device != "cpu":
                        print(f"[{model_name}] CUDA initialization ({cuda_err}), falling back to CPU...", file=sys.stderr)
                        self.model = nagi.load_smol(repo=model_name, device="cpu")
                    else:
                        raise
            elif "huge" in model_name.lower():
                self.model = nagi.load_huge(device=device)
            else:
                try:
                    self.model = nagi.load_big(repo=model_name, device=device)
                except Exception as cuda_err:
                    if device != "cpu":
                        print(f"[{model_name}] CUDA initialization ({cuda_err}), falling back to CPU...", file=sys.stderr)
                        self.model = nagi.load_big(repo=model_name, device="cpu")
                    else:
                        raise
        except Exception as e:
            self.model = None
            self.load_err = str(e)

    def decide(self, state: Dict[str, Any], questions: Dict[str, Any]) -> Dict[str, Any]:
        if self.model:
            t0 = time.perf_counter()
            res = self.model.system_one(state=state, questions=questions)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "answers": res.get("answers", {}),
                "latency_ms": elapsed_ms,
                "engine": "nagi-native"
            }
        else:
            if not self.allow_simulation:
                raise InferenceUnavailable(f"Nagi native model unavailable: {self.load_err}")
            tier_delay = 0.017 if "smol" in self.model_name.lower() else 0.068
            time.sleep(tier_delay)
            answers = {}
            for q_id, q_data in questions.items():
                opts = list(q_data.get("criteria", {}).keys()) or list(q_data.get("options", []))
                chosen = opts[0] if opts else "default"
                answers[q_id] = {
                    "choice": chosen,
                    "confidence": 0.88,
                    "probabilities": {opt: (0.88 if opt == chosen else 0.12 / max(1, len(opts) - 1)) for opt in opts}
                }
            return {
                "answers": answers,
                "latency_ms": tier_delay * 1000,
                "engine": f"nagi-simulated ({self.model_name})"
            }


class LayaEngine(BaseDecisionEngine):
    def __init__(self, model_name: str, device: str = "cuda", allow_simulation: bool = False):
        self.model_name = model_name
        self.device = device
        self.allow_simulation = allow_simulation
        self.agent = None
        self.load_err = "model did not load"
        if self.allow_simulation:
            return
        try:
            import laya
            try:
                self.agent = laya.load(model_name, device=device)
            except Exception as cuda_err:
                if device != "cpu":
                    print(f"[{model_name}] CUDA initialization ({cuda_err}), falling back to CPU...", file=sys.stderr)
                    self.agent = laya.load(model_name, device="cpu")
                else:
                    raise
        except Exception as e:
            self.agent = None
            self.load_err = str(e)

    def decide(self, state: Dict[str, Any], questions: Dict[str, Any]) -> Dict[str, Any]:
        if self.agent:
            st_text = state if isinstance(state, str) else json.dumps(state)
            t0 = time.perf_counter()
            res = self.agent.predict(state=st_text, questions=questions)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            raw_answers = res.get("answers", {}) if isinstance(res, dict) else (res.answers if hasattr(res, "answers") else {})
            normalized_answers = {}
            for q_id, q_val in raw_answers.items():
                if isinstance(q_val, dict):
                    normalized_answers[q_id] = {
                        "choice": q_val.get("choice"),
                        "probabilities": q_val.get("probabilities", {}),
                        "confidence": q_val.get("confidence", 0.0)
                    }
                else:
                    normalized_answers[q_id] = {
                        "choice": getattr(q_val, "choice", None),
                        "probabilities": getattr(q_val, "probabilities", {}),
                        "confidence": getattr(q_val, "confidence", 0.0)
                    }
            return {
                "answers": normalized_answers,
                "latency_ms": elapsed_ms,
                "engine": "laya-native"
            }
        else:
            if not self.allow_simulation:
                raise InferenceUnavailable(f"Laya native model unavailable: {self.load_err}")
            delay = 0.013
            time.sleep(delay)
            answers = {}
            for q_id, q_data in questions.items():
                opts = list(q_data.get("criteria", {}).keys()) or list(q_data.get("options", []))
                chosen = opts[0] if opts else "default"
                answers[q_id] = {
                    "choice": chosen,
                    "confidence": 0.82,
                    "probabilities": {opt: (0.82 if opt == chosen else 0.18 / max(1, len(opts) - 1)) for opt in opts}
                }
            return {
                "answers": answers,
                "latency_ms": delay * 1000,
                "engine": "laya-simulated"
            }


class SemIfEngine(BaseDecisionEngine):
    def __init__(self, model_name: str, device: str = "cuda", allow_simulation: bool = False):
        self.model_name = model_name
        self.device = device
        self.allow_simulation = allow_simulation
        self.scorer = None
        if self.allow_simulation:
            return
        try:
            import semif_phase1
            self.scorer = True
        except Exception:
            self.scorer = None

    def decide(self, state: Dict[str, Any], questions: Dict[str, Any]) -> Dict[str, Any]:
        # The current bridge has no native forward pass; it must not fabricate
        # production answers or benchmark measurements.
        if not self.allow_simulation:
            raise InferenceUnavailable("SemIf native inference is not implemented; --simulate is demo-only")
        delay = 0.051
        time.sleep(delay)
        answers = {}
        for q_id, q_data in questions.items():
            opts = list(q_data.get("criteria", {}).keys()) or list(q_data.get("options", []))
            chosen = opts[0] if opts else "default"
            answers[q_id] = {
                "choice": chosen,
                "confidence": 0.85,
                "probabilities": {opt: (0.85 if opt == chosen else 0.15 / max(1, len(opts) - 1)) for opt in opts}
            }
        return {
            "answers": answers,
            "latency_ms": delay * 1000,
            "engine": "semif-simulated"
        }


def get_engine(engine_name: str, model: str, device: str, api_key: str = None, base_url: str = None, allow_simulation: bool = False, single_attempt: bool = False) -> BaseDecisionEngine:
    if engine_name == "nagi":
        return NagiEngine(model, device, allow_simulation)
    elif engine_name == "laya":
        return LayaEngine(model, device, allow_simulation)
    elif engine_name == "semif":
        return SemIfEngine(model, device, allow_simulation)
    elif engine_name == "jev":
        return MockOrApiJevEngine(model, api_key=api_key, base_url=base_url, allow_simulation=allow_simulation, single_attempt=single_attempt)
    elif engine_name == "llamacpp":
        if allow_simulation or single_attempt:
            raise ValueError("llama.cpp System One does not support simulation or the Jev single-attempt flag")
        return LlamaCppSystemOneEngine(model, api_key=api_key, base_url=base_url)
    else:
        raise ValueError(f"Unknown engine: {engine_name}")


def run_benchmark(engine: BaseDecisionEngine, suite_file: str, iterations: int):
    with open(suite_file, "r") as f:
        suite = json.load(f)

    test_cases = suite.get("cases", [])
    latencies = []
    flips = 0
    total_pairs = 0

    for _ in range(iterations):
        for case in test_cases:
            state = case.get("state", {})
            questions = case.get("questions", {})

            # Standard forward pass
            t0 = time.perf_counter()
            res1 = engine.decide(state, questions)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

            # Option-order flip test: reverse criteria / options
            flipped_questions = {}
            for q_id, q_info in questions.items():
                fq = dict(q_info)
                if "criteria" in fq:
                    keys = list(fq["criteria"].keys())
                    keys.reverse()
                    fq["criteria"] = {k: fq["criteria"][k] for k in keys}
                elif "options" in fq:
                    opts = list(fq["options"])
                    opts.reverse()
                    fq["options"] = opts
                flipped_questions[q_id] = fq

            res2 = engine.decide(state, flipped_questions)

            for q_id in questions.keys():
                total_pairs += 1
                c1 = res1.get("answers", {}).get(q_id, {}).get("choice")
                c2 = res2.get("answers", {}).get(q_id, {}).get("choice")
                if c1 != c2:
                    flips += 1

    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p95_idx = int(len(latencies) * 0.95)
    p95 = latencies[min(p95_idx, len(latencies) - 1)] if latencies else 0.0
    flip_rate = (flips / total_pairs) if total_pairs > 0 else 0.0

    result = {
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "flip_rate": round(flip_rate, 4),
        "samples": len(latencies),
        "flips": flips,
        "total_pairs": total_pairs
    }
    print(f"BENCH_RESULT:{json.dumps(result)}")


def main():
    parser = argparse.ArgumentParser(description="l3ds Decision Model Runner")
    parser.add_argument("--engine", required=True, choices=["nagi", "laya", "semif", "jev", "llamacpp"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--state", type=str, default=None)
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--input-stdin", action="store_true", help="Read bounded JSON {state, questions[, images]} from stdin; keeps input out of argv")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--bench-suite", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--simulate", action="store_true", help="Explicit demo mode; results are synthetic and not benchmark evidence")
    parser.add_argument("--single-attempt", action="store_true", help="Disable hosted SDK retries for one Jev API request")

    args = parser.parse_args()
    if args.single_attempt and (args.engine != "jev" or args.simulate or args.bench_suite):
        parser.error("--single-attempt requires one live Jev invocation without --simulate or --bench-suite")
    if args.engine == "llamacpp" and args.simulate:
        parser.error("--simulate is not available for llama.cpp System One")
    if args.input_stdin and (args.state is not None or args.question is not None or args.bench_suite is not None):
        parser.error("--input-stdin cannot be combined with --state, --question, or --bench-suite")
    if not args.bench_suite:
        try:
            state, questions, images = one_shot_inputs(args)
            if args.engine == "llamacpp":
                _validate_llamacpp_questions(args.model, questions)
        except (ValueError, UnicodeError, RecursionError):
            print("Invalid decision input: expected bounded JSON state/questions", file=sys.stderr)
            return 2
        if images and (args.engine != "llamacpp" or args.model not in LLAMACPP_IMAGE_MODELS):
            print("Decision images require the llama.cpp engine with a supported d3/Clef model", file=sys.stderr)
            return 2

    try:
        engine = get_engine(args.engine, args.model, args.device, api_key=args.api_key, base_url=args.base_url, allow_simulation=args.simulate, single_attempt=args.single_attempt)
        if args.bench_suite:
            run_benchmark(engine, args.bench_suite, args.iterations)
            return 0

        # One-shot decision
        if args.engine == "llamacpp":
            res = engine.decide(state, questions, images=images)
        else:
            res = engine.decide(state, questions)
        print(json.dumps(res, indent=2))
        return 0
    except InferenceUnavailable as error:
        print(f"Inference unavailable: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
