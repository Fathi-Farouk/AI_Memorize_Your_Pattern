"""
AI Memorize Your Pattern — Trainer/Teacher pattern-learning + deployment checking, single file.

============================================================================
 ARCHITECTURE OVERVIEW
============================================================================
Two phases, three AI agents, one deterministic checking engine:

  PHASE 1 — TRAINING (admin uploads confirmed-correct samples, AI learns the format)
    Trainer        -> studies a sample pair, proposes a format spec (JSON)
    Teacher        -> tests that spec against independent extra samples, approves
                       or sends feedback back to the Trainer (loops, capped at max_rounds)
    Code generator -> once approved, "compiles" the spec into a real Python parser
                       function — this is what makes Phase 2 need ZERO further AI calls

  PHASE 2 — DEPLOYMENT (any user checks real script pairs against a saved pattern)
    Uses ONLY the generated parser from Phase 1 — no AI, no network calls, just
    plain deterministic Python: count sites/cells, diff them, flag duplicate
    values, missing semicolons, spread-site implementation, etc.

Signal contract for the training graph (see build_graph() below):
  S1  Admin -> Trainer   : {activation_text, rollback_text}
  S2  Trainer (self)     : studies the pair -> candidate_pattern v1
  S3  Trainer -> Teacher : {pattern: candidate_pattern, round: 1}
  S4  Admin -> Teacher   : {extra_samples: [...]}
  S5  Teacher (self)     : tests candidate_pattern against each extra sample
  S6  Decision           : overall_confidence >= threshold ?
  S7a Teacher -> System  : {approved: true, pattern, confidence, tested_on} -> code generator
  S7b Teacher -> Trainer : {approved: false, issues, failing_samples, round} -> loop
  S8  Trainer (self)     : refines pattern using original sample + feedback -> v(round+1)
      (loops back to S3, capped at max_rounds)

============================================================================
 PROVIDERS
============================================================================
"anthropic" or "huawei" (any OpenAI-compatible internal endpoint). Credentials — provider,
API key, base_url, model — come exclusively from the GUI credentials panel or the request body
for REST calls. .env is never consulted for any of these; there is no fallback. This is
deliberate: nothing should silently run against a shared or stale credential someone didn't
knowingly enter for that session. (.env is still used for unrelated settings like PATTERNS_FILE
and PORT — see below — just not for anything credential-related.)

============================================================================
 SURFACES
============================================================================
  REST API  : POST /train, POST /check, GET /health, GET /patterns  (for other tools/scripts)
  Gradio GUI: mounted at /gui — Training tab + Deployment tab, human-friendly

Run:
  python api.py                      (prints URLs, boots everything)
  uvicorn api:app --reload --port 8787   (dev mode with auto-reload)
Docs:
  http://localhost:8787/docs   (REST API, Swagger)
  http://localhost:8787/gui    (Gradio GUI)
"""

# ============================= Imports =============================
import json
import os
import pathlib
import re
from typing import Optional, TypedDict

import gradio as gr
import httpx
import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

load_dotenv()  # reads .env in the script's own directory if present — see PATTERNS_FILE below
              # for why we anchor paths to __file__ rather than relying on cwd

# Shared instruction block prepended to every AI call (via the 'system' role for Anthropic,
# folded into the user message for the internal/Huawei branch — see call_llm below for why).
SCOPE_GUARD = (
    "You are a specialized assistant embedded inside a script-validation tool for telecom RF "
    "activation/rollback MML scripts. You may only discuss loaded scripts, sites, cells, "
    "parameters, mismatches, and related RF script-validation topics."
)


# ============================= JSON parsing =============================
def parse_ai_json(raw: str):
    """AI responses are sometimes wrapped in code fences, trail off near a token limit, or —
    seen from some models, not Claude — prefaced with commentary/reasoning about the instructions
    ("The user wants a JSON object...") before the actual JSON, instead of just producing it.
    Tries a straight parse, then trimming back to the last closed brace/bracket, then searching
    for a JSON object/array embedded somewhere after leading preamble text."""
    cleaned = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        last = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if last > 0:
            try:
                return json.loads(cleaned[: last + 1])
            except json.JSONDecodeError:
                pass

        # recovery: skip past any leading preamble text and try from the first { or [ onward —
        # handles models that "think out loud" before the real JSON, as long as the JSON itself
        # actually made it into the response before the token budget ran out
        candidates = [i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1]
        first_brace = min(candidates) if candidates else -1
        if first_brace > 0:
            substring = cleaned[first_brace:]
            try:
                return json.loads(substring)
            except json.JSONDecodeError:
                last2 = max(substring.rfind("}"), substring.rfind("]"))
                if last2 > 0:
                    try:
                        return json.loads(substring[: last2 + 1])
                    except json.JSONDecodeError:
                        pass

        preview = raw[:150].replace("\n", "\\n") if raw else "(empty)"
        raise ValueError(f"AI response was not valid JSON (possibly cut off): {e}. Raw response started with: {preview!r}") from e


# ============================= Token usage tracking (session totals) =============================
TOKEN_USAGE = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "cost_usd": 0.0}

# Per-model pricing (USD per million tokens). Cost is accumulated incrementally at call time using
# whichever model actually served that specific request — necessary now that both Anthropic and
# Huawei-internal calls can use several different models within the same session, each billed at
# a different rate. Keys are matched case-insensitively against whatever model string was used.
MODEL_PRICING = {
    # Anthropic — keyed by the real API model ID (what's actually sent in the request), not the
    # human-readable dropdown label
    "claude-fable-5":            {"input": 10.0, "output": 50.0, "cached": 1.0},
    "claude-opus-4-8":           {"input": 5.0, "output": 0.5, "cached": 25.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 0.1, "cached": 5.0},
    "claude-opus-4-7":           {"input": 5.0, "output": 0.5, "cached": 25.0},
    "claude-opus-4-6":           {"input": 5.0, "output": 0.5, "cached": 25.0},
    "claude-sonnet-4-6":         {"input": 3.0, "output": 0.3, "cached": 15.0},
    # Huawei-internal
    "qwen-aw-35b":        {"input": 0.01, "output": 0.01, "cached": 0.001},
    "qwen3.5-122b":       {"input": 0.01, "output": 0.01, "cached": 0.001},
    "qwen3.6-35b":        {"input": 0.01, "output": 0.01, "cached": 0.001},
    "minimax2.7":         {"input": 0.01, "output": 0.01, "cached": 0.001},
    "deepseek-v4-flash":  {"input": 0.01, "output": 0.01, "cached": 0.001},
    "deepseek-v4-pro":    {"input": 0.01, "output": 0.01, "cached": 0.001},
    "kimi-k2.7":          {"input": 0.01, "output": 0.01, "cached": 0.001},
}

ANTHROPIC_MODEL_CHOICES = [
    ("Claude Sonnet 4.6", "claude-sonnet-4-6"),
    ("Claude Opus 4.8", "claude-opus-4-8"),
    ("Claude Opus 4.7", "claude-opus-4-7"),
    ("Claude Opus 4.6", "claude-opus-4-6"),
    ("Claude Haiku 4.5", "claude-haiku-4-5-20251001"),
    ("Claude Fable 5", "claude-fable-5"),
]
HUAWEI_MODEL_CHOICES = ["minimax2.7", "qwen-aw-35b", "qwen3.5-122b", "qwen3.6-35b", "deepseek-v4-flash", "deepseek-v4-pro", "kimi-k2.7"]


def _record_usage(input_tokens: int = 0, output_tokens: int = 0, cached_input_tokens: int = 0, model: Optional[str] = None):
    TOKEN_USAGE["input_tokens"] += input_tokens
    TOKEN_USAGE["output_tokens"] += output_tokens
    TOKEN_USAGE["cached_input_tokens"] += cached_input_tokens
    rates = MODEL_PRICING.get((model or "").strip().lower())
    if rates:
        TOKEN_USAGE["cost_usd"] += (
            (input_tokens / 1_000_000) * rates["input"]
            + (output_tokens / 1_000_000) * rates["output"]
            + (cached_input_tokens / 1_000_000) * rates["cached"]
        )
    # unrecognized model string -> tokens still counted, cost silently not added rather than
    # guessing at a rate; the model name itself is visible in the training/deployment logs if
    # someone needs to reconcile a mismatch


def estimate_cost_usd() -> float:
    return TOKEN_USAGE["cost_usd"]


# ============================= Multi-provider LLM call =============================
class TruncatedResponseError(ValueError):
    """Raised when the model's response hit max_tokens before finishing. This means the API call
    itself succeeded — auth worked, the endpoint responded — it's a content-completeness problem,
    not a reachability problem. Callers that only care about reachability (like the connectivity
    test) should treat this as success."""
    def __init__(self, message, partial_text=""):
        super().__init__(message)
        self.partial_text = partial_text


def call_llm(prompt: str, max_tokens: int = 3072, provider_override: Optional[str] = None,
             api_key_override: Optional[str] = None, base_url_override: Optional[str] = None,
             model_override: Optional[str] = None) -> str:
    """Routes to Anthropic or an internal OpenAI-compatible endpoint (e.g. Huawei). Provider, key,
    base_url and model all come exclusively from the caller (the GUI credentials panel or a REST
    request) — .env is never consulted for any of these, by design, so nothing is silently using
    a stale or shared credential the person didn't knowingly enter."""
    provider = (provider_override or "anthropic").lower()

    if provider == "anthropic":
        api_key = api_key_override
        model = model_override
        base_url = base_url_override
        if not api_key:
            raise ValueError("No Anthropic API key provided — enter one in the GUI credentials panel.")
        if not model:
            raise ValueError("No Anthropic model selected — choose one in the GUI credentials panel.")
        if not base_url:
            raise ValueError("No Anthropic base URL provided — enter one in the GUI credentials panel (the standard value is https://api.anthropic.com).")
        client = Anthropic(api_key=api_key, base_url=base_url)
        try:
            resp = client.messages.create(
                model=model, max_tokens=max_tokens, system=SCOPE_GUARD,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIConnectionError as e:
            raise ValueError(
                f"🔌 Connection problem — could not reach {base_url}. Check the base URL is correct "
                f"and that you have network/VPN access to it. ({e})"
            ) from e
        except anthropic.APITimeoutError as e:
            raise ValueError(f"⏳ Timed out — {base_url} didn't respond in time. It may be overloaded or unreachable. ({e})") from e
        except anthropic.AuthenticationError as e:
            raise ValueError(f"🔑 Authentication failed (HTTP 401) — the API key was rejected. Check it's correct. ({e})") from e
        except anthropic.PermissionDeniedError as e:
            raise ValueError(
                f"🚧 Permission denied (HTTP 403) — either the API key doesn't have access to this model, "
                f"or a network proxy/VPN gate blocked the request before it reached Anthropic. ({e})"
            ) from e
        except anthropic.NotFoundError as e:
            raise ValueError(f"❓ Not found (HTTP 404) — check the model ID is correct and available on this account. ({e})") from e
        except anthropic.RateLimitError as e:
            raise ValueError(f"⏱️ Rate limited (HTTP 429) — too many requests too fast; wait and retry. ({e})") from e
        except (anthropic.InternalServerError, anthropic.OverloadedError) as e:
            raise ValueError(f"☁️ Provider server error — this is on Anthropic's end, not a problem with your request. ({e})") from e
        except anthropic.APIStatusError as e:
            raise ValueError(f"⚠️ Request rejected (HTTP {e.status_code}) — {e}") from e
        partial_text = "".join(b.text for b in resp.content if b.type == "text")
        _record_usage(
            input_tokens=getattr(resp.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "output_tokens", 0) or 0,
            cached_input_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            model=model,
        )
        if resp.stop_reason == "max_tokens":
            raise TruncatedResponseError("Response was cut off at the token limit before finishing.", partial_text)
        return partial_text

    else:
        # generic OpenAI-compatible internal endpoint (Huawei or any similar internal LLM gateway).
        # Deliberately matches the reference client structure exactly: only a "user" message, no
        # separate "system" role — many internal gateways proxying third-party models (minimax,
        # qwen, deepseek, etc.) reject or mishandle the system role, which was the actual cause of
        # requests failing here. Scope instructions are folded into the user turn instead.
        base_url = base_url_override
        api_key = api_key_override
        model = model_override
        if not (base_url and api_key and model):
            missing = [n for n, v in (("base_url", base_url), ("api_key", api_key), ("model", model)) if not v]
            raise ValueError(f"Missing Huawei credentials: {', '.join(missing)} — enter them in the GUI credentials panel.")
        combined_prompt = SCOPE_GUARD + "\n\n" + prompt
        try:
            resp = httpx.post(
                base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": combined_prompt}],
                },
                timeout=120,
            )
        except httpx.ConnectTimeout as e:
            raise ValueError(
                f"⏳ Connection timed out reaching {base_url} — check the base URL is correct and that "
                f"you have network/VPN access to it. ({e})"
            ) from e
        except httpx.ProxyError as e:
            raise ValueError(f"🚧 Proxy error reaching {base_url} — a network proxy blocked or rejected the connection. ({e})") from e
        except httpx.ConnectError as e:
            detail = str(e)
            if any(s in detail for s in ("Name or service not known", "nodename nor servname", "getaddrinfo failed")):
                raise ValueError(f"🔌 Could not resolve the hostname in base URL '{base_url}' — check it's typed correctly. ({e})") from e
            raise ValueError(
                f"🔌 Connection failed reaching {base_url} — the server refused the connection or is "
                f"unreachable from here. Check the base URL and that you have network/VPN access to it. ({e})"
            ) from e
        except httpx.ReadTimeout as e:
            raise ValueError(f"⏳ Connected, but {base_url} took too long to respond (read timeout) — it may be overloaded. ({e})") from e
        except httpx.TimeoutException as e:
            raise ValueError(f"⏳ Request to {base_url} timed out. ({e})") from e
        except httpx.HTTPError as e:
            raise ValueError(f"🔌 Network error reaching {base_url}: {type(e).__name__}: {e}") from e

        if not resp.is_success:
            # surface the ACTUAL error body from the gateway, not just the status code — this is
            # what makes a future failure diagnosable instead of a guessing game. An HTML body
            # (rather than JSON) usually means a corporate proxy, firewall, or VPN login gate
            # blocked the request before it ever reached the model API — a different, earlier
            # failure than anything the model backend itself could report.
            body = resp.text.strip()
            content_type = resp.headers.get("content-type", "")
            is_html = "html" in content_type.lower() or body.startswith("<")
            readable_html = None
            if is_html:
                readable_html = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body)).strip()[:300]

            if resp.status_code in (401, 403):
                if is_html:
                    raise ValueError(
                        f"🚧 Blocked before reaching the model (HTTP {resp.status_code}) — likely a "
                        f"proxy/firewall/VPN login gate, not an API key problem. Page said: {readable_html!r}"
                    )
                raise ValueError(f"🔑 Authentication rejected (HTTP {resp.status_code}) — check the API key is correct and has access. Server said: {body[:300]}")
            if resp.status_code == 404:
                raise ValueError(f"❓ Not found (HTTP 404) — check the model name and base URL path are correct. Server said: {body[:300]}")
            if resp.status_code == 429:
                raise ValueError(f"⏱️ Rate limited (HTTP 429) — too many requests too fast; wait and retry. Server said: {body[:300]}")
            if resp.status_code >= 500:
                raise ValueError(f"☁️ Provider server error (HTTP {resp.status_code}) — this is on their end, not your request. Server said: {body[:300]}")
            if is_html:
                raise ValueError(
                    f"🚧 Blocked or misrouted (HTTP {resp.status_code}) — gateway returned an HTML page "
                    f"instead of JSON, likely a proxy/firewall block. Page said: {readable_html!r}"
                )
            raise ValueError(f"⚠️ Request rejected (HTTP {resp.status_code}): {body[:500]}")
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content")
        # some reasoning-capable models/gateways return chain-of-thought separately from the
        # final answer (e.g. DeepSeek-R1-style 'reasoning_content') — if that field exists here,
        # it's strong evidence the model IS reasoning as expected, just not through the field we're
        # reading, which would point straight at the real fix (read reasoning_content separately,
        # or find a request parameter that disables the reasoning step for this model)
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        reasoning_note = f" [NOTE: a separate reasoning field was also present, {len(reasoning)} chars — see /debug/raw-call to inspect it]" if reasoning else ""
        usage = data.get("usage") or {}
        _record_usage(
            input_tokens=usage.get("prompt_tokens", 0) or 0,
            output_tokens=usage.get("completion_tokens", 0) or 0,
            cached_input_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
        )
        if choice.get("finish_reason") == "length":
            raise TruncatedResponseError(f"Response was cut off at the token limit before finishing.{reasoning_note}", content or "")
        if not content or not content.strip():
            raise ValueError(
                f"Internal endpoint returned empty content (finish_reason={choice.get('finish_reason')!r}, "
                f"completion_tokens={usage.get('completion_tokens')}).{reasoning_note} This usually means the "
                f"model's backend refused, filtered, or failed silently on this prompt rather than a JSON formatting issue."
            )
        return content


def provider_config_status(provider: str, api_key: Optional[str] = None,
                            base_url: Optional[str] = None, model: Optional[str] = None) -> Optional[str]:
    """Returns None if the given provider's config looks complete, or a specific 'what's missing'
    message otherwise — checked before attempting a network call so the error is about
    configuration, not a confusing request failure. GUI input only, .env is never consulted.
    Same three required fields for both providers now — unified shape."""
    missing = []
    if not base_url:
        missing.append("base_url")
    if not api_key:
        missing.append("api_key")
    if not model:
        missing.append("model")
    if missing:
        return f"missing {', '.join(missing)} — enter in the GUI credentials panel"
    return None


def test_api_connectivity(provider: str, api_key: Optional[str] = None,
                           base_url: Optional[str] = None, model: Optional[str] = None) -> tuple:
    """Minimal-cost ping against the chosen provider. Returns (ok, message).
    A TruncatedResponseError still counts as reachable — the model responded and got cut off by
    our own small token budget, which proves auth + connectivity worked; it's not a failure."""
    provider = (provider or "anthropic").lower()
    config_issue = provider_config_status(provider, api_key, base_url, model)
    if config_issue:
        return False, f"{provider}: {config_issue}"
    try:
        call_llm("Reply with just the word: pong", max_tokens=20, provider_override=provider,
                  api_key_override=api_key, base_url_override=base_url, model_override=model)
        return True, f"{provider}: reachable"
    except TruncatedResponseError:
        return True, f"{provider}: reachable"
    except Exception as e:
        return False, f"{provider}: {type(e).__name__}: {e}"


# ============================= Graph state =============================
class TrainState(TypedDict):
    activation_text: str
    rollback_text: str
    extra_samples: list
    candidate_pattern: Optional[dict]
    round: int
    max_rounds: int
    confidence_threshold: float
    approved: bool
    confidence: float
    per_sample_results: list
    teacher_feedback: Optional[str]
    history: list
    provider: Optional[str]
    api_key: Optional[str]
    base_url: Optional[str]
    model: Optional[str]
    generated_code: Optional[str]
    codegen_validation: Optional[dict]


# ============================= Trainer node (S2 / S8) =============================
def trainer_node(state: TrainState) -> dict:
    if state["round"] == 1:
        prompt = (
            "You are studying a CONFIRMED-CORRECT pair of telecom RF network configuration scripts: "
            "one activation script and its matching rollback script. Learn the format conventions from "
            "these two real examples so the same rules can be applied to other files in this exact format "
            "later. Do not assume any particular vendor syntax in advance — infer everything purely from "
            "what you observe.\n\n"
            f"ACTIVATION SAMPLE:\n{state['activation_text'][:7000]}\n\n"
            f"ROLLBACK SAMPLE:\n{state['rollback_text'][:7000]}\n\n"
            "Your ENTIRE response must be a single JSON object and nothing else. Do not restate these "
            "instructions, do not explain your reasoning, do not add any words before or after the JSON — "
            "your first character must be { and your last character must be }. Keep each value to ONE "
            "concise sentence: "
            '{"vendorGuess":"...","siteIdRule":"...","cellIdRule":"...","commandRule":"...","paramRule":"...",'
            '"reversalRule":"...","matchingCommandsRule":"...","commentRule":"...","notes":"..."}'
        )
    else:
        issues_text = state.get("teacher_feedback") or "no specific feedback provided"
        prompt = (
            "You previously proposed a format spec for telecom RF activation/rollback scripts, but the "
            "Teacher agent tested it against independent samples and found problems. Refine the spec.\n\n"
            f"ORIGINAL ACTIVATION SAMPLE:\n{state['activation_text'][:5000]}\n\n"
            f"ORIGINAL ROLLBACK SAMPLE:\n{state['rollback_text'][:5000]}\n\n"
            f"PREVIOUS CANDIDATE SPEC:\n{json.dumps(state['candidate_pattern'])}\n\n"
            f"TEACHER'S FEEDBACK (round {state['round']-1}):\n{issues_text}\n\n"
            "Produce a REVISED JSON object fixing these issues. Your ENTIRE response must be a single JSON "
            "object and nothing else — no restating instructions, no explanation, first character { and "
            "last character }, same keys as before: "
            '{"vendorGuess":"...","siteIdRule":"...","cellIdRule":"...","commandRule":"...","paramRule":"...",'
            '"reversalRule":"...","matchingCommandsRule":"...","commentRule":"...","notes":"..."}'
        )

    raw = call_llm(prompt, max_tokens=6000, provider_override=state.get("provider"), api_key_override=state.get("api_key"), base_url_override=state.get("base_url"), model_override=state.get("model"))
    pattern = parse_ai_json(raw)
    return {
        "candidate_pattern": pattern,
        "history": state["history"] + [{"round": state["round"], "stage": "trainer", "pattern": pattern}],
    }


# ============================= Teacher node (S5 / S6 / S7) =============================
def teacher_node(state: TrainState) -> dict:
    if not state["extra_samples"]:
        raise ValueError(
            "No extra sample pairs provided — the Teacher has nothing independent to test the "
            "candidate pattern against. Upload at least one extra activation+rollback pair."
        )
    pattern = state["candidate_pattern"]
    results = []
    for i, sample in enumerate(state["extra_samples"]):
        prompt = (
            "You are validating a candidate FORMAT SPEC against an independent confirmed-correct sample "
            "pair it has NOT seen before. Assess whether the spec correctly describes how to find site IDs, "
            "cell IDs, commands, and the activation/rollback reversal rule in THIS sample.\n\n"
            f"CANDIDATE SPEC:\n{json.dumps(pattern)}\n\n"
            f"SAMPLE ACTIVATION (excerpt):\n{sample['activation_text'][:3000]}\n\n"
            f"SAMPLE ROLLBACK (excerpt):\n{sample['rollback_text'][:3000]}\n\n"
            "Your ENTIRE response must be a single JSON object and nothing else. Do not restate these "
            "instructions or explain your reasoning — first character {, last character }: "
            '{"extractionQuality": <0-100 integer>, "issues": ["short issue", ...], "notes": "one sentence"}'
        )
        raw = call_llm(prompt, max_tokens=2048, provider_override=state.get("provider"), api_key_override=state.get("api_key"), base_url_override=state.get("base_url"), model_override=state.get("model"))
        result = parse_ai_json(raw)
        result["sampleIndex"] = i
        results.append(result)

    scores = [r.get("extractionQuality", 0) for r in results]
    overall_confidence = sum(scores) / len(scores)
    approved = overall_confidence >= state["confidence_threshold"]

    feedback = None
    if not approved:
        issues = [f"(sample {r['sampleIndex']}) {issue}" for r in results for issue in r.get("issues", [])]
        feedback = "; ".join(issues) if issues else "Extraction quality below threshold on one or more samples."

    return {
        "per_sample_results": results,
        "confidence": overall_confidence,
        "approved": approved,
        "teacher_feedback": feedback,
        "history": state["history"] + [{
            "round": state["round"], "stage": "teacher", "confidence": overall_confidence,
            "approved": approved, "per_sample_results": results,
        }],
    }


# ============================= Graph routing (S6 decision + loop bookkeeping) =============================
# This is plain Python, not an AI decision — the model never decides whether to retry, a fixed
# threshold comparison does. Three outcomes: approved -> move on to code generation; not approved
# but rounds remain -> loop back to the Trainer with feedback; not approved and out of rounds ->
# stop and report failure honestly (no silent "good enough" fallback).
def route_after_teacher(state: TrainState) -> str:
    if state["approved"]:
        return "codegen"
    if state["round"] >= state["max_rounds"]:
        return "done"
    return "retry"


def increment_round(state: TrainState) -> dict:
    return {"round": state["round"] + 1}


# ============================= Code generation: "compile" the pattern into a real parser =============================
# Instead of paying for a chunked AI call on every single line of every future file, ask the AI to
# write the parsing logic ONCE as actual Python code, then run that deterministically forever after
# — same speed/cost/reliability tradeoff reasoning as everywhere else in this app: AI for judgment
# that happens once, plain code for the part that has to run fast and often.
# Two groups, not one flat list: some tokens are dangerous no matter how they appear (module
# names, the import keyword, dunder attributes) — those get a bare word-boundary match. Others
# are common enough as ordinary English words in comments/docstrings (Core-network domain text
# plausibly includes "check input format", "open circuit", etc.) that a bare match on "open" or
# "input" would false-positive on harmless prose — those only count as unsafe when they look like
# an actual function CALL (immediately followed by a parenthesis). "compile" is deliberately
# excluded from the call-check even though compile() is a real risk, because re.compile(...) is
# essential for any regex-based parser and re.compile( would otherwise match too; the bare
# compile() builtin is already blocked structurally by the restricted __builtins__ below.
_UNSAFE_KEYWORD_RE = re.compile(
    r"\b(import|__import__|subprocess|socket|globals|locals|getattr|setattr|delattr|__builtins__)\b"
    r"|os\.|sys\.|shutil\.|pathlib\."
)
_UNSAFE_CALL_RE = re.compile(r"\b(open|input|exec|eval)\s*\(")
_SAFE_BUILTINS = {
    "len": len, "range": range, "enumerate": enumerate, "str": str, "int": int, "float": float,
    "bool": bool, "list": list, "dict": dict, "set": set, "tuple": tuple,
    "None": None, "True": True, "False": False, "isinstance": isinstance,
    "min": min, "max": max, "sum": sum, "sorted": sorted, "reversed": reversed,
    "zip": zip, "map": map, "filter": filter, "any": any, "all": all, "abs": abs,
}


def run_generated_parser(code: str, text: str) -> list:
    """Executes the generated parse_line() over every line of text, then forward-fills siteId and
    cellId across lines that don't carry their own explicit tag. Many real MML formats use a
    section-header convention — one line establishes the site (e.g. '///TMSC TSJ1MSC2'), and every
    command line beneath it implicitly belongs to that site without repeating the tag — rather than
    the inline-per-line convention ('MOD X: Param=1;{(SITEID)}') this tool originally assumed. The
    generated parser only needs to correctly recognize the header line; this deterministic pass (not
    AI, so it can't be inconsistent between training and deployment) carries that context forward,
    the same way a human reading the file would. Shared by validate_generated_code (training-time)
    and deployment checks so they can never disagree about what counts as 'recognized'."""
    namespace = {"re": re, "__builtins__": _SAFE_BUILTINS}
    exec(code, namespace)
    parse_line_fn = namespace.get("parse_line")
    if not callable(parse_line_fn):
        raise ValueError("Generated code did not define a callable parse_line(raw, line_num) function.")

    lines = []
    current_site, current_cell = None, None
    for i, raw in enumerate(text.split("\n")):
        result = parse_line_fn(raw, i + 1)
        if not isinstance(result, dict):
            raise ValueError(f"parse_line returned {type(result).__name__}, expected dict, at line {i+1}")
        if result.get("siteId"):
            current_site = result["siteId"]
        elif not result.get("isBlank") and not result.get("isComment"):
            result["siteId"] = current_site
        if result.get("cellId"):
            current_cell = result["cellId"]
        elif not result.get("isBlank") and not result.get("isComment"):
            result["cellId"] = current_cell
        lines.append(result)
    return lines


def validate_generated_code(code: str, extra_samples: list) -> dict:
    """Runs the generated parser (with the same site/cell forward-fill deployment will use) against
    the extra sample pairs used during Teacher validation, and reports whether it's actually usable
    — never trusts the code just because it compiled."""
    unsafe_match = _UNSAFE_KEYWORD_RE.search(code) or _UNSAFE_CALL_RE.search(code)
    if unsafe_match:
        matched_text = unsafe_match.group(0)
        line_num = code[:unsafe_match.start()].count("\n") + 1
        return {"ok": False, "error": f"Generated code contains a disallowed construct ({matched_text!r} on line {line_num}) and was rejected before execution.",
                "lines_tested": 0, "extraction_rate": 0.0}

    try:
        exec(code, {"re": re, "__builtins__": _SAFE_BUILTINS})
    except Exception as e:
        return {"ok": False, "error": f"Generated code failed to compile: {type(e).__name__}: {e}",
                "lines_tested": 0, "extraction_rate": 0.0}

    tested, recognized = 0, 0
    per_sample = []
    try:
        for idx, sample in enumerate(extra_samples):
            s_tested, s_recognized = 0, 0
            for text in (sample["activation_text"], sample["rollback_text"]):
                for result in run_generated_parser(code, text):
                    if result.get("isBlank"):
                        continue
                    s_tested += 1
                    if result.get("command") and result.get("siteId"):
                        s_recognized += 1
            per_sample.append({
                "sample": idx, "tested": s_tested, "recognized": s_recognized,
                "rate": (s_recognized / s_tested) if s_tested else 0.0,
            })
            tested += s_tested
            recognized += s_recognized
    except Exception as e:
        rate = recognized / tested if tested else 0.0
        return {"ok": False, "error": f"Generated code raised while parsing a real sample: {type(e).__name__}: {e}",
                "lines_tested": tested, "extraction_rate": rate, "per_sample": per_sample}

    rate = recognized / tested if tested else 0.0
    ok = rate >= 0.5
    breakdown = "; ".join(f"sample {p['sample']}: {p['rate']:.0%} ({p['recognized']}/{p['tested']})" for p in per_sample)
    error = None
    if not ok:
        error = (
            f"Only recognized {rate:.0%} of non-blank lines overall — below the 50% usability bar. "
            f"Per-sample breakdown: {breakdown}. If some samples score near 100% and others near 0%, "
            f"the parser handles one format variant but not another — the samples likely mix multiple "
            f"vendor/site-header conventions that need to be trained separately or handled with an "
            f"either/or pattern. If every sample scores near 0%, the issue is more fundamental."
        )
    return {"ok": ok, "error": error, "lines_tested": tested, "extraction_rate": rate, "per_sample": per_sample}


def parse_with_generated_code(text: str, code: str) -> list:
    """Deployment-time fast path: run the stored generated parser, no AI calls at all."""
    return run_generated_parser(code, text)


def convert_pattern_into_code_node(state: TrainState) -> dict:
    # Real sample text, not just the abstract spec — this is the actual fix for a real bug: the
    # model previously only saw the JSON pattern description and had to GUESS plausible-sounding
    # command names/syntax for the domain, which could sound reasonable in English while matching
    # zero real lines. Giving it real text to derive regex from directly closes that gap.
    extra_sample_block = ""
    if state.get("extra_samples"):
        first_extra = state["extra_samples"][0]
        extra_sample_block = (
            "\n\nADDITIONAL REAL SAMPLE (the format must work on this too — use it to cross-check, "
            "don't just pattern-match the first sample):\n"
            f"ACTIVATION:\n{first_extra['activation_text'][:3000]}\n\n"
            f"ROLLBACK:\n{first_extra['rollback_text'][:3000]}\n"
        )

    prompt = (
        "You have learned this format spec for telecom RF activation/rollback scripts, confirmed "
        f"correct against independent samples:\n{json.dumps(state['candidate_pattern'])}\n\n"
        "Here is the REAL sample text this spec was learned from — base every regex on what's "
        "ACTUALLY written here, not on generic assumptions about what a telecom command 'usually' "
        "looks like. Do not invent or guess command names, parameter names, or syntax that doesn't "
        "literally appear in this text.\n\n"
        f"ACTIVATION SAMPLE:\n{state['activation_text'][:5000]}\n\n"
        f"ROLLBACK SAMPLE:\n{state['rollback_text'][:5000]}"
        f"{extra_sample_block}\n\n"
        "Write a single Python function implementing this parsing logic directly with regex/string "
        "operations (not general reasoning), so it can run deterministically without any AI calls "
        "on every future file. EXACT required signature and return shape:\n\n"
        "def parse_line(raw: str, line_num: int) -> dict:\n"
        "    # must return exactly:\n"
        "    # {'lineNum': line_num, 'isBlank': bool, 'isComment': bool, 'command': str or None,\n"
        "    #  'siteId': str or None, 'cellId': str or None,\n"
        "    #  'params': [{'name': str, 'value': str}, ...], 'hasSemicolon': bool}\n\n"
        "Rules:\n"
        "- Derive every regex from the REAL sample text above. Do NOT hardcode a fixed list of exact "
        "command-name strings guessed from the spec's English description — if you're tempted to write "
        "something like `r'(MOD SFP|DEA BRD|ACT BRD)'`, stop and instead write a general pattern that "
        "captures whatever verb+object structure the command lines actually follow (e.g. two or three "
        "uppercase words at the start of the line before a colon), so it still works on command names "
        "that don't happen to appear in these particular samples.\n"
        "- Only the 're' module is available (already imported) — no other imports of any kind.\n"
        "- No file I/O, no network, no subprocess, no exec/eval, no __import__.\n"
        "- Define any regex patterns as module-level constants above the function if that helps.\n\n"
        "Your ENTIRE response must be nothing but the Python code itself (constants + function). Do not "
        "restate these instructions, do not explain your approach, do not add any words before or after "
        "the code — your response must start directly with either a comment, a constant assignment, or "
        "the 'def' keyword."
    )
    raw_code = call_llm(prompt, max_tokens=4096, provider_override=state.get("provider"), api_key_override=state.get("api_key"), base_url_override=state.get("base_url"), model_override=state.get("model"))
    code = re.sub(r"```python|```", "", raw_code).strip()
    # models habitually write "import re" out of reflex even when told not to, even though 're' is
    # already provided in the execution namespace — strip that specific harmless line so it doesn't
    # trip the unsafe-code check, which still blocks every OTHER import (os, sys, subprocess, etc.)
    code = re.sub(r"(?m)^\s*import\s+re\s*$\n?", "", code).strip()
    validation = validate_generated_code(code, state["extra_samples"])
    return {
        "generated_code": code,
        "codegen_validation": validation,
        "history": state["history"] + [{"round": state["round"], "stage": "codegen", "validation": validation}],
    }


# ============================= Graph assembly + training entry point =============================
# Wires the four nodes above into the actual LangGraph state machine:
#   trainer -> teacher -> (approved: convert_pattern_into_code -> END)
#                       -> (not approved, rounds left: increment_round -> trainer, loops)
#                       -> (not approved, out of rounds: END, reported as approved=False)
def build_graph():
    graph = StateGraph(TrainState)
    graph.add_node("trainer", trainer_node)
    graph.add_node("teacher", teacher_node)
    graph.add_node("increment_round", increment_round)
    graph.add_node("convert_pattern_into_code", convert_pattern_into_code_node)
    graph.set_entry_point("trainer")
    graph.add_edge("trainer", "teacher")
    graph.add_conditional_edges("teacher", route_after_teacher, {
        "codegen": "convert_pattern_into_code", "done": END, "retry": "increment_round",
    })
    graph.add_edge("convert_pattern_into_code", END)
    graph.add_edge("increment_round", "trainer")
    return graph.compile()


TRAINER_TEACHER_GRAPH = build_graph()  # compiled once at import time, reused for every training run


def run_training(activation_text: str, rollback_text: str, extra_samples: list,
                  provider: Optional[str] = None, api_key: Optional[str] = None,
                  base_url: Optional[str] = None, model: Optional[str] = None,
                  max_rounds: int = 3, confidence_threshold: float = 90.0) -> dict:
    if not extra_samples:
        raise ValueError(
            "At least one extra sample pair is required — the Teacher validates the Trainer's "
            "candidate pattern against independent samples, so training can't proceed with zero."
        )
    initial_state: TrainState = {
        "activation_text": activation_text, "rollback_text": rollback_text,
        "extra_samples": extra_samples, "candidate_pattern": None, "round": 1,
        "max_rounds": max_rounds, "confidence_threshold": confidence_threshold,
        "approved": False, "confidence": 0.0, "per_sample_results": [],
        "teacher_feedback": None, "history": [], "provider": provider, "api_key": api_key,
        "base_url": base_url, "model": model,
        "generated_code": None, "codegen_validation": None,
    }
    final_state = TRAINER_TEACHER_GRAPH.invoke(initial_state)
    return {
        "approved": final_state["approved"],
        "pattern": final_state["candidate_pattern"],
        "confidence": final_state["confidence"],
        "rounds_used": final_state["round"],
        "history": final_state["history"],
        "generated_code": final_state.get("generated_code"),
        "codegen_validation": final_state.get("codegen_validation"),
    }


# ============================= FastAPI =============================
app = FastAPI(title="Trainer/Teacher Pattern Learning API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST"], allow_headers=["*"])


class SamplePair(BaseModel):
    activation_text: str
    rollback_text: str


class TrainRequest(BaseModel):
    activation_text: str
    rollback_text: str
    extra_samples: list[SamplePair] = Field(default_factory=list)
    provider: str = "anthropic"     # "anthropic" | "huawei" — credentials below must match
    api_key: Optional[str] = None   # required — .env is never consulted
    base_url: Optional[str] = None  # required for provider="huawei"
    model: Optional[str] = None     # required for both providers
    max_rounds: int = 3
    confidence_threshold: float = 90.0


class TrainResponse(BaseModel):
    approved: bool
    pattern: Optional[dict]
    confidence: float
    rounds_used: int
    history: list


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    if not req.activation_text or not req.rollback_text:
        raise HTTPException(status_code=400, detail="activation_text and rollback_text are required")
    try:
        return run_training(
            activation_text=req.activation_text,
            rollback_text=req.rollback_text,
            extra_samples=[s.model_dump() for s in req.extra_samples],
            provider=req.provider,
            api_key=req.api_key,
            base_url=req.base_url,
            model=req.model,
            max_rounds=req.max_rounds,
            confidence_threshold=req.confidence_threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Training failed: {type(e).__name__}: {e}")


class CheckRequest(BaseModel):
    activation_text: str
    rollback_text: str
    pattern_name: Optional[str] = None    # look up a saved pattern's generated parser by name...
    generated_code: Optional[str] = None  # ...or pass a generated parser directly (at least one required)


class CheckResponse(BaseModel):
    result: dict
    markdown: str


@app.post("/check", response_model=CheckResponse)
def check(req: CheckRequest):
    if not req.activation_text or not req.rollback_text:
        raise HTTPException(status_code=400, detail="activation_text and rollback_text are required")
    generated_code = req.generated_code
    if not generated_code:
        if not req.pattern_name:
            raise HTTPException(status_code=400, detail="Provide either 'generated_code' or 'pattern_name'")
        match = next((p for p in load_saved_patterns() if p["name"] == req.pattern_name), None)
        if not match:
            raise HTTPException(status_code=404, detail=f"No saved pattern named '{req.pattern_name}'")
        generated_code, diagnostic = usable_generated_code(match)
        if not generated_code:
            raise HTTPException(status_code=422, detail=f"Pattern '{req.pattern_name}' can't be used yet — {diagnostic}")

    try:
        result = run_deployment_check(req.activation_text, req.rollback_text, generated_code)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Check failed: {type(e).__name__}: {e}")

    return {"result": result, "markdown": format_check_markdown(result)}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/patterns")
def list_patterns():
    return {"patterns": load_saved_patterns()}


class DebugRawRequest(BaseModel):
    prompt: str = "Reply with just the word: pong"
    provider: str = "anthropic"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    max_tokens: int = 500


@app.post("/debug/raw-call")
def debug_raw_call(req: DebugRawRequest):
    """Diagnostic only — bypasses call_llm's content extraction entirely and returns the COMPLETE
    raw API response body, so you can see every field the gateway actually sends back (including
    a separate reasoning/reasoning_content field if one exists, which call_llm doesn't currently
    read). Only meant for troubleshooting a specific provider/model from the browser or curl.
    Credentials must be supplied in the request — .env is never consulted."""
    provider = req.provider.lower()
    if provider == "anthropic":
        if not req.api_key:
            raise HTTPException(status_code=400, detail="api_key is required")
        if not req.model:
            raise HTTPException(status_code=400, detail="model is required")
        client = Anthropic(api_key=req.api_key)
        resp = client.messages.create(model=req.model, max_tokens=req.max_tokens, messages=[{"role": "user", "content": req.prompt}])
        return {"provider": "anthropic", "raw_response": resp.model_dump()}
    else:
        if not (req.base_url and req.api_key and req.model):
            raise HTTPException(status_code=400, detail="base_url, api_key, and model are all required for the internal provider")
        resp = httpx.post(
            req.base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {req.api_key}", "Content-Type": "application/json"},
            json={"model": req.model, "max_tokens": req.max_tokens, "messages": [{"role": "user", "content": req.prompt}]},
            timeout=120,
        )
        return {"provider": "huawei", "status_code": resp.status_code, "raw_response": resp.text}


# ============================= Robust file reading (any vendor/department, any encoding) =============================
# This tool is domain-agnostic by design — RF, Core, IMS, anything — the AI-learned pattern
# doesn't assume any specific vendor or department. But real-world script exports don't always
# use UTF-8: Windows tools often default to cp1252, some legacy systems export as latin-1, and a
# stray non-UTF-8 byte anywhere in the file used to crash the whole pipeline with a raw unhandled
# UnicodeDecodeError instead of a clean, actionable message.
def read_text_file(filepath: str) -> str:
    """Tries common encodings in order; latin-1 is the final fallback because it can decode ANY
    byte sequence without raising (every byte 0-255 maps to a character) — worse than perfect
    fidelity for unusual characters, but infinitely better than crashing the whole run."""
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(filepath, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:  # should be unreachable
        return f.read()


# ============================= File-pairing helpers (shared by Training + Deployment) =============================
# Auto-pairing lets a user drop a pile of mixed activation/rollback files into one upload box
# instead of manually matching act[1] with rb[1], act[2] with rb[2], etc. Role is guessed from
# the filename; pairs are matched by stripping the role words out and comparing what's left.
def classify_file_role(filename: str) -> str:
    """Filename keywords decide activation vs rollback — broadened beyond the literal words
    'activation'/'rollback' since different departments use different terms for the same concept
    (RF: Activation/Rollback, Core teams have been seen using Implement/Execute/Rollback, etc.).
    Rollback-family words are checked FIRST: 'deactivate' contains 'activat' as a substring, so
    checking activation-family words first would misclassify it — this ordering avoids that trap.
    This is a fast, free first pass, not the only signal — see classify_file_role_from_content
    for the fallback when a filename gives no hint at all."""
    n = filename.lower()
    if re.search(r"rollback|revert|undo|deactivat|\bdeact\b|disable|remove|reverse|cancel|\brb\b", n):
        return "rollback"
    if re.search(r"activat|implement|enable|deploy|execut|apply|\bact\b", n):
        return "activation"
    return "unknown"


# Deterministic, content-based fallback — no filename convention required at all. Uses this
# project's own established convention (documented since the very first requirements): a
# parameter value ending in '-1' means 'turn on' (activation), '-0' means 'turn off' (rollback).
# Whichever suffix clearly dominates the file's content decides its role. This is plain pattern
# counting, not an AI call — keeps Deployment's zero-AI-calls guarantee intact while still not
# depending on how someone happened to name the file.
_DIRECTION_SUFFIX_RE = re.compile(r"-([01])\s*[,;\s]")


def classify_file_role_from_content(text: str) -> str:
    suffixes = _DIRECTION_SUFFIX_RE.findall(text)
    if not suffixes:
        return "unknown"
    ones, zeros = suffixes.count("1"), suffixes.count("0")
    if ones == zeros:
        return "unknown"
    return "activation" if ones > zeros else "rollback"


def resolve_file_role(filename: str, text: str) -> tuple:
    """Filename first (fast, free, works when someone named the file sensibly); content-based
    -1/-0 suffix counting as a fallback when the filename gives no signal at all. Returns
    (role, used_content_fallback) so callers can be transparent about which signal was used."""
    role = classify_file_role(filename)
    if role != "unknown":
        return role, False
    return classify_file_role_from_content(text), True


def derive_pair_key(filename: str) -> str:
    n = re.sub(r"\.[^.]+$", "", filename)
    n = re.sub(
        r"activation|activate|implement|enable|deploy|execut(?:e|ed|ion)?|apply|applied|"
        r"rollback|revert|undo|deactivat(?:e|ion)?|\bdeact\b|disable|remove|reverse|cancel|\bact\b|\brb\b",
        "", n, flags=re.IGNORECASE,
    )
    n = re.sub(r"[_\-\s]+", "_", n).strip("_")
    return n.upper()


def auto_pair_extra_files(filepaths: list) -> tuple[list, list]:
    """Given a flat list of uploaded file paths, classify each — filename first, content-based
    -1/-0 suffix fallback second — and pair activation files with rollback files sharing a
    derived key. Returns (pairs, warnings)."""
    acts, rbs, unknowns = {}, {}, []
    content_classified = []
    for fp in filepaths:
        name = os.path.basename(fp)
        text = read_text_file(fp)
        role, used_content = resolve_file_role(name, text)
        key = derive_pair_key(name)
        if role == "activation":
            acts[key] = {"name": name, "text": text}
        elif role == "rollback":
            rbs[key] = {"name": name, "text": text}
        else:
            unknowns.append(name)
        if used_content and role != "unknown":
            content_classified.append(f"{name} (as {role}, from content)")

    pairs, warnings = [], []
    for key, a in acts.items():
        if key in rbs:
            pairs.append({"activation_text": a["text"], "rollback_text": rbs[key]["text"]})
        else:
            warnings.append(f"'{a['name']}' has no matching rollback file — skipped")
    for key, r in rbs.items():
        if key not in acts:
            warnings.append(f"'{r['name']}' has no matching activation file — skipped")
    if content_classified:
        warnings.append(f"Classified by content, not filename: {', '.join(content_classified)}")
    if unknowns:
        warnings.append(f"Could not classify (filename gave no hint, content had no clear -1/-0 majority): {', '.join(unknowns)}")
    return pairs, warnings


def pair_deployment_files_by_content(filepaths: list, generated_code: str) -> tuple:
    """Zero filename dependency: role is decided purely from content (-1/-0 parameter suffix
    majority), and pairing is decided by running every file through the ALREADY-SELECTED
    pattern's generated parser (no extra AI call — it's the same parser about to check the
    files anyway) and matching an activation file to whichever rollback file shares the most
    site IDs. Two files describing the same sites are almost certainly the true pair, regardless
    of what either happens to be named."""
    files_data = []
    for fp in filepaths:
        name = os.path.basename(fp)
        text = read_text_file(fp)
        role = classify_file_role_from_content(text)
        try:
            sites = frozenset(l["siteId"] for l in run_generated_parser(generated_code, text) if l.get("siteId"))
        except Exception:
            sites = frozenset()
        files_data.append({"name": name, "text": text, "role": role, "sites": sites})

    acts = [f for f in files_data if f["role"] == "activation"]
    rbs = [f for f in files_data if f["role"] == "rollback"]
    unknowns = [f for f in files_data if f["role"] == "unknown"]

    pairs, warnings, used = [], [], set()
    for a in acts:
        best_idx, best_overlap = None, 0
        for i, r in enumerate(rbs):
            if i in used:
                continue
            overlap = len(a["sites"] & r["sites"])
            if overlap > best_overlap:
                best_overlap, best_idx = overlap, i
        if best_idx is not None:
            r = rbs[best_idx]
            used.add(best_idx)
            label = f"{a['name']} <--> {r['name']}"
            pairs.append({
                "label": label, "activation_name": a["name"], "rollback_name": r["name"],
                "activation_text": a["text"], "rollback_text": r["text"],
            })
        else:
            warnings.append(f"'{a['name']}' has no rollback file with overlapping sites — skipped")

    for i, r in enumerate(rbs):
        if i not in used:
            warnings.append(f"'{r['name']}' has no activation file with overlapping sites — skipped")
    if unknowns:
        warnings.append(f"Could not determine role from content (-1/-0 suffix count was ambiguous or absent): {', '.join(u['name'] for u in unknowns)}")
    return pairs, warnings


def load_pattern_from_upload(filepath: str) -> tuple:
    """Parses an uploaded pattern JSON file (same shape as one record from learned_patterns.json)
    and returns (generated_code_or_None, diagnostic_or_None, display_name)."""
    try:
        record = json.loads(read_text_file(filepath))
    except (json.JSONDecodeError, UnicodeError, OSError) as e:
        return None, f"Uploaded file is not valid JSON: {e}", None
    if not isinstance(record, dict):
        return None, "Uploaded pattern file must be a JSON object (one saved pattern record).", None
    code, diagnostic = usable_generated_code(record)
    return code, diagnostic, record.get("name", "uploaded pattern")


def _token_values():
    """Returns the four numbers shown in the shared token-stat boxes at the top of the GUI.
    Called via a .then() follow-up after training/checking, not as a direct output of those
    long-running generators — see the .click().then() wiring near the bottom of this file for
    why (keeping it separate avoids duplicate loading overlays on the token boxes)."""
    cost = estimate_cost_usd()
    return (
        TOKEN_USAGE["input_tokens"], TOKEN_USAGE["output_tokens"], TOKEN_USAGE["cached_input_tokens"],
        f"${cost:.4f}",
    )


# ============================= Pattern persistence =============================
DEFAULT_PATTERNS_FILE = str(pathlib.Path(__file__).resolve().parent / "learned_patterns.json")
PATTERNS_FILE = os.environ.get("PATTERNS_FILE", DEFAULT_PATTERNS_FILE)


def load_saved_patterns() -> list:
    if not os.path.exists(PATTERNS_FILE):
        return []
    with open(PATTERNS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def usable_generated_code(pattern_record: dict) -> tuple:
    """Returns (generated_code_or_None, diagnostic_message). The diagnostic distinguishes two
    different failure modes that both result in 'no usable parser', since they need different
    fixes: the Teacher never approving the pattern at all (retrain with different samples) vs.
    a pattern that WAS approved but whose generated code failed its own validation (the format
    is likely too irregular for regex/string-based parsing, or needs more/better extra samples)."""
    validation = pattern_record.get("codegen_validation") or {}
    code = pattern_record.get("generated_code")
    if validation.get("ok"):
        return code, None
    if code and not validation.get("ok"):
        reason = validation.get("error") or "validation failed for an unrecorded reason"
        rate = validation.get("extraction_rate")
        rate_note = f" (recognized {rate:.0%} of test lines)" if rate is not None else ""
        return None, (
            f"a parser WAS generated for this pattern, but it failed its own validation during "
            f"training{rate_note}: {reason} Retrain with more/different extra samples, or check "
            f"the 'Generated parser' code shown after training to see why it's unreliable."
        )
    return None, (
        "this pattern was never approved by the Teacher during training, so no parser was "
        "generated at all. Retrain with samples the Teacher can validate successfully."
    )


def save_pattern(name: str, pattern: dict, confidence: float, rounds_used: int,
                  generated_code: Optional[str] = None, codegen_validation: Optional[dict] = None) -> list:
    from datetime import datetime, timezone
    patterns = load_saved_patterns()
    patterns.append({
        "name": name,
        "pattern": pattern,
        "confidence": confidence,
        "rounds_used": rounds_used,
        "generated_code": generated_code,
        "codegen_validation": codegen_validation,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    })
    with open(PATTERNS_FILE, "w", encoding="utf-8") as f:
        json.dump(patterns, f, indent=2)
    return patterns


# ============================= Deployment: deterministic checking (generated parser + checks) =============================
_IDENTIFIER_PARAM_RE = re.compile(r"^(nrducellid|localcellid|cellid)$", re.IGNORECASE)


def file_stats(lines: list) -> dict:
    active = [l for l in lines if not l["isBlank"] and not l["isComment"]]
    comments = [l for l in lines if l["isComment"]]
    sites, cells = set(), set()
    for l in lines:
        if l.get("siteId"):
            sites.add(l["siteId"])
            if l.get("cellId") is not None:
                cells.add((l["siteId"], l["cellId"]))
    missing_semi = [l for l in active if l.get("command") and not l.get("hasSemicolon")]
    return {
        "total_lines": len(lines), "active_lines": len(active),
        "unique_sites": sites, "unique_cells": cells,
        "comment_count": len(comments), "has_comments": len(comments) > 0,
        "missing_semicolons": missing_semi, "spread_sites": find_spread_sites(lines),
    }


def find_spread_sites(lines: list) -> list:
    seq = [l["siteId"] for l in lines if l.get("siteId")]
    closed, spread, prev = set(), set(), None
    for s in seq:
        if prev is not None and s != prev:
            closed.add(prev)
        if s in closed and s != prev:
            spread.add(s)
        prev = s
    return list(spread)


def find_same_value_matches(act_lines: list, rb_lines: list) -> list:
    """Same value in both activation and rollback for a given site+cell+command+param = the
    rollback wouldn't actually change anything there. RST-like commands and identifier fields
    (site/cell id params) are excluded since those are expected to match, not a bug."""
    def build_map(lines):
        m = {}
        for l in lines:
            if l["isBlank"] or l["isComment"]:
                continue
            if l.get("command") and re.match(r"^RST", l["command"], re.IGNORECASE):
                continue
            for p in l.get("params", []):
                if _IDENTIFIER_PARAM_RE.match(p.get("name", "")):
                    continue
                key = (l.get("siteId"), l.get("cellId"), l.get("command"), p.get("name"))
                m[key] = {"value": p.get("value"), "line": l["lineNum"]}
        return m

    map_a, map_b = build_map(act_lines), build_map(rb_lines)
    results = []
    for key, a_entry in map_a.items():
        b_entry = map_b.get(key)
        if b_entry and a_entry["value"] == b_entry["value"]:
            site_id, cell_id, command, param_name = key
            results.append({
                "siteId": site_id, "cellId": cell_id, "command": command, "paramName": param_name,
                "value": a_entry["value"], "actLine": a_entry["line"], "rbLine": b_entry["line"],
            })
    return results


def build_parameter_summary(act_lines: list, rb_lines: list) -> list:
    """Every parameter's current (rollback) value vs new (activation) value, deduped down to
    distinct combinations rather than one row per site/cell — matches the browser tool's
    /parameter command. Identifier fields (site/cell id params) are excluded, same as elsewhere."""
    def build_map(lines):
        m = {}
        for l in lines:
            if l["isBlank"] or l["isComment"]:
                continue
            for p in l.get("params", []):
                if _IDENTIFIER_PARAM_RE.match(p.get("name", "")):
                    continue
                key = (l.get("siteId"), l.get("cellId"), p.get("name"))
                m[key] = p.get("value")
        return m

    map_a, map_b = build_map(act_lines), build_map(rb_lines)
    all_keys = set(map_a.keys()) | set(map_b.keys())
    agg = {}
    for key in all_keys:
        _, _, param_name = key
        current = map_b.get(key, "—")
        new = map_a.get(key, "—")
        agg_key = (param_name, current, new)
        agg[agg_key] = agg.get(agg_key, 0) + 1

    results = [{"paramName": k[0], "current": k[1], "new": k[2], "count": v} for k, v in agg.items()]
    results.sort(key=lambda r: r["paramName"])
    return results


def run_check(act_lines: list, rb_lines: list) -> dict:
    a, b = file_stats(act_lines), file_stats(rb_lines)
    site_mismatch_ab = sorted(a["unique_sites"] - b["unique_sites"])
    site_mismatch_ba = sorted(b["unique_sites"] - a["unique_sites"])
    cell_mismatch_ab = sorted(a["unique_cells"] - b["unique_cells"])
    cell_mismatch_ba = sorted(b["unique_cells"] - a["unique_cells"])
    same_value_matches = find_same_value_matches(act_lines, rb_lines)
    parameter_summary = build_parameter_summary(act_lines, rb_lines)
    return {
        "activation": {
            "total_lines": a["total_lines"], "active_lines": a["active_lines"],
            "unique_sites": len(a["unique_sites"]), "unique_cells": len(a["unique_cells"]),
            "comment_count": a["comment_count"], "has_comments": a["has_comments"],
            "missing_semicolons": [l["lineNum"] for l in a["missing_semicolons"]],
            "spread_sites": a["spread_sites"],
        },
        "rollback": {
            "total_lines": b["total_lines"], "active_lines": b["active_lines"],
            "unique_sites": len(b["unique_sites"]), "unique_cells": len(b["unique_cells"]),
            "comment_count": b["comment_count"], "has_comments": b["has_comments"],
            "missing_semicolons": [l["lineNum"] for l in b["missing_semicolons"]],
            "spread_sites": b["spread_sites"],
        },
        "site_mismatch": {"activation_only": site_mismatch_ab, "rollback_only": site_mismatch_ba},
        "cell_mismatch": {"activation_only": cell_mismatch_ab, "rollback_only": cell_mismatch_ba},
        "duplicate_values": same_value_matches,
        "parameter_summary": parameter_summary,
        "clean": (
            not site_mismatch_ab and not site_mismatch_ba and not cell_mismatch_ab and not cell_mismatch_ba
            and not a["missing_semicolons"] and not b["missing_semicolons"]
            and not a["spread_sites"] and not b["spread_sites"] and not same_value_matches
        ),
    }


# ============================= Deployment: check using the generated parser only =============================
# No AI-fallback extractor — deployment is 100% deterministic once a pattern has an approved,
# validated generated parser. If that parser isn't available, this fails clearly rather than
# silently falling back to a slower/costlier path.
def run_deployment_check_stream(activation_text: str, rollback_text: str, generated_code: Optional[str] = None):
    """Generator: yields ('progress', message) throughout, then ('done', result_dict) at the end."""
    if not generated_code:
        raise ValueError(
            "This pattern has no working generated parser — deployment checks require one. "
            "Retrain the pattern (the Teacher must approve it before code generation runs), "
            "or pick a different saved pattern that has a validated parser."
        )

    yield ("progress", "Parsing activation script with the generated parser...")
    act_lines = parse_with_generated_code(activation_text, generated_code)
    yield ("progress", "Parsing rollback script with the generated parser...")
    rb_lines = parse_with_generated_code(rollback_text, generated_code)
    yield ("progress", "Running checks...")
    yield ("done", run_check(act_lines, rb_lines))


def run_deployment_check(activation_text: str, rollback_text: str, generated_code: Optional[str] = None) -> dict:
    """Blocking wrapper around run_deployment_check_stream, for the REST endpoint."""
    result = None
    for kind, payload in run_deployment_check_stream(activation_text, rollback_text, generated_code):
        if kind == "done":
            result = payload
    return result


def format_check_markdown(result: dict) -> str:
    """Despite the name (kept for API compatibility), this returns pure HTML, not Markdown —
    needed so each batch's output can be safely nested inside its own wrapper <div> for
    per-batch screenshot capture without breaking on Markdown-inside-HTML parsing quirks."""
    def badge(is_clear: bool) -> str:
        if is_clear:
            return '<span style="color:#1a7f37;font-weight:600;">🟢 Clear</span>'
        return '<span style="color:#cf222e;font-weight:600;">🔴 Issue found</span>'

    def html_table(headers, rows, row_styles=None):
        th_cells = "".join(f'<th style="border:1px solid #ddd;padding:6px 10px;background:#f6f8fa;text-align:left;">{h}</th>' for h in headers)
        body_rows = ""
        for i, row in enumerate(rows):
            style = (row_styles[i] if row_styles else "") or ""
            td_cells = "".join(f'<td style="border:1px solid #ddd;padding:6px 10px;">{c}</td>' for c in row)
            body_rows += f'<tr style="{style}">{td_cells}</tr>'
        return f'<table style="border-collapse:collapse;width:100%;font-size:14px;"><thead><tr>{th_cells}</tr></thead><tbody>{body_rows}</tbody></table>'

    parts = [f'<h3>{"✅ Clean" if result["clean"] else "⚠️ Issues found"}</h3>']

    stats_table = html_table(
        ["Check items", "Activation", "Rollback"],
        [
            ["Total lines", result["activation"]["total_lines"], result["rollback"]["total_lines"]],
            ["Active lines", result["activation"]["active_lines"], result["rollback"]["active_lines"]],
            ["Unique sites", result["activation"]["unique_sites"], result["rollback"]["unique_sites"]],
            ["Unique cells", result["activation"]["unique_cells"], result["rollback"]["unique_cells"]],
            ["Comments (//)", result["activation"]["comment_count"], result["rollback"]["comment_count"]],
        ],
    )

    sm = result["site_mismatch"]
    sm_clear = not (sm["activation_only"] or sm["rollback_only"])
    sm_detail = "none" if sm_clear else f"activation-only <code>{sm['activation_only']}</code>, rollback-only <code>{sm['rollback_only']}</code>"

    cm = result["cell_mismatch"]
    cm_clear = not (cm["activation_only"] or cm["rollback_only"])
    cm_detail = "none" if cm_clear else f"activation-only <code>{cm['activation_only']}</code>, rollback-only <code>{cm['rollback_only']}</code>"

    ms_a, ms_b = result["activation"]["missing_semicolons"], result["rollback"]["missing_semicolons"]
    ms_clear = not (ms_a or ms_b)
    ms_detail = "none" if ms_clear else f"activation lines <code>{ms_a or 'none'}</code>, rollback lines <code>{ms_b or 'none'}</code>"

    sp_a, sp_b = result["activation"]["spread_sites"], result["rollback"]["spread_sites"]
    sp_clear = not (sp_a or sp_b)
    sp_detail = "none" if sp_clear else f"activation <code>{sp_a or 'none'}</code>, rollback <code>{sp_b or 'none'}</code>"

    dv = result["duplicate_values"]
    dv_clear = not dv
    dv_detail = "none" if dv_clear else f"{len(dv)} found — see table below"

    status_table = html_table(
        ["Check items", "Status", "Details"],
        [
            ["Site mismatch", badge(sm_clear), sm_detail],
            ["Cell mismatch", badge(cm_clear), cm_detail],
            ["Missing semicolons", badge(ms_clear), ms_detail],
            ["Spread sites", badge(sp_clear), sp_detail],
            ["Duplicate values", badge(dv_clear), dv_detail],
        ],
    )

    parts.append(
        '<div style="display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start;">'
        f'<div style="flex:1;min-width:280px;">{stats_table}</div>'
        f'<div style="flex:1;min-width:280px;">{status_table}</div>'
        '</div>'
    )

    if dv:
        dv_rows = [[d["siteId"], d["cellId"], d["paramName"], d["value"], d["actLine"], d["rbLine"]] for d in dv[:50]]
        dv_table = html_table(["Site", "Cell", "Param", "Value", "Act line", "RB line"], dv_rows)
        parts.append(f'<p><b>Duplicate values ({len(dv)} found — same value in both scripts):</b></p>{dv_table}')
        if len(dv) > 50:
            parts.append(f'<p><i>...and {len(dv) - 50} more.</i></p>')

    # Parameters table: color-coded by what the current->new relationship actually means —
    # red = same value in both (the rollback wouldn't change anything, likely a real bug),
    # amber = missing from one script entirely, green = a normal, healthy reversal
    ps = result.get("parameter_summary") or []

    def param_status(current, new):
        if current == new and current != "—":
            return "same", '<span style="color:#cf222e;font-weight:600;">🔴 Same value</span>', "background:#ffebe9;"
        if current == "—" or new == "—":
            return "missing", '<span style="color:#9a6700;font-weight:600;">🟡 Missing in one</span>', "background:#fff8e5;"
        return "ok", '<span style="color:#1a7f37;font-weight:600;">🟢 Reversed</span>', ""

    param_rows, param_styles = [], []
    for p in ps[:100]:
        _, status_badge, row_style = param_status(p["current"], p["new"])
        entries_label = f"{p['count']} site/cell entr{'y' if p['count'] == 1 else 'ies'}"
        param_rows.append([p["paramName"], p["current"], p["new"], status_badge, entries_label])
        param_styles.append(row_style)
    param_table = html_table(["Parameter", "Current (rollback)", "New (activation)", "Status", "Applies to"], param_rows, param_styles)

    parts.append(f'<p><b>Parameters — current (rollback) vs new (activation), {len(ps)} distinct combination(s):</b></p>{param_table}')
    if len(ps) > 100:
        parts.append(f'<p><i>...and {len(ps) - 100} more.</i></p>')

    return "".join(parts)


# ============================= Training tab: Gradio wrapper functions =============================
# Everything below runs in the browser session, not the graph — these functions translate GUI
# inputs into the training-graph calls above and stream results back as live UI updates.
def resolve_credentials(provider, anthropic_key, anthropic_url, anthropic_model, huawei_key, huawei_url, huawei_model):
    """Turns the shared credential-panel inputs into (provider, api_key, base_url, model) for
    whichever provider is actually selected — the fields for the OTHER provider are ignored."""
    provider = (provider or "anthropic").lower()
    if provider == "anthropic":
        return provider, (anthropic_key or None), (anthropic_url or None), (anthropic_model or None)
    return provider, (huawei_key or None), (huawei_url or None), (huawei_model or None)


def gradio_train(activation_file, rollback_file, extra_files, provider, anthropic_key, anthropic_url, anthropic_model,
                  huawei_key, huawei_url, huawei_model, max_rounds, confidence_threshold, progress=gr.Progress()):
    """Generator driving the graph via .stream() so the UI updates live, round by round,
    instead of blocking silently until the whole thing finishes. Also drives a visual progress
    bar (gr.Progress) alongside the text log, estimated from max_rounds since the actual number
    of rounds used can be lower if the Teacher approves early."""
    if not activation_file or not rollback_file:
        yield ("⚠️ Upload both a sample activation and a sample rollback file first.", "", "", "", None)
        return

    progress(0, desc="Loading files...")
    activation_text = read_text_file(activation_file)
    rollback_text = read_text_file(rollback_file)

    extra_samples, warnings = [], []
    if extra_files:
        extra_samples, warnings = auto_pair_extra_files(extra_files)

    log_lines = [f"Loaded training pair. {len(extra_samples)} extra sample pair(s) auto-paired for testing."]
    for w in warnings:
        log_lines.append(f"⚠️ {w}")

    if not extra_samples:
        log_lines.append(
            "\n❌ Can't train without at least one extra sample pair — the Teacher needs an "
            "independent activation+rollback pair to test the pattern against. Drop one into "
            "the 'Extra sample files' box above and try again."
        )
        yield ("\n".join(log_lines), "", "", "", None)
        return

    yield ("\n".join(log_lines), "", "", "", None)

    resolved_provider, resolved_key, resolved_url, resolved_model = resolve_credentials(
        provider, anthropic_key, anthropic_url, anthropic_model, huawei_key, huawei_url, huawei_model)

    initial_state: TrainState = {
        "activation_text": activation_text, "rollback_text": rollback_text,
        "extra_samples": extra_samples, "candidate_pattern": None, "round": 1,
        "max_rounds": int(max_rounds), "confidence_threshold": float(confidence_threshold),
        "approved": False, "confidence": 0.0, "per_sample_results": [],
        "teacher_feedback": None, "history": [],
        "provider": resolved_provider, "api_key": resolved_key,
        "base_url": resolved_url, "model": resolved_model,
        "generated_code": None, "codegen_validation": None,
    }

    # Trainer + Teacher per round, plus one final Code generator step if approved — an estimate,
    # since the real round count depends on when (or if) the Teacher approves.
    total_steps = int(max_rounds) * 2 + 1
    steps_done = 0

    final_state = None
    try:
        for step in TRAINER_TEACHER_GRAPH.stream(initial_state, stream_mode="updates"):
            node_name, update = next(iter(step.items()))
            if node_name == "trainer":
                rnd = update["history"][-1]["round"]
                log_lines.append(f"Round {rnd} — Trainer produced a candidate pattern.")
                steps_done += 1
                progress(min(steps_done / total_steps, 0.99), desc=f"Round {rnd}: Trainer proposing pattern")
            elif node_name == "teacher":
                entry = update["history"][-1]
                status = "✅ APPROVED" if entry["approved"] else "❌ not yet"
                log_lines.append(f"Round {entry['round']} — Teacher confidence: {entry['confidence']:.0f}% ({status})")
                if not entry["approved"] and update.get("teacher_feedback"):
                    log_lines.append(f"    feedback: {update['teacher_feedback'][:200]}")
                steps_done += 1
                progress(min(steps_done / total_steps, 0.99), desc=f"Round {entry['round']}: Teacher validating ({entry['confidence']:.0f}%)")
            elif node_name == "increment_round":
                log_lines.append(f"Refining and retesting (round {update['round']})...")
                progress(min(steps_done / total_steps, 0.99), desc=f"Refining for round {update['round']}...")
            elif node_name == "convert_pattern_into_code":
                steps_done = total_steps
                progress(0.99, desc="Generating and validating parser code...")
            final_state = {**initial_state, **update} if final_state is None else {**final_state, **update}
            yield ("\n".join(log_lines), "", "", "", None)
    except Exception as e:
        log_lines.append(f"\n❌ Training failed: {type(e).__name__}: {e}")
        yield ("\n".join(log_lines), "", "", "", None)
        return

    progress(1.0, desc="Done")
    pattern_json = json.dumps(final_state.get("candidate_pattern"), indent=2)
    validation = final_state.get("codegen_validation")
    codegen_status = ""
    if validation is not None:
        codegen_status = (
            f"  \n**Generated parser:** {'✅ working' if validation.get('ok') else '⚠️ not usable'} "
            f"({validation.get('extraction_rate', 0):.0%} extraction rate on {validation.get('lines_tested', 0)} test lines)"
        )
        if not validation.get("ok") and validation.get("error"):
            codegen_status += f"\n\n_{validation['error']}_"
    summary = (
        f"**Approved:** {final_state.get('approved')}  \n"
        f"**Confidence:** {final_state.get('confidence', 0):.0f}%  \n"
        f"**Rounds used:** {final_state.get('round')}"
        f"{codegen_status}"
    )
    log_lines.append("\nDone.")
    last_result = {
        "pattern": final_state.get("candidate_pattern"),
        "confidence": final_state.get("confidence", 0),
        "rounds_used": final_state.get("round"),
        "approved": final_state.get("approved"),
        "generated_code": final_state.get("generated_code"),
        "codegen_validation": final_state.get("codegen_validation"),
    }
    generated_code_display = final_state.get("generated_code") or "# No pattern was approved, so no parser was generated."
    yield ("\n".join(log_lines), summary, pattern_json, generated_code_display, last_result)


# ============================= Deployment tab: Gradio wrapper functions =============================
# Everything below is 100% deterministic once a pattern has a validated generated parser — no AI
# calls happen anywhere in this section, which is exactly the point of Phase 2.
def do_save_pattern(name, last_result):
    if not last_result or not last_result.get("pattern"):
        return "⚠️ Nothing to save yet — run training first.", gr.update()
    if not last_result.get("approved"):
        return "⚠️ This pattern was never approved by the Teacher — saving isn't recommended, but proceeding since you asked.", gr.update()
    if not name:
        name = f"pattern-{last_result.get('confidence', 0):.0f}pct"
    save_pattern(
        name, last_result["pattern"], last_result["confidence"], last_result["rounds_used"],
        generated_code=last_result.get("generated_code"),
        codegen_validation=last_result.get("codegen_validation"),
    )
    patterns = load_saved_patterns()
    table = refresh_saved_patterns_table()
    return f"✅ Saved as '{name}'. {len(patterns)} pattern(s) in {PATTERNS_FILE}.", gr.update(value=table)


def gradio_check(pattern_name, pattern_file, deployment_files):
    """Deployment: any user picks a saved pattern (or imports one directly), uploads their real
    files — one or many activation/rollback pairs at once, auto-paired by CONTENT (role from
    -1/-0 suffix majority, pairing from shared site IDs via the selected pattern's own parser) —
    and gets a check report per pair. Filenames are never inspected at all. 100% deterministic:
    no AI calls happen here, it only runs the pattern's generated parser. If that parser isn't
    available, this fails clearly upfront."""
    if not deployment_files:
        yield ("⚠️ Upload at least one activation + rollback file pair.", "", "")
        return

    if pattern_file:
        generated_code, diagnostic, source_name = load_pattern_from_upload(pattern_file)
        source_desc = f"imported pattern file ('{source_name}')"
    elif pattern_name:
        match = next((p for p in load_saved_patterns() if p["name"] == pattern_name), None)
        if not match:
            yield (f"⚠️ Pattern '{pattern_name}' not found — try refreshing the pattern list.", "", "")
            return
        generated_code, diagnostic = usable_generated_code(match)
        source_desc = f"saved pattern '{pattern_name}'"
    else:
        yield ("⚠️ Select a saved pattern, or import a pattern file.", "", "")
        return

    if not generated_code:
        yield (f"⚠️ Can't use this pattern — {diagnostic}", "", "")
        return

    pairs, warnings = pair_deployment_files_by_content(deployment_files, generated_code)
    log_lines = [f"Using {source_desc}.", f"{len(pairs)} activation/rollback pair(s) matched by content (site overlap), filenames ignored."]
    for w in warnings:
        log_lines.append(f"⚠️ {w}")
    if not pairs:
        log_lines.append("\n❌ No valid activation+rollback pairs found among the uploaded files.")
        yield ("\n".join(log_lines), "", "")
        return
    yield ("\n".join(log_lines), "", "")

    all_markdown, all_results = [], {}
    for pair in pairs:
        log_lines.append(f"\nChecking batch: {pair['label']}...")
        yield ("\n".join(log_lines), "", "")
        try:
            result = run_deployment_check(pair["activation_text"], pair["rollback_text"], generated_code)
            all_results[pair["label"]] = result
            batch_id = "batch-result-" + re.sub(r"[^a-zA-Z0-9_-]", "_", pair["label"])
            batch_content = format_check_markdown(result)
            all_markdown.append(
                f'<div id="{batch_id}" style="margin-bottom:28px;padding-bottom:18px;border-bottom:1px solid #e1e4e8;">'
                '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
                f'<h2 style="margin:0;">Batch: {pair["label"]}</h2>'
                f'<button onclick="captureElementAsImage(\'{batch_id}\')" '
                'style="padding:6px 14px;border-radius:6px;border:1px solid #ccc;background:#f6f8fa;'
                'cursor:pointer;font-size:13px;white-space:nowrap;">📷 Save this batch</button>'
                '</div>'
                f'{batch_content}</div>'
            )
        except Exception as e:
            log_lines.append(f"  ❌ Failed: {type(e).__name__}: {e}")
            all_markdown.append(f"<h2>Batch: {pair['label']}</h2><p>❌ Check failed: {type(e).__name__}: {e}</p>")

    log_lines.append("\nDone.")
    yield ("\n".join(log_lines), "".join(all_markdown), json.dumps(all_results, indent=2))


def refresh_pattern_choices():
    return gr.update(choices=[p["name"] for p in load_saved_patterns()])


def refresh_saved_patterns_table():
    patterns = load_saved_patterns()
    return [[p["name"], f"{p['confidence']:.0f}%", p["rounds_used"], p["saved_at"][:19]] for p in patterns]


def gradio_test_connectivity(provider, anthropic_key, anthropic_url, anthropic_model, huawei_key, huawei_url, huawei_model):
    resolved_provider, resolved_key, resolved_url, resolved_model = resolve_credentials(
        provider, anthropic_key, anthropic_url, anthropic_model, huawei_key, huawei_url, huawei_model)
    ok, msg = test_api_connectivity(resolved_provider, resolved_key, resolved_url, resolved_model)
    icon = "🟢" if ok else "🔴"
    label = "Reachable" if ok else "Not reachable"
    return f"{icon} **{label}** — {msg}"


# ============================= Gradio Blocks UI layout =============================
# Everything from here down is pure layout + event wiring — no new business logic. Component
# variable names match what's referenced in the .click()/.change()/.load() calls near the bottom.
with gr.Blocks(title="AI Memorize Your Pattern") as demo:
    gr.Markdown("<h1 style='text-align:center;color:#5B9BD5;'>AI Memorize Your Pattern</h1>")

    gr.Markdown(
        "<p style='text-align:center;'><b>Description:</b> learns how your telecom RF "
        "activation/rollback script format works from real examples, then checks new script "
        "pairs for site/cell mismatches, duplicate values, missing semicolons, and other "
        "rollback risks — without anyone hand-writing a parser for each vendor's format.</p>"
    )

    gr.HTML(
        "<div style='text-align:center; margin: 4px 0 16px;'>"
        "<svg width='100%' height='150' viewBox='0 0 640 150' xmlns='http://www.w3.org/2000/svg' style='max-width:640px;'>"
        "<rect width='640' height='150' fill='#ffffff' rx='8'/>"
        "<defs><marker id='arrowhead' viewBox='0 0 10 10' refX='8' refY='5' markerWidth='6' markerHeight='6' orient='auto-start-reverse'>"
        "<path d='M2 1L8 5L2 9' fill='none' stroke='#888888' stroke-width='1.5'/></marker></defs>"
        "<rect x='20' y='35' width='170' height='60' rx='10' fill='#DCEEFB' stroke='#5B9BD5' stroke-width='1.5'/>"
        "<text x='105' y='60' text-anchor='middle' font-size='14' font-weight='600' fill='#1F4E79' font-family='sans-serif'>Trainer</text>"
        "<text x='105' y='80' text-anchor='middle' font-size='11' fill='#2E5D8A' font-family='sans-serif'>Learns the pattern</text>"
        "<line x1='190' y1='65' x2='235' y2='65' stroke='#999999' stroke-width='1.5' marker-end='url(#arrowhead)'/>"
        "<rect x='235' y='35' width='170' height='60' rx='10' fill='#DAF0E6' stroke='#4CAF8A' stroke-width='1.5'/>"
        "<text x='320' y='60' text-anchor='middle' font-size='14' font-weight='600' fill='#1B5E42' font-family='sans-serif'>Teacher</text>"
        "<text x='320' y='80' text-anchor='middle' font-size='11' fill='#2C7A57' font-family='sans-serif'>Validates the pattern</text>"
        "<line x1='405' y1='65' x2='450' y2='65' stroke='#999999' stroke-width='1.5' marker-end='url(#arrowhead)'/>"
        "<rect x='450' y='35' width='170' height='60' rx='10' fill='#EEE3F8' stroke='#8E6FC2' stroke-width='1.5'/>"
        "<text x='535' y='60' text-anchor='middle' font-size='14' font-weight='600' fill='#4B2E7A' font-family='sans-serif'>Code generator</text>"
        "<text x='535' y='80' text-anchor='middle' font-size='11' fill='#5F3E96' font-family='sans-serif'>Writes a fast parser</text>"
        "<text x='320' y='122' text-anchor='middle' font-size='11' fill='#888888' font-family='sans-serif'>"
        "↻ Teacher sends feedback back to Trainer until approved</text>"
        "</svg></div>"
    )

    with gr.Accordion("📖 Instructions for new users", open=False):
        gr.Markdown(
            "1. **Train** (Training tab): upload one confirmed-correct activation + rollback "
            "sample pair, plus at least one extra sample pair (required — the Teacher needs "
            "something independent to validate against). Click **Learn pattern from these samples**.\n"
            "2. Watch the live progress as Trainer and Teacher go back and forth. Once approved, "
            "review the **Generated parser** code shown below the result, name the pattern, and "
            "click **Save this pattern**.\n"
            "3. **Check** (Deployment tab): pick your saved pattern from the dropdown, or import "
            "a pattern `.json` file directly. Upload your real activation/rollback script(s) — "
            "one pair or several at once, they're auto-paired by filename — and click **Run check**.\n"
            "4. Read the summary report: line/site/cell counts, comments, site and cell mismatches, "
            "missing semicolons, duplicate values, and spread-site implementation, per batch.\n"
            "5. Use **🔌 Test API connectivity** below any time to confirm your provider is reachable "
            "before training — deployment checks themselves need no AI connection at all."
        )

    with gr.Row():
        input_tok_out = gr.Number(label="Input Tokens", value=0, interactive=False)
        output_tok_out = gr.Number(label="Output Tokens", value=0, interactive=False)
        cached_tok_out = gr.Number(label="Cached Input Tokens", value=0, interactive=False)
        cost_out = gr.Textbox(label="Est. Cost (session total)", value="$0.0000", interactive=False)
    gr.Markdown(
        "<sub>Cost is computed per call using the actual model that served it — each Anthropic and "
        "Huawei-internal model has its own rate. Totals are shared across both tabs below.</sub>"
    )

    gr.Markdown(
        "<p style='text-align:center;'><b>AI provider credentials</b> — required, entered here each "
        "session (never read from .env). Used for training and the connectivity test below. "
        "Deployment checks need none of this.</p>"
    )
    with gr.Row():
        shared_provider_in = gr.Dropdown(choices=["anthropic", "huawei"], value="anthropic", label="Provider", scale=1)

    with gr.Row(visible=True) as anthropic_creds_row:
        anthropic_key_in = gr.Textbox(label="Anthropic API key (required)", type="password", scale=2)
        anthropic_url_in = gr.Textbox(label="Base URL (required)", value="https://api.anthropic.com", scale=2)
        anthropic_model_in = gr.Dropdown(choices=ANTHROPIC_MODEL_CHOICES, value=ANTHROPIC_MODEL_CHOICES[0][1],
                                          allow_custom_value=True, label="Model", scale=2)

    with gr.Row(visible=False) as huawei_creds_row:
        huawei_key_in = gr.Textbox(label="API key (required)", type="password", scale=2)
        huawei_url_in = gr.Textbox(label="Base URL (required)", placeholder="http://models.ascend.huawei.com/v1", scale=2)
        huawei_model_in = gr.Dropdown(choices=HUAWEI_MODEL_CHOICES, value=HUAWEI_MODEL_CHOICES[0],
                                       allow_custom_value=True, label="Model", scale=2)

    shared_provider_in.change(
        lambda p: (gr.update(visible=(p == "anthropic")), gr.update(visible=(p == "huawei"))),
        inputs=[shared_provider_in], outputs=[anthropic_creds_row, huawei_creds_row],
    )

    with gr.Row():
        test_connectivity_btn = gr.Button("🔌 Test API connectivity", scale=1)
        connectivity_status_out = gr.Markdown("Not tested yet.")

    with gr.Tabs():
        with gr.Tab("🧠 Training"):
            gr.Markdown("## Trainer / Teacher pattern learning\nUpload one confirmed-correct sample pair, "
                        "extra pairs to validate against, and watch the agents work.")

            with gr.Row():
                activation_in = gr.File(label="Sample activation .txt", type="filepath")
                rollback_in = gr.File(label="Sample rollback .txt", type="filepath")
            extra_in = gr.File(label="Extra sample files (mix of activation + rollback .txt, auto-paired by filename)",
                                type="filepath", file_count="multiple")
            with gr.Row():
                max_rounds_in = gr.Number(value=3, precision=0, label="Max rounds")
                threshold_in = gr.Slider(minimum=0, maximum=100, value=90, label="Confidence threshold (%)")
            train_btn = gr.Button("🧠 Learn pattern from these samples", variant="primary")

            progress_out = gr.Textbox(label="Progress", lines=8, interactive=False)
            summary_out = gr.Markdown(label="Result")
            pattern_out = gr.Code(label="Learned pattern (JSON)", language="json")
            codegen_out = gr.Code(label="Generated parser (review before trusting on real files)", language="python")
            last_result_state = gr.State(None)

            with gr.Row():
                save_name_in = gr.Textbox(label="Name this pattern (e.g. 'Ericsson-v1')", scale=3)
                save_btn = gr.Button("💾 Save this pattern", scale=1)
            save_status_out = gr.Markdown()

            gr.Markdown("### Saved patterns")
            saved_patterns_table = gr.Dataframe(
                headers=["name", "confidence", "rounds_used", "saved_at"],
                value=[[p["name"], f"{p['confidence']:.0f}%", p["rounds_used"], p["saved_at"][:19]] for p in load_saved_patterns()],
                interactive=False,
            )

        with gr.Tab("✅ Deployment — check your scripts"):
            gr.Markdown("## Run a check using a saved pattern\nAny user, any files — pick a pattern "
                        "someone already trained (or import one directly), upload your real "
                        "activation/rollback scripts — one pair or several at once — and check them. "
                        "This uses zero AI calls: it only runs the pattern's generated parser.")

            with gr.Row():
                pattern_select_in = gr.Dropdown(choices=[p["name"] for p in load_saved_patterns()],
                                                 label="Saved pattern to use", scale=3)
                refresh_patterns_btn = gr.Button("🔄 Refresh list", scale=1)
            pattern_file_in = gr.File(label="...or import a pattern file (.json) instead — overrides the dropdown if provided",
                                       type="filepath")
            deploy_files_in = gr.File(
                label="Your activation + rollback files (multiple allowed — mix of both, auto-paired by filename)",
                type="filepath", file_count="multiple")
            check_btn = gr.Button("✅ Run check", variant="primary")

            check_progress_out = gr.Textbox(label="Progress", lines=8, interactive=False)
            # loads html2canvas properly (via `head`, which actually executes — unlike a <script>
            # tag placed directly in an HTML component's value, which browsers block for security)
            # and defines one reusable capture function that every per-batch button below calls
            gr.HTML(
                value="",
                head='<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>',
                js_on_load=(
                    "window.captureElementAsImage = async function(elementId) {"
                    "  const el = document.getElementById(elementId);"
                    "  if (!el) { alert('Nothing to capture.'); return; }"
                    "  const canvas = await html2canvas(el, {backgroundColor: '#ffffff', scale: 2, "
                    "    ignoreElements: (node) => node.tagName === 'BUTTON'});"
                    "  const link = document.createElement('a');"
                    "  link.download = elementId + '-' + Date.now() + '.png';"
                    "  link.href = canvas.toDataURL('image/png');"
                    "  link.click();"
                    "};"
                ),
            )
            check_result_out = gr.HTML(label="Check result", elem_id="check-result-output")
            check_json_out = gr.Code(label="Raw result (JSON)", language="json")

    train_btn.click(
        gradio_train,
        inputs=[activation_in, rollback_in, extra_in, shared_provider_in, anthropic_key_in, anthropic_url_in, anthropic_model_in,
                huawei_key_in, huawei_url_in, huawei_model_in, max_rounds_in, threshold_in],
        outputs=[progress_out, summary_out, pattern_out, codegen_out, last_result_state],
        show_progress="minimal",
    ).then(
        _token_values, outputs=[input_tok_out, output_tok_out, cached_tok_out, cost_out],
    )

    save_btn.click(
        do_save_pattern,
        inputs=[save_name_in, last_result_state],
        outputs=[save_status_out, saved_patterns_table],
    )

    refresh_patterns_btn.click(refresh_pattern_choices, outputs=[pattern_select_in])

    test_connectivity_btn.click(
        gradio_test_connectivity,
        inputs=[shared_provider_in, anthropic_key_in, anthropic_url_in, anthropic_model_in, huawei_key_in, huawei_url_in, huawei_model_in],
        outputs=[connectivity_status_out],
    )

    check_btn.click(
        gradio_check,
        inputs=[pattern_select_in, pattern_file_in, deploy_files_in],
        outputs=[check_progress_out, check_result_out, check_json_out],
        show_progress="minimal",
    ).then(
        _token_values, outputs=[input_tok_out, output_tok_out, cached_tok_out, cost_out],
    )

    # re-read learned_patterns.json on every new browser session/window — without this, a new
    # tab only ever shows whatever was on disk at server startup, not patterns saved since then
    demo.load(refresh_saved_patterns_table, outputs=[saved_patterns_table])
    demo.load(refresh_pattern_choices, outputs=[pattern_select_in])


# ============================= Mount Gradio onto the same FastAPI app =============================
# /train, /check, /health, /patterns stay available as plain REST endpoints for other tools/scripts;
# /gui serves the human-friendly interface — both run on the same process, same port.
app = gr.mount_gradio_app(app, demo, path="/gui")


# ============================= Entry point =============================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8787))
    print(f"Starting server on http://localhost:{port}")
    print(f"  REST API docs: http://localhost:{port}/docs")
    print(f"  Gradio GUI:    http://localhost:{port}/gui")
    uvicorn.run(app, host="0.0.0.0", port=port)
