"""Register open-weights backbones served over an OpenAI-compatible endpoint.

Why this exists: the managed gateway used originally refuses the logprobs
parameter on every open-weights model it hosts (verified on all five, on both
the ``generate`` and ``chat`` endpoints). Token-level uncertainty is therefore
unobservable there. It *is* observable on providers that serve the same weights
over an OpenAI-compatible API, and on self-hosted vLLM/SGLang/llama.cpp.

The trick is that ``reactxen.utils.model_inference.watsonx_llm`` routes any model
id beginning with ``openai/`` to ``direct_openai_llm``, stripping only that first
prefix. So registering ``openai/meta-llama/Llama-3.3-70B-Instruct-Turbo`` sends
``meta-llama/Llama-3.3-70B-Instruct-Turbo`` to the OpenAI SDK, which honours
``OPENAI_BASE_URL``. Pointing that at a provider or a local server is all it
takes -- the client patch already requests and returns logprobs.

Usage: put the provider's own key in ``.env`` -- one variable per provider, so
adding one never clobbers a working ``OPENAI_API_KEY``. ``--provider`` then sets
``OPENAI_API_KEY``/``OPENAI_BASE_URL`` for that process only.

    # .env
    TOGETHER_API_KEY=...

    python DigitalTwinsV2/pipeline/openweights_models.py together   # probe first
    python DigitalTwinsV2/scripts/rerun_sweep.py capture --provider together ...

Verify before spending budget: ``probe_logprobs(provider)`` makes one tiny call
per model and reports whether logprobs actually come back. Providers change what
they expose; do not trust this table over a live probe.
"""
from __future__ import annotations

import os
from typing import Any

# Context window used for token budgeting; a conservative value is fine.
_DEFAULT_CTX = 128_000

# Model ids as each provider names them. All are open-weights.
# Each provider reads its own key variable, falling back to OPENAI_API_KEY. The
# variable name OPENAI_API_KEY denotes the *protocol* these services speak, not
# the vendor -- an OpenAI key will not authenticate against Together, so keeping
# them in separate variables avoids clobbering a working OpenAI setup.
PROVIDERS: dict[str, dict[str, Any]] = {
    "together": {
        "key_var": "TOGETHER_API_KEY",
        "base_url": "https://api.together.xyz/v1",
        # Verified serverless *and* returning logprobs, 2026-07-30. Models not
        # on the serverless tier answer 400 ("non-serverless model ... create a
        # dedicated endpoint"), so the list is deliberately short rather than
        # aspirational. Re-probe before trusting it: availability changes.
        "models": [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "openai/gpt-oss-120b",
        ],
    },
    "fireworks": {
        "key_var": "FIREWORKS_API_KEY",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "models": [
            "accounts/fireworks/models/llama-v3p3-70b-instruct",
            "accounts/fireworks/models/mixtral-8x22b-instruct",
            "accounts/fireworks/models/qwen2p5-72b-instruct",
        ],
    },
    "deepinfra": {
        "key_var": "DEEPINFRA_API_KEY",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "models": [
            "meta-llama/Llama-3.3-70B-Instruct",
            "mistralai/Mistral-Small-24B-Instruct-2501",
            "Qwen/Qwen2.5-72B-Instruct",
        ],
    },
    "groq": {
        "key_var": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "models": [
            "llama-3.3-70b-versatile",
            "qwen-2.5-32b",
        ],
    },
    # Self-hosted: whatever you loaded into the server.
    "vllm": {
        "key_var": "VLLM_API_KEY",
        "base_url": os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        "models": [],
    },
    "ollama": {
        "key_var": "OLLAMA_API_KEY",
        "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "models": [],
    },
}


# Local servers usually accept any token, so a placeholder is fine there. Remote
# providers must not fall back to OPENAI_API_KEY: an OpenAI key sent to Together
# returns a bare 401, which reads as "the provider is broken" rather than "you
# used the wrong credential".
_LOCAL_PROVIDERS = frozenset({"vllm", "ollama"})


def resolve_key(provider: str) -> str:
    """The provider's own key, or "" if it is not set.

    Returns "" rather than substituting another vendor's key, so callers can fail
    with a message naming the variable to set.
    """
    var = PROVIDERS[provider].get("key_var", "OPENAI_API_KEY")
    key = os.environ.get(var, "")
    if not key and provider in _LOCAL_PROVIDERS:
        return os.environ.get("OPENAI_API_KEY") or "local-no-auth"
    return key


def register(provider: str, extra_models: list[str] | None = None) -> list[str]:
    """Registers a provider's models with ReActXen and returns their prefixed ids.

    Mutates ``model_inference.modelset`` and ``phmforge_runner.MODEL_NAME_TO_ID``
    the same way ``phmforge_runner`` registers its own extra models, so the rest
    of the stack needs no changes.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; have {sorted(PROVIDERS)}")

    spec = PROVIDERS[provider]
    models = list(spec["models"]) + list(extra_models or [])
    if not models:
        raise ValueError(
            f"provider {provider!r} has no models configured; pass extra_models "
            "with the ids your server exposes"
        )

    # direct_openai_llm builds its client from OPENAI_API_KEY / OPENAI_BASE_URL,
    # so point both at the chosen provider for this process only.
    key = resolve_key(provider)
    if not key:
        raise ValueError(
            f"no credentials for {provider!r}: set "
            f"{spec.get('key_var', 'OPENAI_API_KEY')} in .env"
        )
    os.environ["OPENAI_API_KEY"] = key
    os.environ["OPENAI_BASE_URL"] = spec["base_url"]

    import phmforge_runner as runner  # type: ignore
    from reactxen.utils import model_inference as mi  # type: ignore

    prefixed = []
    for name in models:
        pid = f"openai/{name}"
        prefixed.append(pid)
        if pid not in mi.modelset:
            mi.modelset.append(pid)
        runner._extra_ctx.setdefault(pid, _DEFAULT_CTX)
        runner.MODEL_NAME_TO_ID.setdefault(pid, mi.modelset.index(pid))
    return prefixed


def probe_logprobs(provider: str, extra_models: list[str] | None = None) -> int:
    """One tiny call per model; prints whether logprobs are actually returned.

    Real logprobs are strongly left-skewed with substantial mass just below zero.
    A provider that returns a flat or symmetric spread is worth a second look.
    """
    import numpy as np
    from openai import OpenAI

    spec = PROVIDERS[provider]
    models = list(spec["models"]) + list(extra_models or [])
    key = resolve_key(provider)
    if not key:
        print(f"no credentials: set {spec.get('key_var','OPENAI_API_KEY')} in .env")
        return 0
    client = OpenAI(api_key=key, base_url=spec["base_url"])

    n_ok = 0
    print(f"probing {provider} at {client.base_url}")
    for name in models:
        try:
            resp = client.chat.completions.create(
                model=name,
                messages=[{"role": "user", "content": "Name three metals."}],
                max_tokens=40,
                temperature=0,
                logprobs=True,
            )
            # Providers disagree on the shape: OpenAI fills logprobs.content,
            # Together fills a flat logprobs.token_logprobs and leaves .content
            # None. Reading one shape only reports "no logprobs" on the other.
            lp = resp.choices[0].logprobs
            vals: list[float] = []
            if lp is not None:
                if getattr(lp, "content", None):
                    vals = [t.logprob for t in lp.content]
                else:
                    vals = [x for x in (getattr(lp, "token_logprobs", None) or [])
                            if x is not None]
            if vals:
                arr = np.asarray(vals, dtype=float)
                skew = (
                    float(np.mean(((arr - arr.mean()) / arr.std()) ** 3))
                    if arr.std() > 0
                    else 0.0
                )
                print(f"  [OK]   {name:58s} n={len(vals):3d} "
                      f"mean={arr.mean():+.3f} skew={skew:+.2f}")
                n_ok += 1
            else:
                print(f"  [NO]   {name:58s} responded without logprobs")
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERR]  {name:58s} {type(exc).__name__}: {str(exc)[:70]}")

    print(f"\n{n_ok}/{len(models)} models return logprobs")
    return n_ok


def main() -> int:
    import argparse
    import sys
    from pathlib import Path

    ap = argparse.ArgumentParser(description="probe a provider for logprob support")
    ap.add_argument("provider", choices=sorted(PROVIDERS))
    ap.add_argument("--models", nargs="*", default=None,
                    help="extra model ids (required for vllm/ollama)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(root))
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except ImportError:
        pass

    return 0 if probe_logprobs(args.provider, args.models) else 1


if __name__ == "__main__":
    raise SystemExit(main())
