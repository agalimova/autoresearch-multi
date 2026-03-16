"""
LLM-powered slot proposer.

Reads prior results and existing slot files, asks an LLM to propose
a new variant. Falls back to template generation if no API key.

Usage:
    proposer = SlotProposer(slot_dir, results)
    new_code = proposer.propose("build_model")
    # writes build_model/llm_v1.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional


def _call_llm(prompt: str) -> str:
    """Call any available LLM. Tries Anthropic, OpenAI, Gemini in order."""
    # Try Anthropic (Claude)
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception:
            pass

    # Try OpenAI (GPT)
    if os.environ.get("OPENAI_API_KEY"):
        try:
            import openai
            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""
        except Exception:
            pass

    # Try Google Gemini
    if os.environ.get("GEMINI_API_KEY"):
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            model = genai.GenerativeModel("gemini-2.0-flash")
            resp = model.generate_content(prompt)
            return resp.text or ""
        except Exception:
            pass

    # Try OpenRouter (routes to any model)
    if os.environ.get("OPENROUTER_API_KEY"):
        try:
            import httpx
            resp = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                json={"model": "anthropic/claude-sonnet-4-20250514", "max_tokens": 1024,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=30,
            )
            return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            pass

    # Try Together AI
    if os.environ.get("TOGETHER_API_KEY"):
        try:
            import httpx
            resp = httpx.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}"},
                json={"model": "meta-llama/Llama-3-70b-chat-hf", "max_tokens": 1024,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=30,
            )
            return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            pass

    # Try Deepseek
    if os.environ.get("DEEPSEEK_API_KEY"):
        try:
            import httpx
            resp = httpx.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"},
                json={"model": "deepseek-coder", "max_tokens": 1024,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=30,
            )
            return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            pass

    # Try Ollama (local, no key needed)
    try:
        import httpx
        resp = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "")
    except Exception:
        pass

    return ""


_PROVIDERS = [
    ("ANTHROPIC_API_KEY", "claude"),
    ("OPENAI_API_KEY", "gpt-4o"),
    ("GEMINI_API_KEY", "gemini"),
    ("OPENROUTER_API_KEY", "openrouter"),
    ("TOGETHER_API_KEY", "together"),
    ("DEEPSEEK_API_KEY", "deepseek"),
]


def has_llm() -> bool:
    """Check if any LLM is available (API key or local Ollama)."""
    for key, _ in _PROVIDERS:
        if os.environ.get(key):
            return True
    # Check Ollama
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            return True
    except Exception:
        pass
    return False


def llm_name() -> str:
    """Return which LLM will be used."""
    for key, name in _PROVIDERS:
        if os.environ.get(key):
            return name
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            return "ollama (local)"
    except Exception:
        pass
    return "none"


def _build_prompt(
    slot_name: str,
    slot_dir: Path,
    results: list[dict],
    prior_findings: list[dict] | None = None,
) -> str:
    """Build a prompt for the LLM to propose a new slot variant."""
    # Show existing variants
    existing = []
    variant_dir = slot_dir / slot_name
    if variant_dir.exists():
        for f in sorted(variant_dir.glob("*.py"))[:5]:
            existing.append(f"--- {f.name} ---\n{f.read_text()[:300]}")

    # Show top results
    top = sorted(results, key=lambda r: r.get("val_acc", 0), reverse=True)[:5]
    results_str = ""
    for r in top:
        combo = r.get("combo", {})
        acc = r.get("val_acc", 0)
        results_str += f"  {acc:.4f} | {combo.get(slot_name, '?')}\n"

    # Show worst results (dead ends)
    worst = sorted(results, key=lambda r: r.get("val_acc", 0))[:3]
    dead_ends = ""
    for r in worst:
        combo = r.get("combo", {})
        acc = r.get("val_acc", 0)
        if acc > 0:
            dead_ends += f"  {acc:.4f} | {combo.get(slot_name, '?')} (avoid)\n"

    # Prior round findings (cross-pollination)
    findings_str = ""
    if prior_findings:
        for f in prior_findings[-3:]:
            findings_str += (
                f"\n  Round {f.get('round')}: {f.get('insight', '')}"
                f"\n  Proposed: {f.get('proposed', [])}"
                f"\n  Winner: {f.get('winner', {})}\n"
            )

    return f"""You are an ML researcher. Write a Python function for the "{slot_name}" slot.

The function must be named `{slot_name}` and follow the same signature as existing variants.

EXISTING VARIANTS:
{chr(10).join(existing)}

TOP RESULTS SO FAR:
{results_str}

DEAD ENDS (these didn't work well):
{dead_ends}

PRIOR ROUNDS (what was tried before and what happened):
{findings_str if findings_str else "  (first round)"}

Based on ALL of the above, propose a NEW variant that might improve accuracy.
Learn from prior rounds — don't repeat what failed. Build on what worked.

Return ONLY the Python code. No explanation, no markdown fences. Just the code."""


class SlotProposer:
    """Proposes new slot variants using an LLM."""

    def __init__(self, slot_dir: Path, results: list[dict]):
        self.slot_dir = slot_dir
        self.results = results
        self.prior_findings: list[dict] = []
        self._count = 0

    def propose(self, slot_name: str) -> Optional[Path]:
        """Propose a new variant for a slot. Returns path or None."""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None

        prompt = _build_prompt(slot_name, self.slot_dir, self.results, self.prior_findings)
        response = _call_llm(prompt)

        if not response or "def " not in response:
            return None

        # Clean response
        code = _clean_code(response)
        if not code:
            return None

        # Write to slot directory
        self._count += 1
        name = f"llm_v{self._count}"
        out_dir = self.slot_dir / slot_name
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}.py"
        path.write_text(code)
        return path

    def propose_round(self, n: int = 3) -> list[Path]:
        """Propose variants for all variable slots."""
        paths = []
        for slot_name in _variable_slots(self.slot_dir):
            for _ in range(n):
                path = self.propose(slot_name)
                if path:
                    paths.append(path)
                    # Update results context for next proposal
                    print(f"  LLM proposed: {path.parent.name}/{path.name}")
        return paths


def _variable_slots(slot_dir: Path) -> list[str]:
    """Slots with more than 1 variant (worth proposing new ones for)."""
    slots = []
    for d in sorted(slot_dir.iterdir()):
        if not d.is_dir():
            continue
        n = len(list(d.glob("*.py")))
        if n >= 1:
            slots.append(d.name)
    return slots


def _clean_code(response: str) -> str:
    """Extract clean Python code from LLM response."""
    # Remove markdown fences if present
    code = re.sub(r"```python\s*\n?", "", response)
    code = re.sub(r"```\s*$", "", code)
    code = code.strip()

    # Basic validation
    if "def " not in code:
        return ""
    if "import " not in code and "from " not in code:
        return ""

    return code
