import os, requests


def available():
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY"))


def complete(prompt):
    """Returns (text, tokens) or (None, 0) when no key / failure."""
    try:
        if os.getenv("GEMINI_API_KEY"):
            m = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent", params={"key": os.environ["GEMINI_API_KEY"]},
                              json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0}}, timeout=40).json()
            return r["candidates"][0]["content"]["parts"][0]["text"], r.get("usageMetadata", {}).get("totalTokenCount", 0)
        if os.getenv("OPENAI_API_KEY"):
            r = requests.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
                              json={"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "temperature": 0, "messages": [{"role": "user", "content": prompt}]}, timeout=40).json()
            return r["choices"][0]["message"]["content"], r.get("usage", {}).get("total_tokens", 0)
    except Exception:
        pass
    return None, 0
