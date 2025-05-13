# modules/feedback.py
import os
import json
import time
from .config import (
    ENABLE_PRE_SCAN_FEEDBACK,
    FEEDBACK_LOOKBACK_TRADES,
    FEEDBACK_CACHE_FILE,
)
from .simulator import Simulator
from .openai_client import OpenAIClient

class FeedbackEngine:
    def __init__(self):
        self.sim = Simulator()
        self.openai = OpenAIClient()

    def _load_last(self) -> dict:
        try:
            with open(FEEDBACK_CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_last(self, payload: dict):
        os.makedirs(os.path.dirname(FEEDBACK_CACHE_FILE), exist_ok=True)
        with open(FEEDBACK_CACHE_FILE, 'w') as f:
            json.dump(payload, f)

    async def pre_scan_advice(self, summary: dict, recent_trades: list[dict]) -> dict:
        if not ENABLE_PRE_SCAN_FEEDBACK:
            return {"summary": "", "tweaks": {}}

        last = self._load_last().get("advice", {})
        payload = {
            "summary": summary,
            "recent_trades": recent_trades,
            "last_advice": last,
            "request": (
                "Provide a concise performance analysis "
                "and suggest parameter tweaks (entry/exit thresholds, risk limits) "
                "in JSON with 'summary' and 'tweaks'."
            )
        }

        messages = [
            {"role":"system","content":"You are an expert crypto quant."},
            {"role":"user","content":json.dumps(payload)},
        ]
        resp = await self.openai.chat(messages, deep=False, max_tokens=300)
        text = resp.choices[0].message.content.strip()
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            result = {"summary": text, "tweaks": {}}

        tweaks = result.get("tweaks", {})
        if tweaks != last:
            self._save_last({"timestamp": time.time(), "advice": tweaks})
        return result

    async def suggest_thresholds(self, summary: dict) -> dict:
        """
        Existing threshold tuning logic, now via OpenAIClient.
        """
        # Example minimal implementation; you can expand similar to pre_scan_advice.
        payload = {
            "summary": summary,
            "request": "Suggest entry and exit thresholds in JSON {'entry':..., 'exit':...}.",
        }
        messages = [
            {"role":"system","content":"You are a threshold optimization assistant."},
            {"role":"user","content":json.dumps(payload)},
        ]
        resp = await self.openai.chat(messages, deep=False, max_tokens=50)
        try:
            return json.loads(resp.choices[0].message.content)
        except:
            return {"entry": 0.5, "exit": 0.3}
