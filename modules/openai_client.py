# modules/openai_client.py
import os
import time
import asyncio
import openai
from .config import (
    OPENAI_API_KEY,
    OPENAI_MODEL_CHAT,
    OPENAI_MODEL_DEEP,
    OPENAI_TIMEOUT_SEC,
    OPENAI_RETRY_ATTEMPTS,
    OPENAI_RETRY_BACKOFF,
)

class OpenAIClient:
    def __init__(self):
        openai.api_key = OPENAI_API_KEY
        self.model_chat = OPENAI_MODEL_CHAT
        self.model_deep = OPENAI_MODEL_DEEP

    async def _call_in_thread(self, fn, *args, **kwargs):
        """
        Run a blocking OpenAI API call in a thread.
        """
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def chat(self, messages: list[dict], deep: bool = False, max_tokens: int = 200):
        """
        Send a chat completion request with retries, timeout, and thread offloading.
        """
        model = self.model_deep if deep else self.model_chat
        fn = openai.chat.completions.create  # blocking call

        delay = OPENAI_RETRY_BACKOFF
        for attempt in range(1, OPENAI_RETRY_ATTEMPTS + 1):
            try:
                # Offload to thread and enforce timeout
                resp = await asyncio.wait_for(
                    self._call_in_thread(
                        fn,
                        model=model,
                        messages=messages,
                        temperature=0.5,
                        max_tokens=max_tokens,
                    ),
                    timeout=OPENAI_TIMEOUT_SEC
                )
                return resp
            except Exception as e:
                if attempt == OPENAI_RETRY_ATTEMPTS:
                    raise
                await asyncio.sleep(delay)
                delay *= 2  # exponential backoff
