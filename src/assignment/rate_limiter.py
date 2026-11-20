"""
Assignment 11 — Rate Limiter starter (TODO).

Sliding-window, per-user rate limiting. Blocks abuse that other
guardrail layers do not address (flooding / cost attacks).
"""
from __future__ import annotations

from collections import defaultdict, deque
import time

from google.adk.plugins import base_plugin
from google.genai import types


RATE_LIMIT_MESSAGE_PREFIX = "[rate_limiter]"


class RateLimitPlugin(base_plugin.BasePlugin):
    """Block users who exceed max_requests within window_seconds."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        super().__init__(name="rate_limiter")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_windows: dict[str, deque] = defaultdict(deque)
        self.blocked_count = 0
        self.total_count = 0

    def _block_response(self, message: str) -> types.Content:
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        )

    async def on_user_message_callback(self, *, invocation_context, user_message):
        """Return Content to block, or None to allow."""
        self.total_count += 1
        user_id = getattr(invocation_context, "user_id", None) or "anonymous"
        now = time.time()
        window = self.user_windows[user_id]

        # 1. Trượt cửa sổ: bỏ các timestamp đã "hết hạn" (cũ hơn now - window)
        #    deque giữ timestamp theo thứ tự thời gian → cái cũ nhất luôn ở bên trái
        cutoff = now - self.window_seconds
        while window and window[0] <= cutoff:
            window.popleft()

        # 2. Cửa sổ đã đầy → chặn, KHÔNG ghi timestamp (request bị chặn không tính quota)
        if len(window) >= self.max_requests:
            wait = self.window_seconds - (now - window[0])
            self.blocked_count += 1
            return self._block_response(
                f"{RATE_LIMIT_MESSAGE_PREFIX} Rate limit exceeded "
                f"({self.max_requests} requests / {self.window_seconds}s). "
                f"Try again in {wait:.0f}s."
            )

        # 3. Còn quota → ghi lại thời điểm và cho qua
        window.append(now)
        return None
