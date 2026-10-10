import os
import time


class RateLimiter:
    """Spaces calls evenly so a per-minute API quota is never exceeded."""

    def __init__(self, calls_per_minute: float = 0.0):
        self.min_interval = 60.0 / calls_per_minute if calls_per_minute > 0 else 0.0
        self._last_call = 0.0

    @classmethod
    def for_gemini(cls) -> "RateLimiter":
        return cls(float(os.getenv("GEMINI_RPM", "10")))

    def wait(self):
        remaining = self.min_interval - (time.time() - self._last_call)
        if remaining > 0:
            time.sleep(remaining)
        self._last_call = time.time()
