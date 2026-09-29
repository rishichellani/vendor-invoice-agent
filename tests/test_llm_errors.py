"""The user-facing message when both AI engines fail (no network)."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent  # noqa: E402


class RateLimitError(Exception):
    status_code = 429


class LLMUnavailable(unittest.TestCase):
    def call(self, gemini_exc, groq_exc):
        with mock.patch.object(agent, "_call_gemini", side_effect=gemini_exc), \
                mock.patch.object(agent, "_call_groq", side_effect=groq_exc):
            with self.assertRaises(agent.LLMUnavailableError) as ctx:
                agent.call_llm("prompt")
        return ctx.exception

    def test_quota_exhaustion_is_explained_in_plain_language(self):
        exc = self.call(RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded"), RateLimitError("Request too large for model qwen"))
        self.assertTrue(exc.rate_limited)
        self.assertIn("out of quota or rate-limited", str(exc))
        self.assertNotIn("qwen", str(exc))
        self.assertNotIn("org_", str(exc))

    def test_other_failures_are_not_blamed_on_quota(self):
        exc = self.call(ValueError("bad json"), TimeoutError("timed out"))
        self.assertFalse(exc.rate_limited)
        self.assertIn("could not read this invoice", str(exc))

    def test_groq_success_still_works_when_gemini_fails(self):
        with mock.patch.object(agent, "_call_gemini", side_effect=RuntimeError("down")), \
                mock.patch.object(agent, "_call_groq", return_value={"ok": 1}):
            self.assertEqual(agent.call_llm("p"), ({"ok": 1}, "groq"))


if __name__ == "__main__":
    unittest.main()
