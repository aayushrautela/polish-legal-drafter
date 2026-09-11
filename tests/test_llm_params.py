"""Offline tests for the sampling-param compatibility layer (no network)."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from legal_drafter import llm_params as LP


class FakeCompletions:
    """Records create() calls; raises script[i] on the i-th call."""

    def __init__(self, script):
        self.script = script
        self.calls = []

    def create(self, **kwargs):
        i = len(self.calls)
        self.calls.append(kwargs)
        if i < len(self.script) and self.script[i] is not None:
            raise self.script[i]
        return "ok"


class FakeClient:
    def __init__(self, script=()):
        self.chat = SimpleNamespace(completions=FakeCompletions(list(script)))


class SamplingRejection(Exception):
    status_code = 400

    def __init__(self, param="temperature"):
        super().__init__(f"`{param}` is deprecated for this model.")


class OtherBadRequest(Exception):
    status_code = 400

    def __init__(self):
        super().__init__("model id not found")


def setUpModule():
    pass


class LLMParamsTest(unittest.TestCase):
    def setUp(self):
        LP._STRIP_CACHE.clear()
        LP._WARNED.clear()
        LP.DECISIONS.clear()
        for var in ("LLM_SAMPLING", "TEACHER_SAMPLING", "CHECKER_SAMPLING",
                    "JUDGE_SAMPLING"):
            os.environ.pop(var, None)

    def test_mode_precedence(self):
        self.assertEqual(LP.sampling_mode("teacher"), "on")
        os.environ["LLM_SAMPLING"] = "off"
        self.assertEqual(LP.sampling_mode("teacher"), "off")
        os.environ["TEACHER_SAMPLING"] = "on"
        self.assertEqual(LP.sampling_mode("teacher"), "on")
        self.assertEqual(LP.sampling_mode("checker"), "off")
        os.environ["TEACHER_SAMPLING"] = "bogus"  # unknown falls through
        self.assertEqual(LP.sampling_mode("teacher"), "off")

    def test_default_values_elided(self):
        sent = LP.resolve_sampling_params("teacher", "m", temperature=1.0,
                                          top_p=1.0)
        self.assertEqual(sent, {})

    def test_non_default_kept_in_on_mode(self):
        sent = LP.resolve_sampling_params("teacher", "m", temperature=0.2)
        self.assertEqual(sent, {"temperature": 0.2})

    def test_off_strips_everything(self):
        os.environ["TEACHER_SAMPLING"] = "off"
        sent = LP.resolve_sampling_params("teacher", "m", temperature=0.8,
                                          top_p=0.9)
        self.assertEqual(sent, {})
        decision = LP.DECISIONS[("teacher", "m")]
        self.assertEqual(decision["requested"],
                         {"temperature": 0.8, "top_p": 0.9})
        self.assertEqual(decision["sent"], {})

    def test_chat_create_passthrough_and_extra_body(self):
        client = FakeClient()
        resp = LP.chat_create(client, role="teacher", model="m",
                              messages=[{"role": "user", "content": "x"}],
                              temperature=0.2, extra_body={"reasoning_effort": "high"},
                              stream=True)
        self.assertEqual(resp, "ok")
        sent = client.chat.completions.calls[0]
        self.assertEqual(sent["temperature"], 0.2)
        self.assertEqual(sent["extra_body"], {"reasoning_effort": "high"})
        self.assertTrue(sent["stream"])

    def test_chat_create_detection_retry_and_cache(self):
        client = FakeClient(script=[SamplingRejection()])
        resp = LP.chat_create(client, role="teacher", model="alias-x",
                              messages=[], temperature=0.2, stream=True)
        self.assertEqual(resp, "ok")
        self.assertEqual(len(client.chat.completions.calls), 2)
        self.assertIn("temperature", client.chat.completions.calls[0])
        self.assertNotIn("temperature", client.chat.completions.calls[1])
        # Cached: a fresh client never sends temperature again, no retry.
        client2 = FakeClient()
        LP.chat_create(client2, role="teacher", model="alias-x",
                       messages=[], temperature=0.2)
        self.assertEqual(len(client2.chat.completions.calls), 1)
        self.assertNotIn("temperature", client2.chat.completions.calls[0])

    def test_unrelated_400_propagates(self):
        client = FakeClient(script=[OtherBadRequest()])
        with self.assertRaises(OtherBadRequest):
            LP.chat_create(client, role="teacher", model="m",
                           messages=[], temperature=0.2)
        self.assertNotIn("m", LP._STRIP_CACHE)

    def test_rejection_with_no_params_propagates(self):
        client = FakeClient(script=[SamplingRejection()])
        with self.assertRaises(SamplingRejection):
            LP.chat_create(client, role="judge", model="m",
                           messages=[], temperature=1.0)  # elided -> nothing sent
        self.assertEqual(len(client.chat.completions.calls), 1)


if __name__ == "__main__":
    unittest.main()
