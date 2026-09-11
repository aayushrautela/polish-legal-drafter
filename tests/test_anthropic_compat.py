"""Unit tests for the Anthropic compat layer (src/legal_drafter/compat.py).

All tests run against httpx.MockTransport -- no network, no Anthropic access.
"""

import json
import unittest

import httpx

from legal_drafter import compat, llm_params


def _anthropic_response(content, stop_reason="end_turn", usage=None):
    return {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": content,
        "stop_reason": stop_reason,
        "usage": usage or {"input_tokens": 10, "output_tokens": 20},
    }


class AnthropicCompatTest(unittest.TestCase):

    def _client(self, handler, **kw):
        return compat.AnthropicCompatClient(
            base_url="https://fake.test", api_key="k",
            _transport=httpx.MockTransport(handler), **kw)

    # ---------------- request translation ----------------

    def test_system_extraction_and_user_translation(self):
        seen = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(req.content)
            return httpx.Response(200, json=_anthropic_response(
                [{"type": "text", "text": "ok"}]))

        c = self._client(handler)
        c.create(model="claude-sonnet-5", max_tokens=100, stream=False,
                 messages=[
                     {"role": "system", "content": "sys one"},
                     {"role": "system", "content": "sys two"},
                     {"role": "user", "content": "hello"},
                 ])
        body = seen["body"]
        self.assertEqual(body["model"], "claude-sonnet-5")
        self.assertEqual(body["max_tokens"], 100)
        self.assertEqual(body["system"],
                         [{"type": "text", "text": "sys one"},
                          {"type": "text", "text": "sys two"}])
        self.assertEqual(len(body["messages"]), 1)
        self.assertEqual(body["messages"][0]["role"], "user")
        self.assertEqual(body["messages"][0]["content"],
                         [{"type": "text", "text": "hello"}])

    def test_tool_history_translation(self):
        """assistant tool_calls -> tool_use; tool msg -> tool_result in a
        user message; unsigned reasoning_content is NOT sent back."""
        seen = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(req.content)
            return httpx.Response(200, json=_anthropic_response(
                [{"type": "text", "text": "done"}]))

        c = self._client(handler)
        c.create(model="m", stream=False, messages=[
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "",
             "reasoning_content": "unsiged thinking text",
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "keyword_search",
                                          "arguments": "{\"query\": \"najem\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "name": "keyword_search",
             "content": "[{\"chunk_id\": \"a\"}]"},
        ])
        msgs = seen["body"]["messages"]
        self.assertEqual(len(msgs), 3)  # user, assistant, merged user(tool_result)
        asst = msgs[1]
        self.assertEqual(asst["role"], "assistant")
        self.assertEqual(len(asst["content"]), 1)  # tool_use only; no thinking
        self.assertEqual(asst["content"][0]["type"], "tool_use")
        self.assertEqual(asst["content"][0]["name"], "keyword_search")
        self.assertEqual(asst["content"][0]["input"], {"query": "najem"})
        tr = msgs[2]
        self.assertEqual(tr["role"], "user")
        self.assertEqual(tr["content"][0]["type"], "tool_result")
        self.assertEqual(tr["content"][0]["tool_use_id"], "call_1")

    def test_signed_thinking_blocks_roundtrip(self):
        seen = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(req.content)
            return httpx.Response(200, json=_anthropic_response(
                [{"type": "text", "text": "ok"}]))

        c = self._client(handler)
        signed = [{"type": "thinking", "thinking": "plan",
                   "signature": "SIG123"}]
        unsigned = [{"type": "thinking", "thinking": "plan", "signature": ""}]
        c.create(model="m", stream=False, messages=[
            {"role": "assistant", "content": None,
             "thinking_blocks": signed + unsigned,
             "tool_calls": [{"id": "t1", "type": "function",
                             "function": {"name": "f", "arguments": "{}"}}]},
        ])
        blocks = seen["body"]["messages"][0]["content"]
        self.assertEqual(len(blocks), 2)  # signed thinking + tool_use
        self.assertEqual(blocks[0]["type"], "thinking")
        self.assertEqual(blocks[0]["signature"], "SIG123")
        self.assertEqual(blocks[1]["type"], "tool_use")

    def test_temperature_locked_while_thinking(self):
        seen = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(req.content)
            return httpx.Response(200, json=_anthropic_response(
                [{"type": "text", "text": "ok"}]))

        c = self._client(handler)
        c.create(model="m", stream=False, temperature=0.2, messages=[
            {"role": "user", "content": "q"}])
        self.assertNotIn("temperature", seen["body"])
        self.assertIn("temperature=0.2", "; ".join(c.last_notes))

        c2 = self._client(handler, thinking_mode="off")
        c2.create(model="m", stream=False, temperature=0.2, messages=[
            {"role": "user", "content": "q"}])
        self.assertEqual(seen["body"]["temperature"], 0.2)

    def test_response_format_to_output_config(self):
        seen = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(req.content)
            return httpx.Response(200, json=_anthropic_response(
                [{"type": "text", "text": "{}"}]))

        c = self._client(handler)
        schema = {"type": "object", "properties": {"summary": {"type": "string"},
                                                   "contract": {"type": "string"}}}
        c.create(model="m", stream=False, messages=[
            {"role": "user", "content": "draft"}],
            response_format={"type": "json_schema", "json_schema":
                             {"name": "Draft", "strict": True, "schema": schema}})
        self.assertEqual(seen["body"]["output_config"],
                         {"format": {"type": "json_schema", "schema": schema}})

        c.create(model="m", stream=False, messages=[
            {"role": "user", "content": "draft"}],
            response_format={"type": "json_object"})
        self.assertEqual(seen["body"]["output_config"],
                         {"format": {"type": "json_schema",
                                     "schema": {"type": "object"}}})

        c.create(model="m", stream=False, messages=[
            {"role": "user", "content": "draft"}],
            response_format={"type": "text"})
        self.assertNotIn("output_config", seen["body"])

    def test_tool_choice_mapping(self):
        self.assertEqual(compat.translate_tool_choice("auto"), {"type": "auto"})
        self.assertEqual(compat.translate_tool_choice("none"), {"type": "none"})
        self.assertEqual(compat.translate_tool_choice("required"), {"type": "any"})
        self.assertEqual(
            compat.translate_tool_choice(
                {"type": "function", "function": {"name": "search"}}),
            {"type": "tool", "name": "search"})

    def test_tools_translation(self):
        out = compat.translate_tools([
            {"type": "function", "function": {
                "name": "keyword_search", "description": "search",
                "parameters": {"type": "object", "properties": {}}}},
        ])
        self.assertEqual(out, [{"name": "keyword_search",
                                "description": "search",
                                "input_schema": {"type": "object",
                                                 "properties": {}}}])

    def test_hygiene_empty_and_trailing_whitespace(self):
        seen = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(req.content)
            return httpx.Response(200, json=_anthropic_response(
                [{"type": "text", "text": "ok"}]))

        c = self._client(handler)
        c.create(model="m", stream=False, messages=[
            {"role": "user", "content": ""},
            {"role": "user", "content": "real question  "},
            {"role": "assistant", "content": "trailing spaces   "},
        ])
        msgs = seen["body"]["messages"]
        # consecutive users merged into one turn; empty contribution dropped,
        # not placed as a separate "*" block; assistant text rstripped
        self.assertEqual(msgs[0]["content"], [{"type": "text", "text": "real question  "}])
        self.assertEqual(msgs[-1]["content"][0]["text"], "trailing spaces")

    def test_lone_empty_user_gets_placeholder(self):
        seen = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(req.content)
            return httpx.Response(200, json=_anthropic_response(
                [{"type": "text", "text": "ok"}]))

        c = self._client(handler)
        c.create(model="m", stream=False, messages=[
            {"role": "user", "content": ""}])
        self.assertEqual(seen["body"]["messages"][0]["content"],
                         [{"type": "text", "text": "."}])

    # ---------------- response translation ----------------

    def test_response_tool_calls_thinking_usage(self):
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_anthropic_response(
                [{"type": "thinking", "thinking": "plan the search",
                  "signature": "SIG"},
                 {"type": "tool_use", "id": "toolu_1", "name": "semantic_search",
                  "input": {"query": "kaucja"}}],
                stop_reason="tool_use",
                usage={"input_tokens": 100, "output_tokens": 50,
                       "cache_read_input_tokens": 7,
                       "cache_creation_input_tokens": 3}))

        c = self._client(handler)
        resp = c.create(model="m", stream=False, messages=[
            {"role": "user", "content": "q"}])
        msg = resp.choices[0].message
        self.assertEqual(resp.choices[0].finish_reason, "tool_calls")
        self.assertEqual(msg.reasoning_content, "plan the search")
        self.assertEqual(len(msg.tool_calls), 1)
        tc = msg.tool_calls[0]
        self.assertEqual(tc.id, "toolu_1")
        self.assertEqual(tc.function.name, "semantic_search")
        self.assertEqual(json.loads(tc.function.arguments), {"query": "kaucja"})
        self.assertEqual(resp.usage.prompt_tokens, 110)   # 100 + 7 + 3
        self.assertEqual(resp.usage.completion_tokens, 50)

    def test_stop_reason_mapping(self):
        cases = {"end_turn": "stop", "max_tokens": "length",
                 "tool_use": "tool_calls", "refusal": "content_filter",
                 "mystery": "stop"}
        for native, openai in cases.items():
            def handler(req: httpx.Request, _n=native) -> httpx.Response:
                return httpx.Response(200, json=_anthropic_response(
                    [{"type": "text", "text": "x"}], stop_reason=_n))
            c = self._client(handler)
            resp = c.create(model="m", stream=False, messages=[
                {"role": "user", "content": "q"}])
            self.assertEqual(resp.choices[0].finish_reason, openai)

    # ---------------- streaming ----------------

    def test_streaming_single_chunk(self):
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_anthropic_response(
                [{"type": "thinking", "thinking": "think", "signature": "S"},
                 {"type": "tool_use", "id": "tu", "name": "f",
                  "input": {"a": 1}},
                 {"type": "text", "text": "answer"}],
                stop_reason="tool_use",
                usage={"input_tokens": 5, "output_tokens": 6}))

        c = self._client(handler)
        it = c.create(model="m", stream=True, tools=[], messages=[
            {"role": "user", "content": "q"}])
        chunks = list(it)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        # consume exactly like run_beta_doc._stream_turn does
        usage = getattr(chunk, "usage", None)
        self.assertEqual(usage.prompt_tokens, 5)
        d = chunk.choices[0].delta
        parts, reasoning_parts, tcs = [], [], []
        if getattr(d, "content", None):
            parts.append(d.content)
        r = getattr(d, "reasoning", None) or getattr(d, "reasoning_content", None)
        if r:
            reasoning_parts.append(r)
        for tc in getattr(d, "tool_calls", None) or []:
            i = getattr(tc, "index", 0) or 0
            tcs.append((i, tc.function.name, tc.function.arguments))
        self.assertEqual("".join(parts), "answer")
        self.assertEqual("".join(reasoning_parts), "think")
        self.assertEqual(tcs, [(1, "f", json.dumps({"a": 1}))])

    # ---------------- errors ----------------

    def test_error_maps_to_compat_api_error(self):
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {
                "type": "invalid_request_error",
                "message": "temperature is deprecated for this model"}})

        c = self._client(handler)
        with self.assertRaises(compat.CompatAPIError) as ctx:
            c.create(model="m", stream=False, temperature=0.3, messages=[
                {"role": "user", "content": "q"}])
        self.assertEqual(ctx.exception.status_code, 400)
        # llm_params 400-detection recognizes it -> retry path works
        self.assertTrue(llm_params._is_sampling_rejection(ctx.exception))

    def test_llm_params_retry_with_compat_client(self):
        """End-to-end: chat_create strips temperature after the compat 400."""
        calls = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            body = json.loads(req.content)
            if "temperature" in body:
                return httpx.Response(400, json={"error": {
                    "type": "invalid_request_error",
                    "message": "temperature is deprecated for this model"}})
            return httpx.Response(200, json=_anthropic_response(
                [{"type": "text", "text": "ok"}]))

        c = self._client(handler, thinking_mode="off")
        resp = llm_params.chat_create(
            c, role="teacher", model="claude-sonnet-5", stream=False,
            temperature=0.2,
            messages=[{"role": "user", "content": "q"}])
        self.assertEqual(calls["n"], 2)  # 400 with temperature, retry without
        self.assertEqual(resp.choices[0].message.content, "ok")
        llm_params._STRIP_CACHE.discard("claude-sonnet-5")

    # ---------------- factory ----------------

    def test_make_chat_client_routing(self):
        c = compat.make_chat_client(
            "teacher", "http://x", "k", env={"TEACHER_COMPAT": "anthropic"})
        self.assertIsInstance(c, compat.AnthropicCompatClient)
        c = compat.make_chat_client(
            "teacher", "http://x", "k", env={"LLM_COMPAT": "claude"})
        self.assertIsInstance(c, compat.AnthropicCompatClient)
        c = compat.make_chat_client("teacher", "http://x", "k", env={})
        self.assertEqual(type(c).__name__, "OpenAI")
        self.assertEqual(compat.compat_mode("teacher", {"TEACHER_COMPAT": "anthropic"}),
                         "anthropic")
        self.assertEqual(compat.compat_mode("teacher", {}), "openai")

    def test_base_url_v1_suffix(self):
        c = compat.AnthropicCompatClient(
            base_url="https://fake.test/v1", api_key="k",
            _transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json=_anthropic_response(
                    [{"type": "text", "text": "ok"}]))))
        self.assertEqual(c._url(), "https://fake.test/v1/messages")
        c2 = compat.AnthropicCompatClient(
            base_url="https://fake.test", api_key="k",
            _transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json=_anthropic_response(
                    [{"type": "text", "text": "ok"}]))))
        self.assertEqual(c2._url(), "https://fake.test/v1/messages")


if __name__ == "__main__":
    unittest.main()
