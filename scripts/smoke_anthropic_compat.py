"""End-to-end smoke test for the Anthropic compat layer -- NO network.

Starts a local scripted fake of Anthropic's native ``/v1/messages`` API and
drives ``AnthropicCompatClient`` through the exact shapes the real runners
use (run_qa_pairs / run_beta_doc / judge):

  1. research turn (tools + temperature)  -> llm_params 400-retry path,
     thinking captured as reasoning_content, native tool_use -> tool_calls
  2. tool-result turn                     -> tool_result round trip, final
     strict {"summary","contract"} envelope parsed from content
  3. draft turn with response_format      -> request carries native
     output_config.format (constrained decoding = envelope guarantee)
  4. judge-style streaming                -> delta.content consumption

Run:  PYTHONPATH=src python scripts/smoke_anthropic_compat.py
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / ".." / "src"))

from legal_drafter import compat, llm_params  # noqa: E402

TOOL_SCHEMAS = [{
    "type": "function",
    "function": {
        "name": "keyword_search",
        "description": "Exact lexical match for known terms",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}]

ENVELOPE = {"summary": "wynajem mieszkania na 18 miesiecy",
            "contract": "UMOWA NAJMU MIESZKANIA\n\nArt. 659 KC ... " * 40}

THINKING = ("Plan: najpierw get_template, potem keyword_search po art. 659 KC.")


class FakeAnthropic:
    """Scripted /v1/messages: records requests, answers by turn shape."""

    def __init__(self):
        self.bodies: list[dict] = []
        self.reject_temperature = True

    def handler(self, request_body: dict) -> tuple[int, dict]:
        self.bodies.append(request_body)
        if self.reject_temperature and "temperature" in request_body:
            return 400, {"error": {"type": "invalid_request_error",
                                   "message": ("temperature is deprecated "
                                               "for this model")}}
        last = request_body["messages"][-1]
        blocks = last.get("content") or []
        has_tool_result = any(b.get("type") == "tool_result" for b in blocks
                              if isinstance(b, dict))
        if "output_config" in request_body:            # 3. draft phase
            return 200, self._message(text=json.dumps(ENVELOPE,
                                                       ensure_ascii=False))
        if has_tool_result:                            # 2. final research answer
            return 200, self._message(text=json.dumps(ENVELOPE,
                                                       ensure_ascii=False))
        return 200, self._message(                     # 1. tool call
            thinking=THINKING,
            tool_use=("toolu_smoke1", "keyword_search",
                      {"query": "najem mieszkanie art 659"}),
            stop_reason="tool_use")

    @staticmethod
    def _message(text=None, thinking=None, tool_use=None,
                 stop_reason="end_turn"):
        content = []
        if thinking:
            content.append({"type": "thinking", "thinking": thinking,
                            "signature": "sigFAKEsmoke"})
        if tool_use:
            tid, name, inp = tool_use
            content.append({"type": "tool_use", "id": tid, "name": name,
                            "input": inp})
        if text:
            content.append({"type": "text", "text": text})
        return {"id": "msg_smoke", "type": "message", "role": "assistant",
                "model": "claude-sonnet-5", "content": content,
                "stop_reason": stop_reason,
                "usage": {"input_tokens": 120, "output_tokens": 300}}


class _HTTPHandler(BaseHTTPRequestHandler):
    fake: FakeAnthropic

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        status, payload = self.fake.handler(body)
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):  # silence
        pass


def consume_stream(resp):
    """Replicates run_beta_doc._stream_turn's chunk consumption."""
    parts, reasoning_parts, tool_calls = [], [], []
    usage = None
    for chunk in resp:
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        if d is None:
            continue
        if getattr(d, "content", None):
            parts.append(d.content)
        r = getattr(d, "reasoning", None) or getattr(d, "reasoning_content", None)
        if r:
            reasoning_parts.append(r)
        for tc in getattr(d, "tool_calls", None) or []:
            tool_calls.append((tc.id, tc.function.name,
                               tc.function.arguments))
    return "".join(parts), "".join(reasoning_parts), tool_calls, usage


def main() -> int:
    fake = FakeAnthropic()
    _HTTPHandler.fake = fake
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HTTPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    failures: list[str] = []

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f" -- {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    try:
        # Phase A: thinking OFF -> temperature reaches the API -> native 400
        # -> llm_params retries once without it and caches the alias.
        client_off = compat.AnthropicCompatClient(
            base_url=base, api_key="smoke", thinking_mode="off")
        messages = [{"role": "system", "content": "You are a Polish drafter."},
                    {"role": "user", "content": "Potrzebuje umowy najmu."}]
        resp = llm_params.chat_create(
            client_off, role="teacher", model="claude-sonnet-5", stream=True,
            temperature=0.2, tools=TOOL_SCHEMAS, tool_choice="auto",
            messages=messages)
        text, reasoning, tool_calls, usage = consume_stream(resp)
        check("temperature 400 retried without it (1st had it, 2nd clean)",
              len(fake.bodies) == 2 and "temperature" in fake.bodies[0]
              and "temperature" not in fake.bodies[1])
        check("tool_call translated (id/name/json args)",
              tool_calls and tool_calls[0][0] == "toolu_smoke1"
              and tool_calls[0][1] == "keyword_search"
              and json.loads(tool_calls[0][2]).get("query"))

        # Phase B: default adaptive thinking -> thinking summary surfaces as
        # reasoning_content; the compat layer locks temperature itself (no 400).
        client = compat.AnthropicCompatClient(base_url=base, api_key="smoke")
        resp = llm_params.chat_create(
            client, role="teacher", model="claude-opus-4-8", stream=True,
            temperature=0.2, tools=TOOL_SCHEMAS, tool_choice="auto",
            messages=messages)
        text, reasoning, tool_calls, usage = consume_stream(resp)
        check("single request (compat temperature-lock pre-empts the 400)",
              len(fake.bodies) == 3)
        check("temperature never sent while thinking is on",
              "temperature" not in fake.bodies[-1])
        check("reasoning surfaced as reasoning_content", reasoning == THINKING,
              repr(reasoning[:60]))
        check("usage mapped (prompt_tokens=120)", usage.prompt_tokens == 120)

        # --- 2. tool-result turn -> final envelope
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [{"id": "toolu_smoke1",
                                         "type": "function",
                                         "function": {"name": "keyword_search",
                                                      "arguments": tool_calls[0][2]}}]})
        messages.append({"role": "tool", "tool_call_id": "toolu_smoke1",
                         "name": "keyword_search",
                         "content": json.dumps(
                             [{"chunk_id": "kc_659", "text": "art. 659. ..."}])})
        resp = llm_params.chat_create(
            client, role="teacher", model="claude-sonnet-5", stream=True,
            tools=TOOL_SCHEMAS, tool_choice="auto", messages=messages)
        text, reasoning, tool_calls, _usage = consume_stream(resp)
        parsed = json.loads(text) if text else {}
        check("envelope returned as content", parsed.get("summary") == ENVELOPE["summary"]
              and len(parsed.get("contract", "")) > 500)
        check("tool_use id round-trips in tool-result request history",
              any(b.get("type") == "tool_use" and b.get("id") == "toolu_smoke1"
                  for b in fake.bodies[-1]["messages"][-2]["content"]))

        # --- 3. draft phase with response_format -> native output_config
        schema = {"type": "object",
                  "properties": {"summary": {"type": "string"},
                                 "contract": {"type": "string"}},
                  "required": ["summary", "contract"],
                  "additionalProperties": False}
        resp = llm_params.chat_create(
            client, role="teacher", model="claude-sonnet-5", stream=True,
            max_tokens=16384, messages=[
                {"role": "user", "content": "draft the contract"}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "Draft", "strict": True,
                                             "schema": schema}})
        text, reasoning, _tc, _u = consume_stream(resp)
        drafted = json.loads(text) if text else {}
        check("output_config.format present in draft request",
              fake.bodies[-1].get("output_config", {})
              .get("format", {}).get("type") == "json_schema")
        check("draft envelope parsed from constrained output",
              drafted.get("contract", "").startswith("UMOWA"))

        # --- 4. judge-style: content-only consumption already covered above

        # --- summary
        n_temp = sum(1 for b in fake.bodies if "temperature" in b)
        check("exactly one temperature-bearing request (the 400, then locked)",
              n_temp == 1)
    finally:
        server.shutdown()
        server.server_close()

    print(f"\nSMOKE {'FAILED: ' + ', '.join(failures) if failures else 'PASSED'}"
          f" ({len(fake.bodies)} requests)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
