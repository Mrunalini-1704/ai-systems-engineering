"""Customer service MCP server.

Exposes two tools over stdio: get_customer_record and trigger_refund.
stdout is reserved exclusively for JSON-RPC (owned by the MCP stdio
transport) -- all logging goes to stderr.
"""

import sys
import logging

# Configured before anything else can log, and pointed at stderr so stdout
# stays pure JSON-RPC.
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("customer-service-mcp")

import asyncio
import json
import os
import re
import typing
from collections.abc import AsyncIterator
from io import TextIOWrapper

import anyio

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server
from mcp.shared.exceptions import McpError

# fullmatch() + [0-9] rather than match() + ^...\d$: "$" also matches just
# before a trailing newline ("CUST-00001\n" would pass), and \d matches any
# Unicode decimal digit (e.g. Arabic-Indic U+0660), not just ASCII 0-9.
CUSTOMER_ID_PATTERN = re.compile(r"CUST-[0-9]{5}")
CUSTOMER_ID_ERROR = "customer_id must match CUST-XXXXX (X = digit 0-9), e.g. CUST-00042"


def _check_customer_id(v: str) -> str:
    if not CUSTOMER_ID_PATTERN.fullmatch(v):
        raise ValueError(CUSTOMER_ID_ERROR)
    return v


class GetCustomerRecordInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(..., description="Customer ID formatted as CUST-XXXXX")

    @field_validator("customer_id")
    @classmethod
    def validate_customer_id(cls, v: str) -> str:
        return _check_customer_id(v)


class TriggerRefundInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str
    amount: float = Field(
        ..., gt=0, strict=True, allow_inf_nan=False, description="Refund amount, must be positive and finite"
    )

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: float) -> float:
        # Refunds are paid in whole cents (the result is rounded to 2 places),
        # so 0.001 would "succeed" as a refund of 0.00.
        if v < 0.01:
            raise ValueError("amount must be at least 0.01")
        return v
    reason: str = Field(..., min_length=10, description="Reason for refund, minimum 10 characters")

    @field_validator("customer_id")
    @classmethod
    def validate_customer_id(cls, v: str) -> str:
        return _check_customer_id(v)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, v: str) -> str:
        # min_length alone would accept 10 spaces as a "reason".
        stripped = v.strip()
        if len(stripped) < 10:
            raise ValueError("reason must have at least 10 characters excluding leading/trailing whitespace")
        return stripped


MOCK_CUSTOMERS = {
    "CUST-00001": {"name": "Alice Johnson", "email": "alice@example.com", "status": "active"},
    "CUST-00002": {"name": "Bob Smith", "email": "bob@example.com", "status": "active"},
}

server: Server = Server("customer-service-mcp")


def _validation_error_message(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "input"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_customer_record",
            description="Fetch a customer record by customer ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "pattern": "^CUST-[0-9]{5}$",
                        "description": "Customer ID formatted as CUST-XXXXX",
                    }
                },
                "required": ["customer_id"],
                "additionalProperties": False,
            },
        ),
        types.Tool(
            name="trigger_refund",
            description="Trigger a refund for a customer.",
            inputSchema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "pattern": "^CUST-[0-9]{5}$",
                    },
                    "amount": {
                        "type": "number",
                        "minimum": 0.01,
                        "description": "Positive refund amount",
                    },
                    "reason": {
                        "type": "string",
                        "minLength": 10,
                        "description": "Reason for the refund (min 10 characters)",
                    },
                },
                "required": ["customer_id", "amount", "reason"],
                "additionalProperties": False,
            },
        ),
    ]


# NOTE: deliberately NOT using the @server.call_tool() convenience decorator.
# That decorator wraps every handler in a blanket try/except that swallows
# any exception (including McpError) into a tool-level `isError: true`
# CallToolResult -- i.e. still HTTP 200 / JSON-RPC "result", just flagged as
# an error. That's correct for *tool execution* failures, but this task
# requires malformed input to come back as an actual top-level JSON-RPC
# error object (standard error codes like -32602). The low-level request
# loop (`Server._handle_request`) only produces a genuine JSON-RPC error
# when McpError escapes the registered handler directly, so we register the
# CallToolRequest handler ourselves to get that propagation.
async def handle_call_tool(req: types.CallToolRequest) -> types.ServerResult:
    name = req.params.name
    arguments = req.params.arguments or {}
    logger.info("call_tool invoked: name=%s arguments=%s", name, arguments)

    if name == "get_customer_record":
        try:
            parsed = GetCustomerRecordInput.model_validate(arguments)
        except ValidationError as exc:
            message = _validation_error_message(exc)
            logger.warning("validation failed for get_customer_record: %s", message)
            raise McpError(types.ErrorData(code=types.INVALID_PARAMS, message=message))

        record = MOCK_CUSTOMERS.get(parsed.customer_id)
        if record is None:
            # Well-formed input, but the lookup itself failed: that's a tool
            # *execution* error (an isError result the model can see and react
            # to), not a protocol error.
            content = [types.TextContent(type="text", text=f"No record found for {parsed.customer_id}")]
            return types.ServerResult(types.CallToolResult(content=content, isError=True))
        content = [types.TextContent(type="text", text=json.dumps({"customer_id": parsed.customer_id, **record}))]
        return types.ServerResult(types.CallToolResult(content=content, isError=False))

    if name == "trigger_refund":
        try:
            parsed = TriggerRefundInput.model_validate(arguments)
        except ValidationError as exc:
            message = _validation_error_message(exc)
            logger.warning("validation failed for trigger_refund: %s", message)
            raise McpError(types.ErrorData(code=types.INVALID_PARAMS, message=message))

        if parsed.customer_id not in MOCK_CUSTOMERS:
            content = [types.TextContent(type="text", text=f"No record found for {parsed.customer_id}")]
            return types.ServerResult(types.CallToolResult(content=content, isError=True))

        logger.info("refund triggered: customer=%s amount=%.2f", parsed.customer_id, parsed.amount)
        result = {
            "status": "refund_triggered",
            "customer_id": parsed.customer_id,
            "amount": round(parsed.amount, 2),
            "reason": parsed.reason,
        }
        content = [types.TextContent(type="text", text=json.dumps(result))]
        return types.ServerResult(types.CallToolResult(content=content, isError=False))

    # The MCP spec (server/tools, "Error Handling") classifies an unknown
    # tool as a protocol error with -32602 INVALID_PARAMS: the *method*
    # (tools/call) exists, it's the `name` param that's invalid.
    logger.warning("unknown tool requested: %s", name)
    raise McpError(types.ErrorData(code=types.INVALID_PARAMS, message=f"Unknown tool: {name}"))


server.request_handlers[types.CallToolRequest] = handle_call_tool


# Every request method a client may send, with its SDK request model.
# Anything else is answered with -32601 below; the SDK itself would answer an
# unknown method with -32602.
REQUEST_TYPES = {
    typing.get_args(cls.model_fields["method"].annotation)[0]: cls
    for cls in typing.get_args(types.ClientRequest.model_fields["root"].annotation)
}


def _jsonrpc_error_line(req_id, code: int, message: str) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}) + "\n"


def _precheck(line: str) -> tuple[types.JSONRPCMessage | None, str | None]:
    """Return (message, None) for valid input, or (None, a JSON-RPC error
    line) for input the SDK would mishandle.

    The SDK's stdio transport turns an unparseable line into an internal
    exception: the server only emits a "notifications/message" log and the
    caller never gets a response. JSON-RPC 2.0 requires -32700 for invalid
    JSON, -32600 for JSON that isn't a valid request object (e.g. a batch
    array or "jsonrpc": "1.0"), -32601 for an unknown method, and -32602
    for invalid params. The SDK does answer bad params, but only with a
    generic "Invalid request parameters" (after logging ~6 KB of union
    validation errors); checking against the method's own model gives the
    caller the actual problem ("params.name: Field required").
    """
    try:
        message = types.JSONRPCMessage.model_validate_json(line)
    except ValidationError as exc:
        if any(err["type"] == "json_invalid" for err in exc.errors()):
            return None, _jsonrpc_error_line(None, types.PARSE_ERROR, "Parse error")
        req_id = None
        try:
            obj = json.loads(line)
        except (ValueError, RecursionError):
            obj = None
        if isinstance(obj, dict) and isinstance(obj.get("id"), (str, int)) and not isinstance(obj["id"], bool):
            req_id = obj["id"]
        if (isinstance(obj, dict) and obj.get("jsonrpc") == "2.0" and isinstance(obj.get("method"), str)
                and isinstance(obj.get("params"), list)):
            # Structurally a JSON-RPC request (which allows by-position
            # params), but MCP params are always an object.
            return None, _jsonrpc_error_line(req_id, types.INVALID_PARAMS, "params must be an object")
        return None, _jsonrpc_error_line(req_id, types.INVALID_REQUEST, "Invalid Request")
    request = message.root
    if isinstance(request, types.JSONRPCRequest):
        request_type = REQUEST_TYPES.get(request.method)
        if request_type is None:
            return None, _jsonrpc_error_line(request.id, types.METHOD_NOT_FOUND, f"Method not found: {request.method}")
        fields = {"method": request.method}
        if request.params is not None:
            fields["params"] = request.params
        try:
            request_type.model_validate(fields)
        except ValidationError as exc:
            return None, _jsonrpc_error_line(request.id, types.INVALID_PARAMS, _validation_error_message(exc))
    return message, None


# Upper bound on how long to keep serving after stdin closes (see _checked_stdin).
DRAIN_TIMEOUT_SECONDS = 10.0


class _ProtocolWriter:
    """The protocol fd's only writer. The SDK transport and the precheck in
    _checked_stdin both write through it; the lock keeps their lines whole.

    It also tracks which requests are still unanswered (`pending`), so that
    _checked_stdin can hold off EOF until they have been answered."""

    def __init__(self, out: "anyio.AsyncFile[str]") -> None:
        self._out = out
        self._lock = anyio.Lock()
        self.pending: set[str | int] = set()

    async def write(self, data: str) -> None:
        async with self._lock:
            await self._out.write(data)
            await self._out.flush()
        msg = json.loads(data)
        if "result" in msg or "error" in msg:
            self.pending.discard(msg.get("id"))

    async def flush(self) -> None:
        pass  # write() already flushed


async def _checked_stdin(protocol_out: _ProtocolWriter) -> AsyncIterator[str]:
    stdin = anyio.wrap_file(TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace"))
    async for line in stdin:
        if not line.strip():
            continue
        message, error = _precheck(line)
        if error is not None:
            logger.warning("rejected malformed JSON-RPC input: %s", error.strip())
            await protocol_out.write(error)
            continue
        if isinstance(message.root, types.JSONRPCRequest):
            protocol_out.pending.add(message.root.id)
        yield line
    # stdin is closed. Ending this iterator makes the SDK stop serving and
    # drop requests still in flight, so a client that writes its last request
    # and closes stdin (e.g. `cat requests.jsonl | python server.py`) would
    # never get that response. Wait (bounded) for those responses first.
    with anyio.move_on_after(DRAIN_TIMEOUT_SECONDS):
        while protocol_out.pending:
            await anyio.sleep(0.01)


def _reserve_stdout_for_protocol() -> "anyio.AsyncFile[str]":
    """Hand the real stdout to the JSON-RPC transport exclusively.

    Logging to stderr only covers code that uses `logging`. A stray print()
    -- ours, a dependency's, or a C extension / child process writing to
    file descriptor 1 directly -- would still corrupt the protocol stream.
    So: duplicate fd 1 to a private fd that only the transport writes to,
    then repoint fd 1 *and* sys.stdout at stderr. After this, anything that
    isn't the transport and tries to write to "stdout" lands on stderr.
    """
    sys.stdout.flush()
    stdout_fd = sys.stdout.fileno()
    protocol_fd = os.dup(stdout_fd)
    os.dup2(sys.stderr.fileno(), stdout_fd)
    sys.stdout = sys.stderr
    # newline="\n": no "\r\n" translation on Windows -- MCP stdio messages
    # are newline-delimited.
    protocol_out = TextIOWrapper(os.fdopen(protocol_fd, "wb"), encoding="utf-8", newline="\n")
    return anyio.wrap_file(protocol_out)


async def main() -> None:
    protocol_stdout = _ProtocolWriter(_reserve_stdout_for_protocol())
    logger.info("starting customer-service-mcp over stdio")
    async with mcp.server.stdio.stdio_server(stdin=_checked_stdin(protocol_stdout), stdout=protocol_stdout) as (
        read_stream,
        write_stream,
    ):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())