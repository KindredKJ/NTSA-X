"""Fixed-function subprocess worker for NTSA-X test plugins.

The worker reads exactly one bounded JSON command and returns exactly one JSON
response. No imports from plugin packages, arbitrary Python, eval, or shell.
This is NOT an OS-level sandbox or a safe way to run adversarial code.
"""
import json
import sys

MAX_BYTES = 8192
OPERATIONS = frozenset(("increment", "decrement", "identity", "negate"))


def dispatch(command):
    if not isinstance(command, dict) or set(command) != {"operation", "state"}:
        raise ValueError("Invalid command fields")
    operation, state = command["operation"], command["state"]
    if operation not in OPERATIONS:
        raise ValueError("Unknown worker operation")
    if not isinstance(state, dict) or set(state) != {"value", "amount"}:
        raise ValueError("Only value and amount are accepted")
    value, amount = state["value"], state["amount"]
    if type(value) is not int or type(amount) is not int:
        raise ValueError("Values must be integers")
    if abs(value) > 1_000_000_000 or not 0 <= amount <= 1_000_000:
        raise ValueError("Input outside approved limits")
    if operation == "increment":
        output = value + amount
    elif operation == "decrement":
        output = value - amount
    elif operation == "negate":
        output = -value
    else:
        output = value
    if abs(output) > 1_000_000_000:
        raise ValueError("Output outside approved limits")
    return {"value": output}


def serve():
    """Bounded JSON-lines RPC; accepts only approved scalar operations.

    Each request is isolated by protocol validation, not by OS sandboxing.
    The caller supervises this process and terminates it on timeout.
    """
    sys.stdout.write('{"ready":true,"protocol":"ntsa-fixed/v1"}\\n')
    sys.stdout.flush()
    while True:
        raw = sys.stdin.buffer.readline(MAX_BYTES + 1)
        if not raw:
            return 0
        if len(raw) > MAX_BYTES or not raw.endswith(b"\\n"):
            return 2  # Kill protocol session rather than desynchronize frames.
        request_id = None
        try:
            packet = json.loads(raw)
            if not isinstance(packet, dict) or set(packet) != {"id", "operation", "state"}:
                raise ValueError("Bad RPC fields")
            request_id = packet["id"]
            if not isinstance(request_id, str) or not 0 < len(request_id) <= 64:
                raise ValueError("Invalid request ID")
            result = dispatch({"operation": packet["operation"], "state": packet["state"]})
            response = {"id": request_id, "ok": True, "state": result}
        except (ValueError, TypeError, UnicodeDecodeError):
            response = {"id": request_id, "ok": False, "error": "invalid_request"}
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\\n")
        sys.stdout.flush()


def main():
    raw = sys.stdin.buffer.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        return 2
    try:
        packet = json.loads(raw)
        response = {"ok": True, "state": dispatch(packet)}
    except (ValueError, TypeError, UnicodeDecodeError):
        response = {"ok": False, "error": "invalid_request"}
    sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(serve() if sys.argv[1:] == ["--serve"] else main())
