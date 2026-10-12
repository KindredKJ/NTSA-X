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
    sys.exit(main())
