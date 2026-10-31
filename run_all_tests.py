"""Run every task's test suite in sequence and print a summary.

Each suite starts its own server subprocesses on fixed local ports, so they
run one at a time. Usage: python run_all_tests.py
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
SUITES = [
    ("task-1-mcp-server", "test_stdio_isolation.py"),
    ("task-1-mcp-server", "test_adversarial.py"),
    ("task-1-mcp-server", "test_hardening.py"),
    ("task-2-mcp-gateway", "test_gateway.py"),
    ("task-2-mcp-gateway", "test_gateway_hardening.py"),
    ("task-3-llm-guardrail", "test_redaction_unit.py"),
    ("task-3-llm-guardrail", "test_redaction.py"),
    ("task-3-llm-guardrail", "test_gateway_sse.py"),
    ("task-4-rate-limit-router", "test_router.py"),
    ("task-4-rate-limit-router", "test_concurrency.py"),
    ("task-4-rate-limit-router", "test_cross_tenant_latency.py"),
    ("task-4-rate-limit-router", "test_extended_probes.py"),
    ("task-4-rate-limit-router", "test_hardening.py"),
]


def main() -> None:
    results = []
    for folder, script in SUITES:
        start = time.monotonic()
        proc = subprocess.run(
            [sys.executable, script], cwd=ROOT / folder, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=600,
        )
        elapsed = time.monotonic() - start
        checks = [line for line in proc.stdout.splitlines() if line.startswith(("[PASS]", "[FAIL]"))]
        passed = sum(line.startswith("[PASS]") for line in checks)
        ok = proc.returncode == 0
        results.append(ok)
        print(f"{'OK  ' if ok else 'FAIL'} {folder + '/' + script:<55} {passed:>3}/{len(checks):<3} checks  {elapsed:5.1f}s")
        if not ok:
            for line in checks:
                if line.startswith("[FAIL]"):
                    print(f"       {line}")
            if not checks:
                print("       " + (proc.stderr.strip().splitlines() or ["(no output)"])[-1])

    print(f"\n{sum(results)}/{len(results)} suites passed.")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
