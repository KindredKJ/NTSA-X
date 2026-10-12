"""Evidence-gated convergence; internal agreement never substitutes for proof."""
from dataclasses import dataclass
from typing import Mapping, Sequence

REQUIRED_CHECKS = ("state", "semantic", "causal", "constraint", "evidence", "delta")


@dataclass(frozen=True)
class Check:
    category: str
    passed: bool
    source: str
    external: bool = False

    def __post_init__(self):
        if self.category not in REQUIRED_CHECKS:
            raise ValueError(f"Unknown verification category: {self.category}")
        if not self.source or not self.source.strip():
            raise ValueError("Verification check requires an attributed source")
        if type(self.passed) is not bool or type(self.external) is not bool:
            raise TypeError("passed and external must be bool")


@dataclass(frozen=True)
class ConvergenceReport:
    converged: bool
    score: float
    failed: tuple[str, ...]
    missing: tuple[str, ...]
    contradictions: tuple[str, ...]
    counter_orbit_agreement: bool | None
    external_evidence: bool


class ConvergenceEngine:
    def evaluate(
        self,
        checks: Sequence[Check],
        *,
        contradictions: Sequence[str] = (),
        forward: Mapping | None = None,
        reverse: Mapping | None = None,
    ) -> ConvergenceReport:
        if (forward is None) != (reverse is None):
            raise ValueError("Counter-orbital comparison requires both trajectories")
        checks = tuple(checks)
        if any(not isinstance(check, Check) for check in checks):
            raise TypeError("Verifier must return Check records")
        found = {category: [] for category in REQUIRED_CHECKS}
        for check in checks:
            found[check.category].append(check)
        missing = tuple(category for category, group in found.items() if not group)
        failed = tuple(category for category, group in found.items() if group and not all(c.passed for c in group))
        # A self-reported evidence check is not an external attestation.
        external_evidence = any(c.passed and c.external for c in found["evidence"])
        agreement = None if forward is None else dict(forward) == dict(reverse)
        contradiction_items = tuple(str(item) for item in contradictions)
        successful_categories = sum(bool(found[c]) and all(v.passed for v in found[c]) for c in REQUIRED_CHECKS)
        score = successful_categories / len(REQUIRED_CHECKS)
        converged = (
            not missing
            and not failed
            and not contradiction_items
            and external_evidence
            and agreement is not False
        )
        return ConvergenceReport(
            bool(converged), score, failed, missing,
            contradiction_items, agreement, external_evidence,
        )
