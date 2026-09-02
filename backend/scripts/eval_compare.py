"""Compare two eval runs — the model-selection question, answered reproducibly.

This project already chose a model by measurement once, by hand, and wrote the
result down three times: README.md says a ten-case eval over a "20-tool schema"
with gpt-4o-mini at 10/10, core/config.py repeats it with rupee costs, and
.env.example says "22-tool" and 9/10. Only the last is right about the tool
count. Three sources of truth is how they came to disagree, and a table built by
hand is why there was nothing to re-run when they did.

    python scripts/eval_compare.py evals/runs/<a>.json evals/runs/<b>.json

Refuses to compare runs whose prompt versions or tool count differ unless told
to. Comparing a model against a different prompt measures both at once and
attributes the result to whichever you were thinking about.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Hard-gated metrics: any regression at all is a regression. The rest get a
# tolerance, because a stochastic system moves a point or two between runs and
# flagging that would make every comparison look alarming.
EXACT = {
    "extraction.skill_precision",
    "extraction.salary_verbatim",
    "extraction.needs_review",
    "assistant.forbidden_tool",
    "assistant.query_fidelity",
    "assistant.no_invented_dates",
    "rubric.evidence_only",
}

COMPARABLE = (
    "extraction_prompt_version",
    "assistant_prompt_version",
    "rubric_prompt_version",
    "tool_count",
)


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_comparable(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    return [
        f"{key}: {a['run'].get(key)} vs {b['run'].get(key)}"
        for key in COMPARABLE
        if a["run"].get(key) != b["run"].get(key)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument(
        "--force",
        action="store_true",
        help="compare even when the prompts or the tool count differ",
    )
    args = parser.parse_args()

    a, b = load(args.baseline), load(args.candidate)

    if differences := check_comparable(a, b):
        print("These runs are not directly comparable:")
        for line in differences:
            print(f"  {line}")
        if not args.force:
            print("\nRe-run both against the same prompts, or pass --force and say so.")
            return 2
        print("\nComparing anyway (--force). The delta includes the prompt change.\n")

    for run, label in ((a, "baseline"), (b, "candidate")):
        header = run["run"]
        print(
            f"{label:<10} {header['provider']}/{header['model']:<28} "
            f"{header['total_tokens']:>7} tokens  {header['git_sha']}"
        )
    print()

    names = sorted(set(a["metrics"]) | set(b["metrics"]))
    width = max(len(n) for n in names)
    regressions = 0

    print(f"{'metric':<{width}}  {'baseline':>9} {'candidate':>10} {'delta':>8}")
    print("-" * (width + 32))
    for name in names:
        before, after = a["metrics"].get(name), b["metrics"].get(name)
        if before is None or after is None:
            missing = "baseline" if before is None else "candidate"
            print(
                f"{name:<{width}}  {'—' if before is None else f'{before:9.3f}'} "
                f"{'—' if after is None else f'{after:10.3f}'} {'':>8}  not in {missing}"
            )
            continue

        delta = after - before
        allowed = 0.0 if name in EXACT else args.tolerance
        regressed = delta < -allowed
        regressions += regressed
        flag = "  REGRESSED" if regressed else ""
        print(f"{name:<{width}}  {before:9.3f} {after:10.3f} {delta:+8.3f}{flag}")

    print()
    if regressions:
        print(
            f"{regressions} metric(s) regressed. Tolerance {args.tolerance}, zero for hard gates."
        )
    else:
        print("No regressions.")
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
