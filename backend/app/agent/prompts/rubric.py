"""The resume-match rubric prompt.

Only 15% of the final score, deliberately. The model is asked to judge fit
against retrieved evidence from the candidate's own resume — not to invent a
number from a vibe. The other 85% is arithmetic the user can check.
"""

RUBRIC_PROMPT_VERSION = "2026-08-30.1"

RUBRIC_SYSTEM_PROMPT = """\
You assess how well a candidate fits a role, for the candidate's own use. They \
are deciding where to spend limited application effort, so useful beats kind: \
an inflated score wastes their week.

You are given the role's requirements and, for each one, the passages retrieved \
from the candidate's actual resume. Judge ONLY against those passages.

Rules:

1. EVIDENCE ONLY. If nothing in the retrieved passages supports a requirement, \
it is not met. Do not assume a Python developer knows Django, or that someone \
with AWS experience knows Kubernetes. Absence of evidence is absence.

2. STRENGTHS must quote or closely paraphrase something the resume actually \
says. A strength you cannot point at is not a strength.

3. GAPS should name the specific missing requirement, not a vague deficiency. \
"No Kafka or streaming experience shown" — not "could be stronger technically".

4. SCORE 0.0-1.0 on evidenced fit alone. Ignore how prestigious the company is, \
how well the resume is written, and how enthusiastic the posting sounds. \
0.9+ means nearly every requirement is evidenced; 0.5 means about half; below \
0.3 means the candidate would be screened out.

5. NARRATIVE: two or three sentences, addressed to the candidate, saying \
whether this is worth applying to and what would most improve their odds. Be \
concrete and be honest."""


def build_rubric_prompt(
    *,
    title: str,
    company: str,
    requirements_with_evidence: list[tuple[str, list[str]]],
    matched_skills: list[str],
    missing_skills: list[str],
) -> str:
    lines = [f"ROLE: {title} at {company}", ""]

    if matched_skills:
        lines.append(f"Skills the resume evidences: {', '.join(matched_skills)}")
    if missing_skills:
        lines.append(f"Required skills not found in the resume: {', '.join(missing_skills)}")
    lines.append("")
    lines.append("REQUIREMENTS, each with the passages retrieved from the resume:")
    lines.append("")

    for index, (requirement, evidence) in enumerate(requirements_with_evidence, start=1):
        lines.append(f"{index}. {requirement}")
        if evidence:
            for passage in evidence:
                lines.append(f"   > {passage}")
        else:
            lines.append("   > (nothing relevant found in the resume)")
        lines.append("")

    return "\n".join(lines)
