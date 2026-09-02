"""Measuring the half of the agent that unit tests cannot reach.

Every test under `tests/` stubs the model, deliberately — CI must not depend on
a third-party API. That leaves exactly one thing unmeasured, and it happens to
be the thing the prompts are made of: whether a real model obeys the rules it
was given. Nothing catches a prompt edit that fixes one posting and breaks three
others, which is what `README.md` means by "caught by review rather than by a
test".

Three things follow from that, and they shape everything here.

**Scoring is deterministic wherever it can be.** A model grading a model is a
second thing to debug. Most of what these prompts claim is mechanically
checkable: a skill either appears in the posting or it does not, a tool either
was called or was not, a quoted salary either is a substring of the source or is
invented. Judgement is reserved for the few places nothing else works.

**Ground truth is sparse and declared.** A case says what it is about and
nothing more. Whole golden objects would force every case to be right about
prose nobody reads, and turn every prompt tweak into a diff people approve
without looking.

**Aggregate gating, per-case reporting.** A stochastic system graded case by
case shows red on a good day and teaches everyone to ignore it. Thresholds are
per surface; the failure message names which cases moved.
"""
