---
name: review-pr
description: Deep code review of a branch as a PR against dev. Focus on intent, correctness, reuse, and test semantics.
argument-hint: <branch-name> [base-branch]
allowed-tools: [Read, Glob, Grep, Bash, Agent]
---

# PR Review

Review the branch `$ARGUMENTS` as a pull request. If no base branch is given after the branch name, default to `dev`.

## Setup

1. Parse arguments: first token is the feature branch, optional second token is the base branch.
2. Fetch and diff:
   ```
   git fetch origin <branch>
   git log --oneline <base>..<branch>
   git diff --stat <base>..<branch>
   git diff <base>..<branch>
   ```
3. If the diff is large (>800 lines), launch parallel Explore agents to study affected areas of the codebase. Otherwise, read the changed files and their surrounding infrastructure directly.

## Review Methodology

Work through these phases in order. Each phase builds understanding that informs the next.

### Phase 1: Understand intent

Before evaluating any code, answer:
- **What problem does this PR solve?** Derive this from the diff, commit messages, and any linked issues.
- **What is the ideal solution to that problem?** Consider the codebase as it exists today. What is the most direct, minimal way to achieve the goal using existing infrastructure?
- **How does the PR's approach compare to the ideal?** Where does it diverge, and is each divergence justified?

This framing prevents reviewing code in isolation. Every subsequent judgment — "is this correct?", "is this redundant?" — is anchored to whether it serves the intent effectively.

### Phase 2: Correctness

Evaluate whether every changed file — production and test code alike — is correct:

- **Production code**: Does the change achieve the stated intent? Are there edge cases, off-by-one errors, or silent failures? Trace the data flow from entry point through to effect.
- **Test code**: What property does each test claim to verify? Trace from inputs through execution to assertions. Can the test produce false passes (assertions too loose, setup makes the test trivially true)? Can it produce false failures (tolerances too tight, depends on unrelated behavior)?

### Phase 3: Approach and reuse

For every changed file — production and test — evaluate whether the approach is the best way to accomplish its goal:

- **Are there simpler alternatives?** Could existing code paths, utilities, helpers, runners, or APIs be used instead of writing new code? Would a different design reduce complexity or risk?
- **Is new code justified?** When the PR introduces new functions, helpers, or patterns, determine whether existing infrastructure could serve the same purpose. Be precise: identify the specific existing code, what it does, and what gap (if any) prevents direct reuse.
- **Critical nuance on reuse**: A helper that does not return intermediate values (e.g., asserts internally but doesn't expose results) cannot be reused for code that needs those intermediates. Acknowledge structural limitations honestly. Do NOT recommend reuse that requires modifying existing APIs — that changes scope. Instead, note what existing code *does* expose and suggest the minimal composition that avoids reimplementation.

The distinction to draw:
- **Avoidable duplication**: input construction, env var setup, model instantiation, config validation — things existing code already does and exposes.
- **Unavoidable divergence**: the new code genuinely needs something the existing infrastructure wasn't designed for.

### Phase 4: Minimality and integration

- **Minimality**: Does the PR touch only what it needs to? Flag unrelated cleanups, speculative additions, or defensive code that guards against impossible states. Does it test things beyond its stated goal that are already covered elsewhere (adding fragility without coverage)?
- **Integration**: Does the new code fit surrounding conventions? Does it introduce a second way to do something that already has an established pattern?
- **Efficiency** (tests): Are test variants split across multiple functions when they could be a single parametrized function? Does the parametrization create excessive combinations relative to marginal coverage?
- **Missing coverage**: Given the production change, are there scenarios that should be tested but aren't?

## Output Format

```
## PR Review: <branch> -> <base>

### Intent
[What the PR is trying to accomplish and whether the approach is sound]

### Correctness
[Per-file: does it work? Gaps, edge cases, false-pass/false-fail risks]

### Reuse and Approach
[What existing infrastructure could be leveraged, what was reimplemented, what's justified vs. avoidable]

### Minimality and Integration
[Scope creep, convention fit, parametrization efficiency, missing coverage]

### Summary
| Area | Verdict | Key Issues |
|------|---------|------------|
| ...  | ...     | ...        |
```
