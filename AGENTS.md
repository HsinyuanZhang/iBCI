# Global agent rules

## Communication

- Default to Simplified Chinese for user-facing text; keep code, commands, and technical identifiers in English.
- Lead with impact and conclusions, followed by actions, decisions needed, and essential evidence. Omit non-applicable items.
- Write in concise, coherent paragraphs. Use lists only for direct comparisons or sequential steps.
- Use plain, concrete words. Avoid jargon, filler, repetitive summaries, and unsolicited comparisons.
- Include only technical details essential for understanding conclusions, assessing risk, or reproducing results.

## Instruction Priority

- Respect system, platform, and safety constraints.
- Explicit user instructions override skills, memory, and default preferences.
- A project-level `AGENTS.md` supplements or overrides global rules only within that project's scope.

## Execution

- When starting work or fixing issues, drive autonomously until the objective is fully met.
- Before asking questions, finish all authorized work that yields reviewable progress. Seek approval on concrete results, not concepts.
- Challenge unhelpful suggestions directly; do not pander.
- Do not add warnings, disclaimers, approval steps, or compliance checklists for hypothetical risks.

## Parallel subagents

This section is an explicit instruction to spawn sub-agents.

When the user sends two or more tasks in one turn, or one request splits into two or more independent work items, and those items are not complex, dispatch them in parallel. Do not serialize independent work on the parent.

Worker by product:

- Cursor: Grok. Set the Task/subagent model to `grok-4.6` (or the current Grok slug). Do not inherit a non-Grok parent model.
- Codex: Terra. Spawn with `model` `gpt-5.6-terra` and `fork_turns` `"none"` or a bounded integer so the override applies. Do not use a full-history fork for these workers.
- ZCode: GLM 5.3 Flash. Always use the flash model for subagents/Task tool. Do not use the full GLM 5.3 model for child agents.

Give each child a closed assignment: goal, in-scope paths, done criteria, and what to return. Children must not expand scope.

The parent does not implement the leaf work. The parent splits, launches, inspects each child's evidence and diffs, rejects or re-dispatches failures, then integrates and answers the user.

Keep the work on the parent when there is only one bounded item the parent can finish faster than dispatch overhead, or when the work is complex: architecture, underspecified design, cross-cutting refactors, shared mutable state, or a single sequential chain of decisions.

## Testing & Verification

- Do not write tests for reversible, low-impact, or implementation-mirroring changes.
- Run tests proportionate to the changes. Once passed, expand or repeat testing only if new changes, failures, or ambiguities arise; otherwise, finish the task.
- Delete unused temporary files before completion.
