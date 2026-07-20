# docs/agents — progressive disclosure for AI agents

This folder is the **level-2** layer of a three-level progressive-disclosure
structure (the same pattern as Anthropic's Agent Skills: load a little context by
default, load detail only when the task needs it):

- **Level 1** — `/CLAUDE.md` (repo root): ~40 lines, always in context. What the repo
  is, how to run it, hard rules, and a routing table into this folder.
- **Level 2** — these topic files: read the ONE file matching your task. Each is
  self-contained and states which source files it summarizes.
- **Level 3** — source code, tests, and `docs/superpowers/` (spec + plans): the
  ground truth. Level-2 files cite exact paths so you can drop down when needed.

Rules for maintaining this layer:
- Keep level 1 under ~50 lines; push detail down, never up.
- A level-2 file summarizes; it must not contradict level 3. When code changes,
  update the matching level-2 file in the same commit.
- Human-facing teaching material lives in `docs/tutorial/` (Chinese), not here.
