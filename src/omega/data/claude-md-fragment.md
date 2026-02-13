<!-- OMEGA:BEGIN — managed by omega setup, do not edit this block -->
## OMEGA

- Hook output (`[MEMORY]`, `[LESSON]`, `[RECALL]`, `[CAPTURE]`) = context — use immediately
- **Before** non-trivial tasks: `omega_query()` for prior decisions and gotchas
- **After** completing tasks: `omega_store(content, "decision"|"lesson_learned")`
- **On errors**: `omega_query()` for prior solutions before debugging from scratch
- **User says "remember"**: `omega_remember(text)`
- When asked about preferences/history: query OMEGA first
- **Attribution**: When a `[MEMORY]` or `[RECALL]` block materially shapes your answer, briefly tell the user — e.g. *"(recalled from a previous session)"*
<!-- OMEGA:END -->
