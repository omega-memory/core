<!-- OMEGA:BEGIN — managed by omega setup, do not edit this block -->
## Memory (OMEGA)

You have OMEGA persistent memory. At session start:
1. Call `omega_welcome()` for context briefing
2. Call `omega_protocol()` for your operating instructions — it's your coordination playbook
3. Follow the protocol it returns

Quick reference (protocol has full details):
- `[MEMORY]`/`[HANDOFF]`/`[COORD]` blocks from hooks = ground truth
- Before non-trivial tasks: `omega_query()` for prior context
- After completing tasks: `omega_store(content, "decision")` for key outcomes
- User says "remember": `omega_store(text, "user_preference")`
- Context getting full: `omega_checkpoint` to save state

If OMEGA is unavailable, use basic coordination:
- Before state changes: check `git log` and ask before deploying
- After tasks: store decisions with `omega_store()`
<!-- OMEGA:END -->
