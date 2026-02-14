#!/bin/bash
# Simulated OMEGA + Claude Code conversation for README demo GIF
# Run: vhs assets/demo.tape

# ── Colors ───────────────────────────────────────────────────────
BOLD='\033[1m'
RESET='\033[0m'
WHITE='\033[1;37m'
GRAY='\033[0;37m'
GREEN='\033[0;32m'
BGREEN='\033[1;32m'
PURPLE='\033[0;35m'
BPURPLE='\033[1;35m'
CYAN='\033[0;36m'
BCYAN='\033[1;36m'
DGRAY='\033[0;90m'

# ── Helpers ──────────────────────────────────────────────────────
type_slow() {
    local text="$1" delay="${2:-0.04}"
    for ((i=0; i<${#text}; i++)); do
        printf '%s' "${text:$i:1}"
        sleep "$delay"
    done
}

pause() { sleep "${1:-1}"; }

# ── Session 1 ───────────────────────────────────────────────────
clear
printf "\n"
printf "  ${WHITE}${BOLD}* Claude Code${RESET}  ${DGRAY}|${RESET}  ${PURPLE}OMEGA${RESET} ${DGRAY}47 memories${RESET}\n"
printf "  ${DGRAY}--------------------------------------${RESET}\n"
printf "\n"
pause 1.5

# User asks about the error
printf "  ${BGREEN}>${RESET} "
type_slow "Build failing: \"require() of ES Module not supported\"" 0.035
printf "\n\n"
pause 1.2

# Claude responds
printf "  ${GRAY}CJS/ESM mismatch — ${WHITE}node-fetch@2${GRAY} uses CommonJS${RESET}\n"
printf "  ${GRAY}but your project has ${WHITE}\"type\": \"module\"${GRAY}.${RESET}\n"
printf "\n"
pause 0.6
printf "  ${GRAY}Fix:${RESET}\n"
printf "  ${CYAN}  npm install node-fetch@3${RESET}\n"
printf "\n"
pause 2

# User confirms
printf "  ${BGREEN}>${RESET} "
type_slow "Fixed, thanks!" 0.045
printf "\n\n"
pause 0.8

# OMEGA auto-capture
printf "  ${BGREEN}✓${RESET} ${BPURPLE}OMEGA${RESET} ${DGRAY}auto-captured:${RESET}\n"
printf "    ${BCYAN}[lesson]${RESET} ${DGRAY}require() of ES Module — CJS/ESM mismatch${RESET}\n"
printf "    ${DGRAY}with node-fetch@2. Upgrade to v3 or use built-in fetch.${RESET}\n"
pause 3.5

# ── Transition ──────────────────────────────────────────────────
clear
printf "\n\n\n\n"
printf "  ${DGRAY}            - - -${RESET}  ${WHITE}${BOLD}3 days later${RESET}  ${DGRAY}- - -${RESET}\n"
printf "\n\n"
pause 2.5

# ── Session 2 ───────────────────────────────────────────────────
clear
printf "\n"
printf "  ${WHITE}${BOLD}* Claude Code${RESET}  ${DGRAY}|${RESET}  ${PURPLE}OMEGA${RESET} ${DGRAY}53 memories${RESET}\n"
printf "  ${DGRAY}--------------------------------------${RESET}\n"
printf "\n"
pause 1.5

# User types second question
printf "  ${BGREEN}>${RESET} "
type_slow "Same \"require() of ES Module\" in the analytics service" 0.035
printf "\n\n"
pause 1

# OMEGA surfaces the memory
printf "  ${BPURPLE}● OMEGA recalled:${RESET}\n"
pause 0.4
printf "    ${BCYAN}[lesson]${RESET} ${GRAY}require() of ES Module — CJS/ESM mismatch.${RESET}\n"
printf "    ${GRAY}Upgrade to ESM-native version.${RESET}\n"
printf "    ${DGRAY}Stored 3 days ago  |  accessed 1 time${RESET}\n"
printf "\n"
pause 2

# Claude responds using recalled context
printf "  ${GRAY}Same pattern. Checking ${WHITE}package.json${GRAY}...${RESET}\n"
pause 0.6
printf "  ${GRAY}Found it — ${WHITE}got@11${GRAY} is ESM-only but loaded via require().${RESET}\n"
printf "\n"
printf "  ${GRAY}Fix:${RESET}\n"
printf "  ${CYAN}  npm install got@12${RESET}\n"
printf "\n"
pause 4
