#!/bin/bash
# Simulated before/after OMEGA demo for README GIF
# Run: vhs assets/demo.tape

# ── Colors ───────────────────────────────────────────────────────
BOLD='\033[1m'
RESET='\033[0m'
WHITE='\033[1;37m'
GRAY='\033[0;37m'
BGREEN='\033[1;32m'
BPURPLE='\033[1;35m'
CYAN='\033[0;36m'
BCYAN='\033[1;36m'
DGRAY='\033[0;90m'
BRED='\033[1;31m'

# ── Helpers ──────────────────────────────────────────────────────
type_slow() {
    local text="$1" delay="${2:-0.04}"
    for ((i=0; i<${#text}; i++)); do
        printf '%s' "${text:$i:1}"
        sleep "$delay"
    done
}

pause() { sleep "${1:-1}"; }

# ══════════════════════════════════════════════════════════════════
# ACT 1: WITHOUT MEMORY
# ══════════════════════════════════════════════════════════════════
clear
printf "\n"
printf "  ${DGRAY}───────────────────────────────────────${RESET}\n"
printf "  ${WHITE}${BOLD}  Without memory${RESET}\n"
printf "  ${DGRAY}───────────────────────────────────────${RESET}\n"
printf "\n"
pause 1

printf "  ${WHITE}${BOLD}Session 1${RESET}  ${DGRAY}Monday 9am${RESET}\n\n"
pause 0.5

printf "  ${BGREEN}>${RESET} "
type_slow "Build failing: \"require() of ES Module not supported\"" 0.025
printf "\n\n"
pause 0.6

printf "  ${GRAY}CJS/ESM mismatch. ${WHITE}node-fetch@2${GRAY} is CommonJS${RESET}\n"
printf "  ${GRAY}but your project has ${WHITE}\"type\": \"module\"${GRAY}.${RESET}\n"
pause 0.3
printf "  ${GRAY}Fix:${RESET} ${CYAN}npm install node-fetch@3${RESET}\n\n"
pause 0.8

printf "  ${BGREEN}>${RESET} "
type_slow "That fixed it." 0.04
printf "\n"
pause 1.2

# Session 2 without memory
clear
printf "\n"
printf "  ${WHITE}${BOLD}Session 2${RESET}  ${DGRAY}Thursday 2pm${RESET}\n\n"
pause 0.5

printf "  ${BGREEN}>${RESET} "
type_slow "Same \"require() of ES Module\" error in analytics service" 0.025
printf "\n\n"
pause 0.6

printf "  ${GRAY}Let me investigate...${RESET}\n"
pause 0.5
printf "  ${GRAY}Checking ${WHITE}package.json${GRAY}... reading config...${RESET}\n"
pause 0.5
printf "  ${GRAY}Analyzing dependency tree...${RESET}\n"
pause 0.6
printf "  ${GRAY}Found it: CJS/ESM mismatch. ${WHITE}got@11${GRAY} loaded via require().${RESET}\n"
printf "  ${GRAY}Fix:${RESET} ${CYAN}npm install got@12${RESET}\n\n"
pause 0.4

printf "  ${BRED}  Same problem. Same investigation. From scratch.${RESET}\n"
pause 2.5

# ══════════════════════════════════════════════════════════════════
# ACT 2: WITH OMEGA
# ══════════════════════════════════════════════════════════════════
clear
printf "\n"
printf "  ${DGRAY}───────────────────────────────────────${RESET}\n"
printf "  ${WHITE}${BOLD}  With OMEGA${RESET}\n"
printf "  ${DGRAY}───────────────────────────────────────${RESET}\n"
printf "\n"
pause 1

printf "  ${WHITE}${BOLD}Session 1${RESET}  ${DGRAY}|${RESET}  ${BPURPLE}OMEGA${RESET} ${DGRAY}47 memories${RESET}\n\n"
pause 0.5

printf "  ${BGREEN}>${RESET} "
type_slow "Build failing: \"require() of ES Module not supported\"" 0.025
printf "\n\n"
pause 0.6

printf "  ${GRAY}CJS/ESM mismatch. ${WHITE}node-fetch@2${GRAY} is CommonJS${RESET}\n"
printf "  ${GRAY}but your project has ${WHITE}\"type\": \"module\"${GRAY}.${RESET}\n"
pause 0.3
printf "  ${GRAY}Fix:${RESET} ${CYAN}npm install node-fetch@3${RESET}\n\n"
pause 0.6

printf "  ${BGREEN}✓${RESET} ${BPURPLE}OMEGA${RESET} ${DGRAY}auto-captured:${RESET}\n"
printf "    ${BCYAN}[lesson]${RESET} ${DGRAY}CJS/ESM mismatch: upgrade to ESM-native version${RESET}\n"
pause 2

# Transition
clear
printf "\n\n\n"
printf "  ${DGRAY}              - - -${RESET}  ${WHITE}${BOLD}3 days later${RESET}  ${DGRAY}- - -${RESET}\n\n\n"
pause 1.8

# Session 2 with OMEGA
clear
printf "\n"
printf "  ${WHITE}${BOLD}Session 2${RESET}  ${DGRAY}|${RESET}  ${BPURPLE}OMEGA${RESET} ${DGRAY}53 memories${RESET}\n\n"
pause 0.5

printf "  ${BGREEN}>${RESET} "
type_slow "Same \"require() of ES Module\" in analytics service" 0.025
printf "\n\n"
pause 0.5

printf "  ${BPURPLE}● OMEGA recalled:${RESET}\n"
pause 0.2
printf "    ${BCYAN}[lesson]${RESET} ${GRAY}CJS/ESM mismatch: upgrade to ESM-native version.${RESET}\n"
printf "    ${DGRAY}Stored 3 days ago  |  confidence 0.94${RESET}\n\n"
pause 0.8

printf "  ${GRAY}Same pattern! Checking ${WHITE}package.json${GRAY}...${RESET}\n"
pause 0.3
printf "  ${GRAY}Found ${WHITE}got@11${GRAY}. Fix:${RESET} ${CYAN}npm install got@12${RESET}\n\n"
pause 0.4

printf "  ${BGREEN}  Instant recall. No re-investigation.${RESET}\n"
pause 3.5
