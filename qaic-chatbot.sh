#!/bin/bash

# Wrapper script to run chatbot demos using qaic-demo.sh
# Usage:
#   ./qaic-chatbot.sh [simple] [--build] [chatbot args...]
#
# Examples:
#   ./qaic-chatbot.sh                              # Run full chatbot
#   ./qaic-chatbot.sh simple                       # Run simple chatbot
#   ./qaic-chatbot.sh --build                      # Rebuild and run full chatbot
#   ./qaic-chatbot.sh --num-cores 8                # Run with 8 cores
#   ./qaic-chatbot.sh simple --build               # Rebuild and run simple chatbot

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
CHATBOT_SCRIPT="./qaic-chatbot.py"
DEMO_ARGS=()

for arg in "$@"; do
    if [ "$arg" == "simple" ]; then
        CHATBOT_SCRIPT="./qaic-chatbot-simple.py"
    else
        DEMO_ARGS+=("$arg")
    fi
done

# Call qaic-demo.sh with the chatbot script and any other arguments
exec "$SCRIPT_DIR/qaic-demo.sh" "${DEMO_ARGS[@]}" "$CHATBOT_SCRIPT"
