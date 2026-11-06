#!/bin/bash

# Setup environment for chatbot demo
export PYTHONPATH=/opt/qti-aic/dev/lib/x86_64:$PYTHONPATH

# Check if running simple or full version
if [ "$1" == "simple" ]; then
    echo "Running simple chatbot demo..."
    python3 qaic-chatbot-simple.py
else
    echo "Running full chatbot demo with autoregressive generation..."
    python3 qaic-chatbot.py
fi
