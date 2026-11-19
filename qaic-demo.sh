#!/usr/bin/bash

# Parse options
BUILD=0
DEMO_CMD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD=1
            shift
            ;;
        *)
            DEMO_CMD=$1
            shift
            ;;
    esac
done

# Set default demo command if not specified
if [ -z "$DEMO_CMD" ]; then
    DEMO_CMD="./qaic-demo.py"
fi

# Build the container image if requested or if it doesn't exist
if [ $BUILD -eq 1 ]; then
    echo "Building container image..."
    podman build --tag qaic-demo --group-add=qaic .
elif ! podman image exists qaic-demo; then
    echo "Container image not found. Building..."
    podman build --tag qaic-demo --group-add=qaic .
else
    echo "Using existing qaic-demo container image (use --build to rebuild)"
fi

echo ""
echo "Starting demo..."
podman run -it \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --device=/dev/accel/accel0 \
  --security-opt=label=disable \
  --group-add=keep-groups \
  qaic-demo $DEMO_CMD
