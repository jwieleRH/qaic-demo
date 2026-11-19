# Container Usage Guide

## Quick Start

Run the interactive demo selector:

```bash
./qaic-demo.sh
```

This will:
1. Build the container image
2. Show a menu to select which demo to run
3. Run the selected demo

## Demo Options

```
1) Image Classification (ViT)       - Original demo
2) Chatbot - Simple                 - Next token prediction
3) Chatbot - Full                   - Autoregressive text generation
```

## Manual Container Usage

### Build the Image

```bash
podman build --tag qaic-demo --group-add=qaic .
```

### Run Specific Demos

**Image Classification (default):**
```bash
podman run -it --device=/dev/accel/accel0 \
    --security-opt=label=disable --group-add=keep-groups \
    qaic-demo
```

**Simple Chatbot:**
```bash
podman run -it --device=/dev/accel/accel0 \
    --security-opt=label=disable --group-add=keep-groups \
    qaic-demo ./qaic-chatbot-simple.py
```

**Full Chatbot:**
```bash
podman run -it --device=/dev/accel/accel0 \
    --security-opt=label=disable --group-add=keep-groups \
    qaic-demo ./qaic-chatbot.py
```

## Container Structure

```
/qaic-demo/
├── qaic-demo.py              # Image classification
├── qaic-chatbot-simple.py    # Simple chatbot
├── qaic-chatbot.py           # Full chatbot
└── 000000039769.jpg          # Test image
```

## Files Not Included in Container

These files are for local development and documentation only:
- `*.sh` - Wrapper scripts
- `*.md` - Documentation
- `*.txt` - Summaries
- `vit_env/` - Virtual environment

## Notes

- The base image already includes all necessary Python packages
- `/dev/accel/accel0` must be accessible for AIC100 inference
- First run of each demo downloads models and compiles them
- Compiled models are cached in the container (not persisted)
