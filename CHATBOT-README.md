# QAIC Chatbot Demo

This directory contains chatbot demos for the Qualcomm Cloud AI 100 accelerator, following the same pattern as the image classification demo.

## Files

- **qaic-chatbot-simple.py**: Simple single-pass prediction (easier to run, good for testing)
- **qaic-chatbot.py**: Full autoregressive text generation (more complex, true chatbot)
- **qaic-chatbot.sh**: Wrapper script to run either version

## Prerequisites

1. Qualcomm Cloud AI 100 SDK installed at `/opt/qti-aic`
2. Python packages:
   ```bash
   pip install transformers torch numpy qaic
   ```
3. AIC100 device accessible via `/dev/accel/accel*`

## Quick Start

### Simple Version (Recommended First)

The simple version predicts the next token for your input - good for testing the compilation and runtime:

```bash
./qaic-chatbot.sh simple
```

This will:
1. Download GPT-2 model from HuggingFace
2. Export to ONNX (chatbot_simple.onnx)
3. Compile to AIC100 binary (aic-chatbot-simple-binary/)
4. Run inference on AIC100

**Example interaction:**
```
You: Hello, how are
Top 5 predicted next tokens:
  1. ' you' (probability: 0.8234)
  2. ' we' (probability: 0.0456)
  3. ' things' (probability: 0.0234)
  ...
Simple completion: Hello, how are you
```

### Full Version (Autoregressive Generation)

The full version generates complete responses token-by-token:

```bash
./qaic-chatbot.sh
```

This implements proper autoregressive generation, generating up to 50 new tokens per response.

**Example interaction:**
```
You: What is artificial intelligence?
Bot: Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems.
```

## How It Works

Both demos follow the same pattern as `qaic-demo.py`:

```
1. Load Model (GPT-2)
   ↓
2. Export to ONNX
   ↓
3. Compile with qaic-exec
   ↓
4. Run on AIC100 via qaic.Session
```

### Simple Version

- Uses **fixed-size** input (64 tokens)
- Single forward pass through the model
- Returns logits for next token prediction
- Fast and deterministic

### Full Version

- Uses **variable-size** input (up to 128 tokens)
- Iterative generation (max 50 new tokens)
- Maintains conversation history
- More realistic chatbot behavior

## Compilation Details

### qaic-exec Command

```bash
/opt/qti-aic/exec/qaic-exec \
    -aic-hw \
    -aic-hw-version=2.0 \
    -compile-only \
    -convert-to-fp16 \
    -aic-num-cores=4 \
    -m=chatbot_model.onnx \
    -aic-binary-dir=aic-chatbot-binary
```

**Flags:**
- `-aic-hw`: Target AIC100 hardware
- `-aic-hw-version=2.0`: AIC100 Pro hardware version
- `-compile-only`: Don't run inference during compilation
- `-convert-to-fp16`: Use FP16 for better performance
- `-aic-num-cores=4`: Use 4 NSP cores
- `-onnx-define-symbol`: Define symbolic dimensions (full version only)

## Runtime Details

Both versions use the `qaic` Python SDK:

```python
import qaic

# Create session from compiled QPC
chatbot_sess = qaic.Session(
    model_path='aic-chatbot-binary/programqpc.bin',
    num_activations=1
)

# Load model to device
chatbot_sess.setup()

# Run inference
output = chatbot_sess.run(input_dict)
```

## Comparison to Image Classification Demo

| Aspect              | Image Demo (qaic-demo.py)    | Chatbot Demo                |
|---------------------|------------------------------|------------------------------|
| **Model**           | ViT (Vision Transformer)     | GPT-2 (Language Model)      |
| **Input**           | Image (3×224×224)            | Token IDs (1×64 or 1×128)   |
| **Output**          | Class logits (1000 classes)  | Token logits (vocab_size)   |
| **Task**            | Single-shot classification   | Sequential generation       |
| **Complexity**      | Simple forward pass          | Autoregressive loop         |

## Troubleshooting

### Compilation Errors

If `qaic-exec` fails:
1. Check ONNX export: `python -c "import onnx; onnx.checker.check_model('chatbot_model.onnx')"`
2. Verify ONNX opset version (should be 14+)
3. Check AIC100 device availability: `ls /dev/accel/`

### Runtime Errors

If inference fails:
1. Verify session creation: check `programqpc.bin` exists
2. Check input shapes match compiled model
3. Ensure correct dtype for inputs (usually int64 or float16)

### Out of Memory

If compilation runs out of memory:
- Reduce sequence length (e.g., 64 → 32)
- Reduce number of cores (e.g., 4 → 2)
- Use smaller model (gpt2 is smallest)

## Model Variants

You can try different GPT-2 variants by changing `model_name`:

```python
model_name = 'gpt2'         # 124M params (fastest)
model_name = 'gpt2-medium'  # 355M params
model_name = 'gpt2-large'   # 774M params
model_name = 'gpt2-xl'      # 1.5B params (slowest)
```

Larger models provide better quality but require more memory and longer compilation times.

## Advanced Usage

### Custom Prompts

In `qaic-chatbot.py`, you can modify the prompt format:

```python
# Instruction-following format
prompt = f"Instruction: {user_input}\nResponse:"

# Q&A format  
prompt = f"Q: {user_input}\nA:"

# Chat format (current)
prompt = f"Human: {user_input}\nAssistant:"
```

### Temperature Sampling

For more diverse outputs, implement temperature sampling instead of greedy:

```python
# Current: greedy (take argmax)
next_token_id = next_token_logits.argmax()

# With temperature
temperature = 0.7
probs = numpy.exp(next_token_logits / temperature)
probs = probs / probs.sum()
next_token_id = numpy.random.choice(len(probs), p=probs)
```

### Batch Inference

Compile with larger batch size for throughput:

```bash
-onnx-define-symbol=batch_size,8
```

Then process multiple prompts simultaneously.

## Performance Notes

**Expected Performance (GPT-2 base, FP16, 4 NSPs):**
- Compilation time: ~2-5 minutes
- Single token generation: ~1-5ms
- Total response (50 tokens): ~50-250ms

**Optimization Tips:**
- Use FP16: `-convert-to-fp16` (faster, lower memory)
- More NSP cores: `-aic-num-cores=8` or `=14` (parallel execution)
- Shorter sequences: Smaller `max_length` (less computation)

## References

- Original image demo: `qaic-demo.py`
- QAic API documentation: `/opt/qti-aic/examples/python/README`
- Transformers library: https://huggingface.co/docs/transformers
- GPT-2 model card: https://huggingface.co/gpt2

## License

Same as original qaic-demo - refer to parent directory LICENSE if applicable.
