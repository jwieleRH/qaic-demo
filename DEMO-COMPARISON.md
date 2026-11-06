# Demo Comparison: Image Classification vs. Chatbot

This document compares the original image classification demo with the new chatbot demos to highlight the patterns and differences.

## Side-by-Side Code Comparison

### 1. Model Loading

**Image Demo (qaic-demo.py):**
```python
from transformers import ViTForImageClassification, ViTImageProcessor

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
```

**Chatbot Demo (qaic-chatbot-simple.py):**
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
```

**Pattern**: Both use HuggingFace Transformers, different model types.

---

### 2. ONNX Export

**Image Demo:**
```python
dummy_input = torch.randn(1, 3, 224, 224)  # Batch, channels, height, width

torch.onnx.export(model,
                  dummy_input,
                  'model.onnx',
                  export_params=True,
                  input_names=['input'],
                  output_names=['output'])
```

**Chatbot Demo:**
```python
dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_length))
dummy_attention_mask = torch.ones((1, seq_length), dtype=torch.long)

torch.onnx.export(model,
                  (dummy_input_ids, dummy_attention_mask),
                  'chatbot_simple.onnx',
                  export_params=True,
                  opset_version=14,
                  input_names=['input_ids', 'attention_mask'],
                  output_names=['logits'])
```

**Differences**:
- Image: Single input tensor (pixel values)
- Chatbot: Two input tensors (token IDs + attention mask)
- Chatbot needs higher opset version (14 vs. default)

---

### 3. Compilation

**Image Demo:**
```bash
/opt/qti-aic/exec/qaic-exec \
  -aic-hw \
  -aic-hw-version=2.0 \
  -compile-only \
  -convert-to-fp16 \
  -aic-num-cores=4 \
  -m=model.onnx \
  -onnx-define-symbol=batch_size,1 \
  -aic-binary-dir=aic-binary-dir
```

**Chatbot Demo:**
```bash
/opt/qti-aic/exec/qaic-exec \
  -aic-hw \
  -aic-hw-version=2.0 \
  -compile-only \
  -convert-to-fp16 \
  -aic-num-cores=4 \
  -m=chatbot_simple.onnx \
  -aic-binary-dir=aic-chatbot-simple-binary
```

**Pattern**: Nearly identical! Just different model file and output directory.

---

### 4. Session Setup

**Image Demo:**
```python
vit_sess = qaic.Session(
    model_path=aic_binary_dir + '/programqpc.bin',
    num_activations=1
)

vit_sess.setup()
```

**Chatbot Demo:**
```python
chatbot_sess = qaic.Session(
    model_path=aic_binary_dir + '/programqpc.bin',
    num_activations=1
)

chatbot_sess.setup()
```

**Pattern**: Exactly the same API!

---

### 5. Input Preparation

**Image Demo:**
```python
# Download and open image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = PIL.Image.open(requests.get(url, stream=True).raw)

# Process image
inputs = processor(images=image, return_tensors='pt')
input_shape, input_type = vit_sess.model_input_shape_dict['input']
input_data = inputs['pixel_values'].numpy().astype(input_type)
input_dict = {'input': input_data}
```

**Chatbot Demo:**
```python
# Get user text input
user_input = input("You: ").strip()

# Tokenize text
inputs = tokenizer(user_input,
                   return_tensors='pt',
                   padding='max_length',
                   truncation=True,
                   max_length=seq_length)

input_ids = inputs['input_ids'].numpy().astype(input_ids_type)
attention_mask = inputs['attention_mask'].numpy().astype(attention_mask_type)

input_dict = {
    'input_ids': input_ids,
    'attention_mask': attention_mask
}
```

**Differences**:
- Image: Single input from preprocessor
- Chatbot: Multiple inputs from tokenizer
- Both: Convert to numpy, cast to correct dtype, package in dict

---

### 6. Inference Execution

**Image Demo:**
```python
output = vit_sess.run(input_dict)
```

**Chatbot Demo:**
```python
output = chatbot_sess.run(input_dict)
```

**Pattern**: Identical API!

---

### 7. Output Processing

**Image Demo:**
```python
output_shape, output_type = vit_sess.model_output_shape_dict['output']
logits = numpy.frombuffer(output['output'], 
                          dtype=output_type).reshape(output_shape)
predicted_class_idx = logits.argmax(-1).item()
print('Predicted class:', model.config.id2label[predicted_class_idx])
```

**Chatbot Demo (Simple):**
```python
output_shape, output_type = chatbot_sess.model_output_shape_dict['logits']
logits = numpy.frombuffer(output['logits'],
                          dtype=output_type).reshape(output_shape)
next_token_logits = logits[0, actual_length, :]
next_token_id = next_token_logits.argmax()
predicted_token = tokenizer.decode([next_token_id])
```

**Similarities**:
- Both: Get output buffer, reshape, apply argmax
- Both: Decode result to human-readable form

**Differences**:
- Image: Single prediction (class label)
- Chatbot: Sequence prediction (token-by-token)

---

## Complexity Comparison

### Single-Pass Prediction

**Image Demo**: ✅ Simple
- Load image → Process → Infer → Get class
- One forward pass = one result

**Chatbot Simple**: ✅ Simple  
- Input text → Tokenize → Infer → Get next token
- One forward pass = one token

### Autoregressive Generation

**Chatbot Full**: ⚠️ Complex
- Input text → Tokenize → Loop:
  - Infer → Get token → Append to sequence → Repeat
- Multiple forward passes = complete response
- Requires state management and stopping conditions

---

## Data Types Comparison

| Aspect          | Image Demo              | Chatbot Demo            |
|-----------------|-------------------------|-------------------------|
| **Input Type**  | Float (pixel values)    | Int64 (token IDs)       |
| **Input Shape** | (1, 3, 224, 224)        | (1, 64) or (1, 128)     |
| **Output Type** | Float (logits)          | Float (logits)          |
| **Output Shape**| (1, 1000) classes       | (1, seq_len, 50257) vocab |
| **Task Type**   | Classification          | Sequence prediction     |

---

## Model Sizes Comparison

### Vision Transformer (ViT)
- **Model**: google/vit-base-patch16-224
- **Parameters**: ~86M
- **Download Size**: ~350MB
- **ONNX Size**: ~350MB
- **QPC Size**: ~200-300MB (FP16)

### GPT-2 Variants
| Model       | Parameters | Download | ONNX   | QPC (est.) |
|-------------|-----------|----------|--------|------------|
| gpt2        | 124M      | ~500MB   | ~500MB | ~250MB     |
| gpt2-medium | 355M      | ~1.4GB   | ~1.4GB | ~700MB     |
| gpt2-large  | 774M      | ~3.1GB   | ~3.1GB | ~1.5GB     |
| gpt2-xl     | 1.5B      | ~6.2GB   | ~6.2GB | ~3GB       |

---

## Performance Comparison

### Compilation Time
- **Image (ViT)**: ~2-3 minutes
- **Chatbot (GPT-2)**: ~2-5 minutes
- Both similar, depends on model size and complexity

### Inference Time (Single Forward Pass)
- **Image (ViT)**: ~5-10ms per image
- **Chatbot (GPT-2)**: ~1-5ms per token

### Total Latency
- **Image**: ~5-10ms (single-shot)
- **Chatbot Simple**: ~1-5ms (single token)
- **Chatbot Full**: ~50-250ms (50 tokens × 1-5ms)

---

## Use Case Patterns

### When to Use Image Demo Pattern
- Single-shot predictions
- Fixed input size
- Classification tasks
- Examples: image classification, object detection, segmentation

### When to Use Chatbot Simple Pattern
- Next-token prediction
- Sequence tagging
- Single-pass sequence models
- Examples: sentiment analysis, named entity recognition

### When to Use Chatbot Full Pattern
- Text generation
- Autoregressive models
- Multi-step inference
- Examples: chatbots, code generation, translation, summarization

---

## Common Patterns (Apply to Both)

1. **Model Selection**: Use HuggingFace Transformers
2. **ONNX Export**: `torch.onnx.export()` with dummy inputs
3. **Compilation**: `qaic-exec` with FP16 optimization
4. **Runtime**: `qaic.Session()` from Python SDK
5. **Input Prep**: Preprocess → numpy → correct dtype
6. **Inference**: `session.run(input_dict)`
7. **Output**: Decode logits to human-readable form

---

## Extending These Demos

### Image Demo Extensions
- Object detection: YOLOv5, DETR
- Semantic segmentation: SegFormer, Mask R-CNN
- Image-to-image: Super-resolution, style transfer

### Chatbot Demo Extensions
- Different models: T5, BART, BLOOM
- Instruct-tuning: Alpaca, Vicuna
- Multimodal: CLIP, BLIP (text + image)
- Longer context: LongFormer, BigBird

### General Extensions
- Batch inference (multiple inputs)
- Streaming inference (real-time)
- Multi-device inference (model parallelism)
- Quantization (INT8 for speed)

---

## Key Takeaways

1. **Same API, Different Models**: Both demos use identical runtime APIs
2. **ONNX is the Bridge**: Export any PyTorch model to ONNX, compile for AIC100
3. **Pattern is Consistent**: Load → Export → Compile → Infer → Decode
4. **Complexity Varies**: Simple models (single-pass) vs. complex (autoregressive)
5. **Performance Matters**: Choose right model size for your latency/accuracy needs

The chatbot demos prove that the pattern from `qaic-demo.py` generalizes to any transformer-based model, making it easy to deploy diverse ML workloads on AIC100!
