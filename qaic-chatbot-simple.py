#!/bin/env python3

import os, shutil, sys, torch, numpy
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import qaic

print("="*60)
print("Simple Chatbot Demo for Qualcomm Cloud AI 100")
print("="*60 + "\n")

# Choose GPT-2 model for text generation
print("Loading GPT-2 model...")
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Define sequence length (must be fixed for ONNX export)
seq_length = 64

print("Exporting to ONNX...")

# Create dummy input for ONNX export
dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_length))
dummy_attention_mask = torch.ones((1, seq_length), dtype=torch.long)

# Export to ONNX with fixed batch size and sequence length
torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    'chatbot_simple.onnx',
    export_params=True,
    opset_version=14,
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={}  # No dynamic axes for simpler compilation
)

print("Compiling for AIC100...")

# Compile for AIC100
aic_binary_dir = 'aic-chatbot-simple-binary'

if os.path.exists(aic_binary_dir):
    shutil.rmtree(aic_binary_dir)

cmd = f'/opt/qti-aic/exec/qaic-exec -aic-hw -aic-hw-version=2.0 -compile-only -convert-to-fp16 \
-aic-num-cores=4 -m=chatbot_simple.onnx -aic-binary-dir={aic_binary_dir}'

ret = os.system(cmd)
if ret != 0:
    print(f"Compilation failed with error code {ret}")
    sys.exit(1)

print("\nInitializing AIC100 session...")

# Create the AIC100 session
try:
    chatbot_sess = qaic.Session(
        model_path=aic_binary_dir + '/programqpc.bin',
        num_activations=1
    )
    chatbot_sess.setup()
except Exception as e:
    print(f"Failed to initialize session: {e}")
    sys.exit(1)

# Get input/output shapes
input_ids_shape, input_ids_type = chatbot_sess.model_input_shape_dict['input_ids']
attention_mask_shape, attention_mask_type = chatbot_sess.model_input_shape_dict['attention_mask']
output_shape, output_type = chatbot_sess.model_output_shape_dict['logits']

print(f"Model ready!")
print(f"  Input shape: {input_ids_shape}")
print(f"  Output shape: {output_shape}")

print("\n" + "="*60)
print("Chatbot ready! (single-pass generation)")
print("Note: This demo predicts the next token for your input")
print("Type your message (or 'quit' to exit)")
print("="*60 + "\n")

def predict_next_tokens(prompt_text, num_predictions=5):
    """Predict the next few tokens given a prompt."""

    # Tokenize input
    inputs = tokenizer(
        prompt_text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=seq_length
    )

    input_ids = inputs['input_ids'].numpy().astype(input_ids_type)
    attention_mask = inputs['attention_mask'].numpy().astype(attention_mask_type)

    # Prepare inputs
    input_dict = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    # Run model on AIC100
    output = chatbot_sess.run(input_dict)

    # Get logits
    logits = numpy.frombuffer(output['logits'], dtype=output_type).reshape(output_shape)

    # Find the position of the last real token (before padding)
    actual_length = attention_mask.sum() - 1  # -1 because we want to predict from last token

    # Get logits for next token prediction
    next_token_logits = logits[0, actual_length, :]

    # Get top predictions
    top_indices = next_token_logits.argsort()[-num_predictions:][::-1]
    top_tokens = [tokenizer.decode([idx]) for idx in top_indices]
    top_probs = numpy.exp(next_token_logits[top_indices]) / numpy.exp(next_token_logits[top_indices]).sum()

    return list(zip(top_tokens, top_probs))

# Interactive loop
while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("Chatbot: Goodbye!")
        break

    if not user_input:
        continue

    # Get predictions
    predictions = predict_next_tokens(user_input, num_predictions=5)

    print("\nTop 5 predicted next tokens:")
    for i, (token, prob) in enumerate(predictions, 1):
        print(f"  {i}. '{token}' (probability: {prob:.4f})")

    # Simple completion using top prediction
    completion = user_input + predictions[0][0]
    print(f"\nSimple completion: {completion}\n")

print("\nChatbot session ended.")
