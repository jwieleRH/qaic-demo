#!/bin/env python3

import os, shutil, sys, torch, numpy
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import qaic

# Choose GPT-2 model for text generation
print("Loading GPT-2 model...")
model_name = 'gpt2'  # Can also use 'gpt2-medium', 'gpt2-large', etc.
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()

# Set pad token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Define maximum sequence length
max_length = 128  # Adjust based on memory constraints

print("Exporting to ONNX...")

# Create dummy input for ONNX export
# GPT-2 expects input_ids and attention_mask
dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, max_length))
dummy_attention_mask = torch.ones((1, max_length), dtype=torch.long)

# Export to ONNX
torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    'chatbot_model.onnx',
    export_params=True,
    opset_version=14,  # GPT-2 needs opset 14 or higher
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'attention_mask': {0: 'batch_size', 1: 'sequence'},
        'logits': {0: 'batch_size', 1: 'sequence'}
    }
)

print("Compiling for AIC100...")

# Compile for AIC100
aic_binary_dir = 'aic-chatbot-binary'

if os.path.exists(aic_binary_dir):
    shutil.rmtree(aic_binary_dir)

cmd = f'/opt/qti-aic/exec/qaic-exec -aic-hw -aic-hw-version=2.0 -compile-only -convert-to-fp16 \
-aic-num-cores=4 -m=chatbot_model.onnx \
-onnx-define-symbol=batch_size,1 \
-onnx-define-symbol=sequence,{max_length} \
-aic-binary-dir={aic_binary_dir}'

ret = os.system(cmd)
if ret != 0:
    print(f"Compilation failed with error code {ret}")
    sys.exit(1)

print("\nInitializing AIC100 session...")

# Create the AIC100 session
chatbot_sess = qaic.Session(
    model_path=aic_binary_dir + '/programqpc.bin',
    num_activations=1
)

chatbot_sess.setup()  # Load the model to the device

print("\n" + "="*60)
print("Chatbot ready! Type your messages (or 'quit' to exit)")
print("="*60 + "\n")

def generate_response(prompt_text, max_new_tokens=50):
    """Generate a response to the given prompt."""

    # Tokenize input
    inputs = tokenizer(
        prompt_text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=max_length
    )

    input_ids = inputs['input_ids'].numpy().astype(numpy.int64)
    attention_mask = inputs['attention_mask'].numpy().astype(numpy.int64)

    # Get input shapes and types from session
    input_ids_shape, input_ids_type = chatbot_sess.model_input_shape_dict['input_ids']
    attention_mask_shape, attention_mask_type = chatbot_sess.model_input_shape_dict['attention_mask']

    # Prepare inputs
    input_dict = {
        'input_ids': input_ids.astype(input_ids_type),
        'attention_mask': attention_mask.astype(attention_mask_type)
    }

    # Get current sequence length (non-padded)
    actual_length = attention_mask.sum()

    # Generate tokens one at a time
    generated_ids = input_ids[0].copy()

    for _ in range(max_new_tokens):
        # Run model on AIC100
        output = chatbot_sess.run(input_dict)

        # Get logits
        output_shape, output_type = chatbot_sess.model_output_shape_dict['logits']
        logits = numpy.frombuffer(output['logits'], dtype=output_type).reshape(output_shape)

        # Get next token (greedy sampling - take highest probability)
        # Use the logits from the last actual token position
        next_token_logits = logits[0, actual_length - 1, :]
        next_token_id = next_token_logits.argmax()

        # Check for end of sequence
        if next_token_id == tokenizer.eos_token_id:
            break

        # Update generated sequence
        if actual_length < max_length:
            generated_ids[actual_length] = next_token_id
            input_dict['attention_mask'][0, actual_length] = 1
            input_dict['input_ids'][0, actual_length] = next_token_id
            actual_length += 1
        else:
            # Sequence is full
            break

    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[:actual_length], skip_special_tokens=True)
    return generated_text

# Interactive chatbot loop
conversation_history = ""

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("Chatbot: Goodbye!")
        break

    if not user_input:
        continue

    # Build prompt with conversation history
    if conversation_history:
        prompt = conversation_history + f"\nHuman: {user_input}\nAssistant:"
    else:
        prompt = f"Human: {user_input}\nAssistant:"

    # Generate response
    full_response = generate_response(prompt, max_new_tokens=50)

    # Extract just the assistant's response
    if "Assistant:" in full_response:
        bot_response = full_response.split("Assistant:")[-1].strip()
        # Stop at next "Human:" if present
        if "Human:" in bot_response:
            bot_response = bot_response.split("Human:")[0].strip()
    else:
        bot_response = full_response[len(prompt):].strip()

    print(f"Bot: {bot_response}\n")

    # Update conversation history (keep last few exchanges to fit in context)
    conversation_history = prompt + " " + bot_response
    # Trim history if getting too long
    if len(conversation_history) > max_length * 2:
        # Keep only the most recent part
        conversation_history = conversation_history[-max_length * 2:]

print("\nChatbot session ended.")
