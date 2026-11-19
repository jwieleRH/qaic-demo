#!/usr/bin/env python3

import argparse, os, shutil, sys, torch, numpy
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import qaic

# Parse command-line arguments
parser = argparse.ArgumentParser(description='DistilGPT2 chatbot running on Qualcomm AIC100')
parser.add_argument('--num-cores', type=int,
                    default=int(os.environ.get('QAIC_NUM_CORES', '4')),
                    help='Number of AIC100 cores to use (default: 4 or QAIC_NUM_CORES env var)')
parser.add_argument('--max-length', type=int, default=128,
                    help='Maximum sequence length (default: 128)')
args = parser.parse_args()

# Choose DistilGPT2 for text generation
print("Loading DistilGPT2 model...")
model_name = 'distilgpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()

# Set pad token to EOS token
tokenizer.pad_token = tokenizer.eos_token

# Define maximum sequence length
max_length = args.max_length

print("Exporting to ONNX...")

# Disable KV caching for simpler ONNX export
model.config.use_cache = False

# Create dummy inputs for ONNX export
dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, max_length))
dummy_attention_mask = torch.ones((1, max_length), dtype=torch.long)

# Export to ONNX
torch.onnx.export(
    model,
    {
        'input_ids': dummy_input_ids,
        'attention_mask': dummy_attention_mask
    },
    'chatbot_model.onnx',
    export_params=True,
    opset_version=14,
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

num_cores = args.num_cores
print(f"Using {num_cores} AIC100 cores")
# Try without FP16 conversion to avoid numerical issues
cmd = f'/opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=2.0 \
-aic-num-cores={num_cores} -m=chatbot_model.onnx \
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
print("DistilGPT2 Chatbot ready! Ask me anything (or 'quit' to exit)")
print("Note: This model works best with specific prompts like:")
print("  'Q: What is AI? A:'")
print("  'The best thing about machine learning is'")
print("="*60 + "\n")

def generate_response(user_input, debug=False, max_new_tokens=50):
    """Generate a response using DistilGPT2.

    Args:
        user_input: User's question or prompt
        debug: If True, print debug information
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        Generated text (continuation of the prompt)
    """

    # Format the prompt to encourage Q&A style
    # Add "Q: ... A:" format to help the model understand it's a question
    if not user_input.strip().endswith(':'):
        prompt = f"Q: {user_input}\nA:"
    else:
        prompt = user_input

    # Encode the input
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=max_length - max_new_tokens  # Leave room for generation
    )

    # Get actual input length before padding
    input_length = inputs['input_ids'].shape[1]

    # Pad to max_length (right padding for GPT models during inference)
    padding_length = max_length - input_length
    if padding_length > 0:
        input_ids = torch.cat([
            inputs['input_ids'],
            torch.full((1, padding_length), tokenizer.pad_token_id, dtype=torch.long)
        ], dim=1)
        attention_mask = torch.cat([
            inputs['attention_mask'],
            torch.zeros((1, padding_length), dtype=torch.long)
        ], dim=1)
    else:
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

    input_ids_np = input_ids.numpy().astype(numpy.int64)
    attention_mask_np = attention_mask.numpy().astype(numpy.int64)

    if debug:
        print(f"[DEBUG] Prompt: {prompt}")
        print(f"[DEBUG] Input length: {input_length}, Padded to: {max_length}")

    # Get input shapes and types
    ids_shape, ids_type = chatbot_sess.model_input_shape_dict['input_ids']
    mask_shape, mask_type = chatbot_sess.model_input_shape_dict['attention_mask']

    # Prepare inputs
    input_dict = {
        'input_ids': input_ids_np.astype(ids_type),
        'attention_mask': attention_mask_np.astype(mask_type)
    }

    # Current sequence length
    current_length = input_length
    generated_ids = input_ids_np[0].copy()
    new_tokens = []

    # Generate tokens
    for step in range(max_new_tokens):
        # Run model
        output = chatbot_sess.run(input_dict)

        # Get logits
        output_shape, output_type = chatbot_sess.model_output_shape_dict['logits']
        logits = numpy.frombuffer(output['logits'], dtype=output_type).reshape(output_shape)

        # Get logits for the last generated position
        next_token_logits = logits[0, current_length - 1, :]

        # Apply temperature sampling for diversity
        temperature = 0.8
        next_token_logits = next_token_logits / temperature

        # Sample from distribution
        logits_exp = numpy.exp(next_token_logits - numpy.max(next_token_logits))
        probs = logits_exp / logits_exp.sum()
        next_token_id = int(numpy.random.choice(len(probs), p=probs))

        if debug and step < 5:
            top_indices = next_token_logits.argsort()[-3:][::-1]
            print(f"[DEBUG] Step {step}: token_id={next_token_id}, token='{tokenizer.decode([next_token_id])}'")
            print(f"[DEBUG]   Top 3: {[(int(idx), f'{next_token_logits[idx]:.2f}', tokenizer.decode([idx])) for idx in top_indices]}")

        # Stop if EOS or newline (for Q&A format)
        if next_token_id == tokenizer.eos_token_id:
            if debug:
                print(f"[DEBUG] EOS at step {step}")
            break

        # Add to generated sequence
        new_tokens.append(next_token_id)

        # Update for next iteration
        if current_length < max_length:
            generated_ids[current_length] = next_token_id
            input_dict['input_ids'][0, current_length] = next_token_id
            input_dict['attention_mask'][0, current_length] = 1
            current_length += 1
        else:
            break

        # Stop at newline for Q&A format
        decoded_new = tokenizer.decode(new_tokens, skip_special_tokens=True)
        if '\n' in decoded_new:
            if debug:
                print(f"[DEBUG] Newline detected at step {step}")
            # Trim at newline
            new_tokens = tokenizer.encode(decoded_new.split('\n')[0], add_special_tokens=False)
            break

    if debug:
        print(f"[DEBUG] Generated {len(new_tokens)} new tokens")

    # Decode only the new tokens
    if new_tokens:
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    else:
        response = "[No response generated]"

    return response

# Interactive chatbot loop
# Flan-T5 handles each turn independently (no complex state management)

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("Bot: Goodbye!")
        break

    if not user_input:
        continue

    # Generate response (enable debug for first message)
    try:
        # Enable debug mode if user types 'debug'
        debug_mode = user_input.lower().startswith('debug:')
        if debug_mode:
            user_input = user_input[6:].strip()

        bot_response = generate_response(user_input, debug=debug_mode)
        print(f"Bot: {bot_response}\n")
    except Exception as e:
        print(f"Error generating response: {e}\n")
        import traceback
        traceback.print_exc()

print("\nChatbot session ended.")
