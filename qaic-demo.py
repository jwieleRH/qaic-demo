#!/bin/env python3

import os, shutil, sys, requests, torch, numpy, PIL
from transformers import ViTForImageClassification, ViTImageProcessor
import qaic


# Choose the Vision Transformers model for classifying images and its image input preprocessor
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Convert to onnx
dummy_input = torch.randn(1, 3, 224, 224)       # Batch, channels, height, width

torch.onnx.export(model,                        # PyTorch model
             dummy_input,               # Input tensor
             'model.onnx',              # Output file
             export_params = True,      # Export the model parameters
             input_names   = ['input'], # Input tensor names
             output_names  = ['output'] # Output tensor names
             )

#compile
aic_binary_dir = 'aic-binary-dir'

if os.path.exists(aic_binary_dir):
    shutil.rmtree(aic_binary_dir)

cmd = '/opt/qti-aic/exec/qaic-exec -aic-hw -aic-hw-version=2.0 -compile-only -convert-to-fp16 \
-aic-num-cores=4 -m=model.onnx -onnx-define-symbol=batch_size,1 -aic-binary-dir=' + aic_binary_dir
os.system(cmd)

# Get input
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = PIL.Image.open(requests.get(url, stream=True).raw)

# Run the model
# Create the AIC100 session and prepare inputs and outputs

vit_sess = qaic.Session(model_path= aic_binary_dir+'/programqpc.bin',
num_activations=1)

inputs = processor(images=image, return_tensors='pt')
input_shape, input_type = vit_sess.model_input_shape_dict['input']
input_data = inputs['pixel_values'].numpy().astype(input_type)
input_dict = {'input': input_data}

output_shape, output_type = vit_sess.model_output_shape_dict['output']

# Run model on AIC100

vit_sess.setup() # Load the model to the device.
output = vit_sess.run(input_dict) # Execute on AIC100 now.

# Obtain the prediction by finding the highest probability among all classes

logits = numpy.frombuffer(output['output'], dtype=output_type).reshape(output_shape)
predicted_class_idx = logits.argmax(-1).item()
print('Predicted class:', model.config.id2label[predicted_class_idx])

