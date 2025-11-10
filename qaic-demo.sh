#!/usr/bin/bash

python3 -m venv vit_env
source vit_env/bin/activate

pip3 install pip -U
pip3 install /opt/qti-aic/dev/lib/x86_64/qaic-0.0.1-py3-none-any.whl
pip3 install requests numpy Pillow onnx==1.16.0 transformers==4.46.3
pip3 install torch@https://download.pytorch.org/whl/cpu/torch-2.4.1%2Bcpu-cp310-cp310-linux_x86_64.whl

./qaic-demo.py

