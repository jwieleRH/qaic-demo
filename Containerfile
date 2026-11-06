FROM ghcr.io/quic/cloud_ai_inference_ubuntu24:1.20.4.0

WORKDIR ./qaic-demo/

COPY ./qaic-demo.py .

CMD ./qaic-demo.py
