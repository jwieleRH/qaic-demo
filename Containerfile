FROM ghcr.io/qualcomm/cloud_ai_inference_ubuntu26:1.23.2.0

WORKDIR /qaic-demo

# Copy demo scripts
COPY qaic-demo.py .
COPY qaic-chatbot.py .
COPY qaic-chatbot-simple.py .

# Copy test image for image classification demo
COPY 000000039769.jpg .

# Default to image classification demo
# Use CMD instead of ENTRYPOINT so it can be easily overridden
CMD ["./qaic-demo.py"]
