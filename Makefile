DEMO_CMD = ./qaic-demo.py

build:
	podman build --tag qaic-demo --group-add=qaic .

run:
	podman run -it --device=/dev/accel/accel0 --security-opt=label=disable --group-add=keep-groups qaic-demo $(DEMO_CMD)
