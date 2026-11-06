#!/usr/bin/bash

podman build --tag qaic-demo --group-add=qaic .

podman run -it --device=/dev/accel/accel0 --security-opt=label=disable --group-add=keep-groups qaic-demo
