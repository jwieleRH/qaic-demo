#!/usr/bin/bash

podman build --tag qaic-demo .

podman run -it --device=/dev/accel/accel0 --security-opt=label=disable qaic-demo
