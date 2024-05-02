#! /bin/bash

docker run -it \
 --gpus all \
 --ipc=host --ulimit memlock=-1 --ulimit stack=60900663296 \
 --rm \
 --mount type=bind,source="$(pwd)"/,target=/home/ \
 -p 9000:9000 \
 vam-dub
