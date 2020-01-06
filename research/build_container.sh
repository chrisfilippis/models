#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

git pull

docker stop cfil_models

docker rm cfil_models

docker build --tag=cfil_models:latest .

docker run -it --name cfil_models --gpus all cfil_models:latest bash

# echo "copying init model file"
# docker cp ../../deeplabv3_pascal_train_aug_2018_01_04.tar.gz cfil_models:/var/project/deeplab/datasets/tooth/init_models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz
# docker cp ../../deeplabv3_pascal_train_aug cfil_models:/var/project/deeplab/datasets/tooth/init_models/deeplabv3_pascal_train_aug

# docker run -it --name cfil_models cfil_models:latest bash