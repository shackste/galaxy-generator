#!/bin/bash

## this script installs all required dependencies, assuming that you use the docker ufoym/deepo:pytorch

apt-get update && apt-get install -y python3-opencv

pip install ipython astropy photutils opencv-python wandb statmorph jupyter corner chamferdist
