# Evaluation of the BigGAN model trained for Galaxy Zoo image generation

This package contains Pytorch implementation of the evaluation pipeline for the BigGAN model for Galaxy Zoo dataset

## Installation
To install conda environment for the evaluation pipeline run
```bash
conda env create -f environment_evaluation.yaml
```

## Dataset
We use Galaxy Zoo dataset from [Kaggle competition](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge).

## Evaluation

1. Download pretrained SimCLR encoder from [link](https://drive.google.com/file/d/1lOXiTBcbI3AnoNiFmrk_1keQVKqbAwjB/view?usp=sharing)
2. Download pretrained classifier from [link](https://drive.google.com/file/d/1B9SMUFFldvDEgHrUQVmFTPSxuiRZ3sfk/view?usp=sharing)
3. Fill the configuration file. Example is provided in `configs/biggan_eval.yaml`
4. Run
```bash
python python_modules/evaluation/main.py -c <path to config file>
```
