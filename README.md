# Scene Classification with EfficientNet and ResNet

This repository implements a lightweight and flexible PyTorch-based image classification supporting custom training on small datasets using ResNet and EfficientNet.

## Project Structure

```
.
├── data/			# Image data organized by class folders
│   ├── train/           
│   └── test/           
├── model/
│   ├── efficientnet.py  	# Custom EfficientNet implementation
│   └── resnet.py        	# Custom ResNet implementation
├── saved_models/        	# Trained model parameters
│   ├── efficientnet.pth
│   ├── resnet18.pth
│   ├── resnet34.pth
│   └── resnet50.pth
├── utils/
│   ├── dataloader.py    	# Dataloader
│   ├── eval.py          	# Evaluation function (Top-K Accuracy)
│   ├── lr_finder.py     	# Learning rate finder
├── find_lr.py           	# Script to find the best learning rate
├── train.py			# Training
├── test.py			# Evaluation
├── main.py              	# Entry point for training/testing
└── environment.yaml		# Environment configuration
```

## Getting started

### 1. Environment Setup

```bash
conda env create -f environment.yaml
conda activate CNNcls-env
```

### 2. Prepare Data

Dataset should be organized as:

```
data/train/
  ├── class1/
  ├── class2/
  └── ...
```

### 3. Train the Model

Here is an example command that will train ResNet34:

```bash
python main.py \
	--phase train \
	--train_data_dir ./data/train/ \
	--model resnet34 \
	--lr 1e-3 \
	--epochs 15 \
	--dropout 0.2 \
```

Configs including: 

* phase: Choose to train or test the model. 
* train_data_dir: Path to training data directory.
* model: Which model architecture to use.
* lr: Learning rate.
* epochs: Number of training epochs.
* dropout: Dropout rate.

### 4. Test the Model

```bash
python main.py
	--phase test \
	--test_data_dir ./data/test/ \
	--model resnet34 \
```

You need to train the model first before running the evaluation.

## Model Options

| Model Name           | Description                    |
| -------------------- |--------------------------------|
| `efficientnet`       | Torch-based EfficientNet       |
| `resnet18/34/50/101` | Torch-based ResNet18/34/50/101 |