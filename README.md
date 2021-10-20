# Multi-Task Meta-Learning Modification with Stochastic Approximation

This repository contains the code for the paper 
"Multi-Task Meta-Learning Modification with Stochastic Approximation"

![Method pipeline](https://github.com/andrewbo29/mtm-meta-learning-sa/blob/master/mtm_pipeline.png)

## Dependencies
This code has been tested on Ubuntu 16.04 with Python 3.8 and PyTorch 1.7.

To install the required dependencies:
```
pip install -r requirements.txt
```

## Usage
### MAML
### Prototypical Networks
To train with ResNet-12 backbone on miniImageNet 1-shot 5-way benchmark:
```
python protonet/train.py --network ResNet12 --tracking --train-shot 1 --val-shot 1
```
To test with ResNet-12 backbone on miniImageNet 1-shot 5-way benchmark:
```
python protonet/test.py --network ResNet12 --shot 1
```


## Acknowledgments

Code in protonet folder is based on the implementation of [**MetaOptNet**](https://github.com/kjunelee/MetaOptNet). All .py files in this folder except for dataloaders.py and optimize.py were adopted from this implementation and modified afterwards. A copy of Apache License, Version 2.0 is available in protonet folder.

Code in maml folder is based on the implementation [**Torchmeta**](https://github.com/tristandeleu/pytorch-meta).
