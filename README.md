# Multi-Task Meta-Learning Modification with Stochastic Approximation

This repository contains the code for the paper<br/> 
"Multi-Task Meta-Learning Modification with Stochastic Approximation".

![Method pipeline](https://github.com/andrewbo29/mtm-meta-learning-sa/blob/master/mtm_pipeline.png)

## Dependencies
This code has been tested on Ubuntu 16.04 with Python 3.8 and PyTorch 1.8.

To install the required dependencies:
```
pip install -r requirements.txt
```

## Usage
To reproduce the results on benchmarks described in our article, use the following scripts. To vary types of the experiments, change the parameters of the scripts responsible for shot and way (e.g. 1-shot 5-way or 5-shot 2-way).

### MAML
Multi-task modification (MTM) for Model-Agnostic Meta-Learning (MAML) ([Finn et al., 2017](https://arxiv.org/abs/1703.03400)).

### Prototypical Networks
Multi-task modification (MTM) for Prototypical Networks (ProtoNet) ([Snell et al., 2017](https://arxiv.org/abs/1703.05175)).

To train ProtoNet MTM SPSA-Track with ResNet-12 backbone on miniImageNet 1-shot 5-way benchmark:
```
python protonet/train.py --network ResNet12 --tracking --train-shot 1 --train-way 5 --val-shot 1 --val-way 5
```
To test ProtoNet MTM SPSA-Track with ResNet-12 backbone on miniImageNet 1-shot 5-way benchmark:
```
python protonet/test.py --network ResNet12 --shot 1 --way 5
```


## Acknowledgments

Our code uses some dataloaders and models implementations from [**Torchmeta**](https://github.com/tristandeleu/pytorch-meta).

Code in protonet folder is based on the implementation of [**MetaOptNet**](https://github.com/kjunelee/MetaOptNet). All .py files in this folder except for dataloaders.py and optimize.py were adopted from this implementation and modified afterwards. A copy of Apache License, Version 2.0 is available in protonet folder.

Code in maml folder is based on the implementation from [**Torchmeta**](https://github.com/tristandeleu/pytorch-meta).
