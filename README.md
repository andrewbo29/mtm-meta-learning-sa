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
To reproduce the results on benchmarks described in our article, use the following scripts. To vary types of the experiments, change the parameters of the scripts responsible for benchmark dataset, shot and way (e.g. miniImageNet 1-shot 5-way or CIFAR-FS 5-shot 2-way).

### MAML
Multi-task modification (MTM) for Model-Agnostic Meta-Learning (MAML) ([Finn et al., 2017](https://arxiv.org/abs/1703.03400)).

As we have described in the paper, MAML MTM-SPSA is trained on top of the reproduced models. First, we define how to get reproduced models, then how we run MTM SPSA, finally how to run tests. We define run examples for all 4 datasets used.

**miniImageNet (reproduced 1-shot 2-way):**
```
python maml/train.py ./datasets/ \
    --run-name reproduced-miniimagenet \
    --dataset miniimagenet \
    --num-ways 2 \
    --num-shots 1 \
    --num-steps 5 \
    --step-size 0.01 \
    --hidden-size 32 \
    --batch-size 4 \
    --num-workers 4 \
    --num-epochs 300 \
    --use-cuda \
    --output-folder ./results
```

**miniImageNet (MTM SPSA-Track 1-shot 2-way):**
```
python maml/train.py ./datasets/ \
    --run-name mini-imagenet-mtm-spsa-track \
    --load "./results/reproduced-miniimagenet/model.th" \
    --dataset miniimagenet \
    --num-ways 2 \
    --num-shots 1 \
    --num-steps 5 \
    --step-size 0.01 \
    --task-weighting spsa-track \
    --normalize-spsa-weights-after 100 \
    --hidden-size 32 \
    --batch-size 4 \
    --num-workers 4 \
    --num-epochs 40 \
    --use-cuda \
    --output-folder ./results
```

**tieredImageNet (reproduced 1-shot 2-way):**
```
python maml/train.py ./datasets/ \
    --run-name reproduced-tieredimagenet \
    --dataset tieredimagenet \
    --num-ways 2 \
    --num-shots 1 \
    --num-steps 5 \
    --step-size 0.01 \
    --hidden-size 32 \
    --batch-size 4 \
    --num-workers 4 \
    --num-epochs 300 \
    --use-cuda \
    --output-folder ./results
```

**tieredImageNet (MTM SPSA 1-shot 2-way):**
```
python maml/train.py ./datasets/ \
    --run-name tiered-imagenet-mtm-spsa \
    --load "./results/reproduced-tieredimagenet/model.th" \
    --dataset tieredimagenet \
    --num-ways 2 \
    --num-shots 1 \
    --num-steps 5 \
    --step-size 0.01 \
    --task-weighting spsa-delta \
    --normalize-spsa-weights-after 100 \
    --hidden-size 32 \
    --batch-size 4 \
    --num-workers 4 \
    --num-epochs 40 \
    --use-cuda \
    --output-folder ./results
```

**FC100 (reproduced 5-shot 5-way):**
```
python maml/train.py ./datasets/ \
    --run-name reproduced-fc100 \
    --dataset fc100 \
    --num-ways 5 \
    --num-shots 5 \
    --num-steps 5 \
    --step-size 0.01 \
    --hidden-size 32 \
    --batch-size 4 \
    --num-workers 4 \
    --num-epochs 300 \
    --use-cuda \
    --output-folder ./results
```

**FC100 (MTM SPSA-Coarse 5-shot 5-way):**
```
python maml/train.py ./datasets/ \
    --run-name fc100-mtm-spsa-coarse \
    --load "./results/reproduced-fc100/model.th" \
    --dataset fc100 \
    --num-ways 5 \
    --num-shots 5 \
    --num-steps 5 \
    --step-size 0.01 \
    --task-weighting spsa-per-coarse-class \
    --hidden-size 32 \
    --batch-size 4 \
    --num-workers 4 \
    --num-epochs 40 \
    --use-cuda \
    --output-folder ./results
```

**CIFAR-FS (reproduced 1-shot 5-way):**
```
python maml/train.py ./datasets/ \
    --run-name reproduced-cifar \
    --dataset cifarfs \
    --num-ways 5 \
    --num-shots 1 \
    --num-steps 5 \
    --step-size 0.01 \
    --hidden-size 32 \
    --batch-size 4 \
    --num-workers 4 \
    --num-epochs 600 \
    --use-cuda \
    --output-folder ./results
```

**CIFAR-FS (MTM Inner First-Order 1-shot 5-way):**
```
python maml/train.py ./datasets/ \
    --run-name cifar-mtm-inner-first-order \
    --load "./results/reproduced-cifar/model.th" \
    --dataset cifarfs \
    --num-ways 5 \
    --num-shots 1 \
    --num-steps 5 \
    --step-size 0.01 \
    --task-weighting gradient-novel-loss \
    --use-inner-optimizer \
    --hidden-size 32 \
    --batch-size 4 \
    --num-workers 4 \
    --num-epochs 40 \
    --use-cuda \
    --output-folder ./results
```

**CIFAR-FS (MTM Backprop 5-shot 5-way):**
```
python maml/train.py ./datasets/ \
    --run-name cifar-mtm-backprop \
    --load "./results/reproduced-cifar-5shot-5way/model.th" \
    --dataset cifarfs \
    --num-ways 5 \
    --num-shots 5 \
    --num-steps 5 \
    --step-size 0.01 \
    --task-weighting gradient-novel-loss \
    --hidden-size 32 \
    --batch-size 4 \
    --num-workers 4 \
    --num-epochs 40 \
    --use-cuda \
    --output-folder ./results
```

**To test** any of the above-described runs execute:
```
python maml/test.py ./results/path-to-config/config.json --num-steps 10 --use-cuda
```

For instance, **to test miniImageNet (MTM SPSA-Track)** run the following:
```
python maml/test.py ./results/mini-imagenet-mtm-spsa-track/config.json --num-steps 10 --use-cuda
```


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

Code in maml folder is based on the extended implementation from [**Torchmeta**](https://github.com/tristandeleu/pytorch-meta) and [**pytorch-maml**](https://github.com/tristandeleu/pytorch-maml). The code has been updated so that baseline scores more closely follow those of the original MAML paper, additional dataset support has been introduced. We follow the MIT license, which is available in the MAML folder.
