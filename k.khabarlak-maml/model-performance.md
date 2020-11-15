# Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks
Chelsea Finn, Pieter Abbeel, Sergey Levine. 2017

Accuracy metrics for most common few-shot datasets:

## Omniglot

| Model                          | 1-shot 5-way | 5-shot 5-way      | 1-shot 20-way     | 5-shot 20-way     |
| ------------------------------ | ------------ | ----------------- | ----------------- | ----------------- |
| MAML (as desc in <sup>1</sup>) | 98.7%        | 99.9%             | 95.8%             | 98.9%             |
| MAML (reproduced)              | 98.4%        | 99.4%<sup>2</sup> | 91.0%<sup>2</sup> |      <sup>2</sup> |
<br/> <br/>

## CIFAR-FS

| Model                          | 1-shot 2-way | 5-shot 2-way | 1-shot 5-way | 5-shot 5-way |
| ------------------------------ | ------------ | ------------ | ------------ | ------------ |
| MAML (as desc in <sup>3</sup>) | 82.8%        | 88.3%        | 58.9%        | 71.5%        |
| MAML (reproduced)              |              |              |              |              |
<br/> <br/>

## MiniImageNet


| Model             | 1-shot 2-way      | 5-shot 2-way      | 1-shot 5-way       | 5-shot 5-way       |
| ----------------- | ----------------- | ----------------- | ------------------ | ------------------ |
| MAML              | 74.9%<sup>3</sup> | 84.4%<sup>3</sup> | 48.70%<sup>1</sup> | 63.11%<sup>1</sup> |
| MAML (reproduced) |                   |                   |                    |                    |


<br/> <br/>

<sup>1</sup> Chelsea Finn, Pieter Abbeel, Sergey Levine. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." arXiv preprint arXiv:1703.03400 (2017).  
<sup>2</sup> Batch sizes have been decreased to reduce GPU memory usage. See scripts for exact configuration  
<sup>3</sup> Bertinetto, Luca, et al. "Meta-learning with differentiable closed-form solvers." arXiv preprint arXiv:1805.08136 (2018).