Accuracy metrics for most common few-shot datasets:

## Omniglot

| Model                 | 1-shot 5-way | 5-shot 5-way      | 1-shot 20-way     | 5-shot 20-way     |
| --------------------- | ------------ | ----------------- | ----------------- | ----------------- |
| MAML (as desc in [1]) | 98.7%        | 99.9%             | 95.8%             | 98.9%             |
| MAML (reproduced)     | 98.4%        | 99.4%<sup>1</sup> | 91.0%<sup>1</sup> | 94.4%<sup>1</sup> |

<br/> <br/>

## CIFAR-FS

| Model                 | 1-shot 2-way      | 5-shot 2-way      | 1-shot 5-way | 5-shot 5-way      |
| --------------------- | ----------------- | ----------------- | ------------ | ----------------- |
| MAML (as desc in [2]) | 82.8%             | 88.3%             | 58.9%        | 71.5%             |
| MAML (reproduced)     | 75.0%             | 87.4%             | 54.9%        | 70.1%             |

<br/> <br/>

## MiniImageNet


| Model             | 1-shot 2-way      | 5-shot 2-way       | 1-shot 5-way       | 5-shot 5-way        |
| ----------------- | ----------------- | ------------------ | ------------------ | ------------------- |
| MAML              | 74.9% [2]         | 84.4% [2]          | 48.70% [1]         | 63.11% [1]          |
| MAML (reproduced) |                   | 71.12%<sup>2</sup> | 47.12%<sup>2</sup> | 63.15% <sup>2</sup> |

<br/> <br/>

## References
1. Chelsea Finn, Pieter Abbeel, Sergey Levine. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." arXiv preprint arXiv:1703.03400 (2017).
2. Bertinetto, Luca, et al. "Meta-learning with differentiable closed-form solvers." arXiv preprint arXiv:1805.08136 (2018).

## Notes

<sup>1</sup> Batch sizes have been decreased to reduce GPU memory usage. See scripts for exact configuration  
<sup>2</sup> Preliminary results