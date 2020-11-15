# Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks
Chelsea Finn, Pieter Abbeel, Sergey Levine. 2017

## Omniglot
Accuracy

| Model                    | 1-shot 5-way | 5-shot 5-way      | 1-shot 20-way     | 5-shot 20-way     |
| ------------------------ | ------------ | ----------------- | ----------------- | ----------------- |
| MAML (paper<sup>1</sup>) | 98.7%        | 99.9%             | 95.8%             | 98.9%             |
| MAML (ours)              | 98.4%        | 99.4%<sup>2</sup> | 91.0%<sup>2</sup> |      <sup>2</sup> |
<br/> <br/>

## MiniImagenet

5-way accuracy  

| Model        | 1-shot | 5-shot |
| ------------ | ------ | ------ |
| MAML (paper) | 48.70% | 63.11% |


<br/> <br/>

<sup>1</sup> Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. Chelsea Finn, Pieter Abbeel, Sergey Levine  
<sup>2</sup> Batch sizes have been decreased to reduce GPU memory usage. See scripts for exact configuration  