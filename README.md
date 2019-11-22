
Implementation of Lookahead optimizer and RAdam.

# Lookahead Optimizer

Pytorch implementation of [Lookahead Optimizer](https://arxiv.org/pdf/1907.08610.pdf)

![Lookahead Algorithm](./fig/lookahead.png)


# RAdam Optimizer

[RAdam optimizer](https://arxiv.org/pdf/1908.03265.pdf)

![RAdam Algorithm](./fig/RAdam.png)

# Dependencies
  * PyTorch

# Usage

  Lookahead
  
  ```python
  from Lookahead import Lookahead
  optim = torch.optim.Adam(model.parameters(), lr=0.001 )
  optimizer = Lookahead( optim, alpha= 0.6 , k = 10)
  ```

  RAdam

  ```python
  from RAdam import RAdam
  optim = RAdam(model.parameters(), lr=1e-3, weight_decay=1e-4)
  ```
  
 # Results
 
   - Reported results of run on CIFAR-10 with 3 different seed.
   - Used architecture ResNet-18 and trained for 90 epochs.
   - Used lr schedule to divide learning rate by 5 after every 30 epochs.
  
  | Seed / Optimizer |  SGD  | Adamw | Lookahead with SGD | RAdam  | Lookahead with Radam |
  | :----------------|:-----:|:-----:|:------------------:|:------:|---------------------:|
  |         42       | 93.31 | 92.78 |         93.34      |  93.01 |         93.21        |
  |         17       | 93.30 | 92.77 |         93.36      |  93.02 |         93.20        |
  |         11       | 93.33 | 92.78 |         93.40      |  93.03 |         93.14        |



