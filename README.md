
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



