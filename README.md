# Lookahead Optimizer

Pytorch implementation of [Lookahead Optimizer](https://arxiv.org/pdf/1907.08610.pdf)

# Dependencies
  * PyTorch

# Usage

  from Lookahead import Lookahead <br/>
  optim = torch.optim.Adam(model.parameters(), lr=0.001 ) <br/>
  optimizer = Lookahead( optim, alpha= 0.6 , k = 10) <br/>
