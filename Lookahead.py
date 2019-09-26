import torch
from torch.optim import Optimizer
from collections import defaultdict
import itertools as it


class Lookahead(Optimizer):
    r'''Implements Lookahead optimizer.

    It's been proposed in paper: Lookahead Optimizer: k steps forward, 1 step back
    (https://arxiv.org/pdf/1907.08610.pdf)

    Args:
        optimizer: The optimizer object used in inner loop for fast weight updates.
        alpha:     The learning rate for slow weight update.
                   Default: 0.5
        k:         Number of iterations of fast weights updates before updating slow
                   weights.
                   Default: 5

    Example:
        > optim = Lookahead(optimizer)
        > optim = Lookahead(optimizer, alpha=0.6, k=10)
    '''
    def __init__(self, optimizer, alpha=0.5, k=5):
        assert(0.0 <= alpha <= 1.0)
        assert(k >= 1)
        self.optimizer = optimizer
        self.alpha = alpha
        self.k = k
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        for group in self.param_groups:
            group['k_counter'] = 0
        self.slow_weights = [[param.clone().detach() for param in group['params']] for group in self.param_groups]
    
    def step(self, closure=None):
        loss = self.optimizer.step(closure)

        for group, slow_weight in zip(self.param_groups, self.slow_weights):
            group['k_counter'] += 1
            if group['k_counter'] < self.k:
                continue
            if group['k_counter'] == self.k:
                for param, weight in zip(group['params'], slow_weight):
                    weight.data.add_(self.alpha, (param.data - weight.data))
                    param.data.copy_(weight.data)
                group['k_counter'] = 0
        return loss

    def state_dict(self):
        fast_dict = self.optimizer.state_dict()
        fast_state = fast_dict['state']
        param_groups = fast_dict['param_groups']
        slow_state = {(id(k) if isinstance(k, torch.Tensor) else k): v
                        for k, v in self.state.items()}
        return {
            'fast_state': fast_state,
            'param_groups': param_groups,
            'slow_state': slow_state
        }

    def load_state_dict(self, state_dict):
        fast_dict = {
            'state': state_dict['fast_state'],
            'param_groups': state_dict['param_groups']
        }
        slow_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups']
        }
        super(Lookahead, self).load_state_dict(slow_dict)
        self.optimizer.load_state_dict(fast_dict)

    def add_param_group(self, param_group):
        param_group['k_counter'] = 0
        self.optimizer.add_param_group(param_group)
