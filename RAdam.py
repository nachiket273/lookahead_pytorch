import math
import torch
from torch.optim import Optimizer


class RAdam(Optimizer):
    r"""Implements RAdam algorithm.
    It has been proposed in `ON THE VARIANCE OF THE ADAPTIVE LEARNING
    RATE AND BEYOND(https://arxiv.org/pdf/1908.03265.pdf)`_.
    
    Arguments:
        params (iterable):      iterable of parameters to optimize or dicts defining
                                parameter groups
        lr (float, optional):   learning rate (default: 1e-3)
        betas (Tuple[float, float], optional):  coefficients used for computing
                                                running averages of gradient and 
                                                its square (default: (0.9, 0.999))
        eps (float, optional):  term added to the denominator to improve
                                numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional):    whether to use the AMSGrad variant of this
                                        algorithm from the paper `On the Convergence 
                                        of Adam and Beyond`_(default: False)
        
        sma_thresh:             simple moving average threshold.
                                Length till where the variance of adaptive lr is intracable.
                                Default: 4 (as per paper)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, sma_thresh=4):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(RAdam, self).__init__(params, defaults)

        self.radam_buffer = [[None, None, None] for ind in range(10)]
        self.sma_thresh = sma_thresh

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                old = p.data.float()

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                buffer = self.radam_buffer[int(state['step']%10)]

                if buffer[0] == state['step']:
                    sma_t, step_size = buffer[1], buffer[2]
                else:                 
                    sma_max_len = 2/(1-beta2) - 1  
                    beta2_t = beta2 ** state['step']
                    sma_t = sma_max_len - 2 * state['step'] * beta2_t /(1 - beta2_t)
                    buffer[0] = state['step']
                    buffer[1] = sma_t

                    if sma_t > self.sma_thresh :
                        rt = math.sqrt(((sma_t - 4) * (sma_t - 2) * sma_max_len)/((sma_max_len -4) * (sma_max_len - 2) * sma_t))
                        step_size = group['lr'] * rt * math.sqrt((1 - beta2_t)) / (1 - beta1 ** state['step'])                      
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])                        
                    buffer[2] = step_size

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], old)

                if sma_t > self.sma_thresh :
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p.data.add_(-step_size, exp_avg)

        return loss
