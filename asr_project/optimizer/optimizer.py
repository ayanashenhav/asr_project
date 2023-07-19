# from lion_pytorch import Lion
import numpy as np
from torch.optim import AdamW, Adam, Optimizer


def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []
    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params

def get_optimizer(params, optim_conf):
    wd = optim_conf['wd']
    lr = optim_conf['lr']
    betas = optim_conf['betas']
    eps = optim_conf['eps']

    has_wd = optim_conf['wd'] > 0

    if optim_conf['filter_by_requires_grad']:
        params = list(filter(lambda t: t.requires_grad, params))

    if optim_conf['group_wd_params'] and has_wd:
        wd_params, no_wd_params = separate_weight_decayable_params(params)

        params = [
            {'params': wd_params},
            {'params': no_wd_params, 'weight_decay': 0},
        ]

    # if optim_conf['use_lion']:
    #     return Lion(params, lr=lr, betas=betas, weight_decay=wd)

    if not has_wd:
        return Adam(params, lr=lr, betas=betas, eps=eps)

    return AdamW(params, lr=lr, betas=betas, weight_decay=wd, eps=eps)

class ScheduledOptim(Optimizer):
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, config, optimizer, current_step):

        super().__init__(optimizer.param_groups, optimizer.defaults)
        self._optimizer = optimizer
        model_name = config["model.name"]
        self.n_warmup_steps = config["train.scheduler.warm_up_step"]
        self.p = config["train.scheduler.p"]
        self.anneal_steps = config["train.scheduler.anneal_steps"]
        self.anneal_rate = config["train.scheduler.anneal_rate"]
        self.current_step = current_step
        self.encoder_hidden = config[f"model.modules.{model_name}.dim"]
        self.init_lr = np.power(self.encoder_hidden, -0.5)
        self.param_groups = self._optimizer.param_groups

    def step(self, closure):
        self._update_learning_rate()
        self._optimizer.step(closure=closure)

    def zero_grad(self, set_to_none: bool = False):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)
        self.param_groups = self._optimizer.param_groups

    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step * self.p, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step * self.p,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def state_dict(self):
        return self._optimizer.state_dict()

