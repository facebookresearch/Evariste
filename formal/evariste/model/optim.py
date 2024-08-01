# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Any, Tuple, Union, Dict
import re
import math
import inspect

import torch
from torch import optim


class Adam(optim.Optimizer):
    """
    Same as https://github.com/pytorch/pytorch/blob/master/torch/optim/adam.py,
    without amsgrad, with step in a tensor, and states initialization in __init__.
    It was important to add `.item()` in `state['step'].item()`.

    This implementation is modified from torch.optim.Adam based on:
    `Fixed Weight Decay Regularization in Adam`
    (see https://arxiv.org/abs/1711.05101)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = 0  # torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # if group['weight_decay'] != 0:
                #     grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                # denom = exp_avg_sq.sqrt().clamp_(min=group['eps'])

                bias_correction1 = 1 - beta1 ** state["step"]  # .item()
                bias_correction2 = 1 - beta2 ** state["step"]  # .item()
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                if group["weight_decay"] != 0:
                    p.data.add_(p.data, alpha=-group["weight_decay"] * group["lr"])

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class AdamInverseSqrtWithWarmup(Adam):
    """
    Decay the LR based on the inverse square root of the update number.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`warmup-init-lr`) until the configured
    learning rate (`lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.
    During warmup:
        lrs = torch.linspace(warmup_init_lr, lr, warmup_updates)
        lr = lrs[update_num]
    After warmup:
        lr = decay_factor / sqrt(update_num)
    where
        decay_factor = lr * sqrt(warmup_updates)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        warmup_updates=4000,
        warmup_init_lr=1e-7,
        exp_factor=0.5,
    ):
        assert (isinstance(params[0], dict)) or isinstance(
            params[0], torch.nn.parameter.Parameter
        )
        if isinstance(params[0], torch.nn.parameter.Parameter):
            params = [{"params": params}]
        self.exp_factor = []
        self.warmup_updates = []
        self.warmup_init_lr = []
        self.lr_step = []
        self.decay_factor = []
        for param_group in params:
            # linearly warmup for the first warmup_updates
            self.warmup_updates.append(
                param_group.get("warmup_updates", warmup_updates)
            )
            self.warmup_init_lr.append(
                param_group.get("warmup_init_lr", warmup_init_lr)
            )
            warmup_end_lr = param_group.get("lr", lr)
            self.lr_step.append(
                (warmup_end_lr - self.warmup_init_lr[-1]) / self.warmup_updates[-1]
            )

            # then, decay prop. to the inverse square root of the update number
            self.exp_factor.append(param_group.get("exp_factor", exp_factor))
            self.decay_factor.append(
                warmup_end_lr * self.warmup_updates[-1] ** self.exp_factor[-1]
            )
            if "lr" in param_group:
                del param_group["lr"]

        super().__init__(
            params, lr=warmup_init_lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        assert all(
            len(self.param_groups) == len(p)
            for p in [
                self.lr_step,
                self.exp_factor,
                self.warmup_init_lr,
                self.warmup_updates,
                self.decay_factor,
            ]
        )
        # total number of updates
        for param_group in self.param_groups:
            param_group["num_updates"] = 0

    def get_lr_for_step(self, i, num_updates):
        if num_updates < self.warmup_updates[i]:
            return self.warmup_init_lr[i] + num_updates * self.lr_step[i]
        else:
            return self.decay_factor[i] * (num_updates ** -self.exp_factor[i])

    def step(self, closure=None):
        super().step(closure)
        for i, param_group in enumerate(self.param_groups):
            param_group["num_updates"] += 1
            param_group["lr"] = self.get_lr_for_step(i, param_group["num_updates"])


class AdamWithWarmup(Adam):
    """
    Decay the LR based on the inverse square root of the update number.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`warmup-init-lr`) until the configured
    learning rate (`lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.
    During warmup:
        lrs = torch.linspace(warmup_init_lr, lr, warmup_updates)
        lr = lrs[update_num]
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        warmup_updates=4000,
        warmup_init_lr=1e-7,
    ):
        assert (isinstance(params[0], dict)) or isinstance(
            params[0], torch.nn.parameter.Parameter
        )
        if isinstance(params[0], torch.nn.parameter.Parameter):
            params = [{"params": params}]
        self.warmup_updates = []
        self.warmup_init_lr = []
        self.warmup_end_lr = []
        self.lr_step = []
        for param_group in params:
            # linearly warmup for the first warmup_updates
            self.warmup_updates.append(
                param_group.get("warmup_updates", warmup_updates)
            )
            self.warmup_init_lr.append(
                param_group.get("warmup_init_lr", warmup_init_lr)
            )
            self.warmup_end_lr.append(param_group.get("lr", lr))
            self.lr_step.append(
                (self.warmup_end_lr[-1] - self.warmup_init_lr[-1])
                / self.warmup_updates[-1]
            )
            if "lr" in param_group:
                del param_group["lr"]

        super().__init__(
            params, lr=warmup_init_lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        assert all(
            len(self.param_groups) == len(p)
            for p in [self.lr_step, self.warmup_init_lr, self.warmup_updates]
        )
        # total number of updates
        for param_group in self.param_groups:
            param_group["num_updates"] = 0

    def get_lr_for_step(self, i, num_updates):
        if num_updates < self.warmup_updates[i]:
            return self.warmup_init_lr[i] + num_updates * self.lr_step[i]
        else:
            return self.warmup_end_lr[i]

    def step(self, closure=None):
        super().step(closure)
        for i, param_group in enumerate(self.param_groups):
            param_group["num_updates"] += 1
            param_group["lr"] = self.get_lr_for_step(i, param_group["num_updates"])


class AdamCosineWithWarmup(Adam):
    """
    Assign LR based on a cyclical schedule that follows the cosine function.
    See https://arxiv.org/pdf/1608.03983.pdf for details.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``).
    During warmup::
      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]
    After warmup::
      lr = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(t_curr / t_i))
    where ``t_curr`` is current percentage of updates within the current period
    range and ``t_i`` is the current period range, which is scaled by ``t_mul``
    after every iteration.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        warmup_updates=4000,
        warmup_init_lr=1e-7,
        min_lr=1e-9,
        init_period=1000000,
        period_mult=1,
        lr_shrink=0.75,
    ):
        super().__init__(
            params, lr=warmup_init_lr, betas=betas, eps=eps, weight_decay=weight_decay
        )

        # linearly warmup for the first warmup_updates
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        warmup_end_lr = lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates

        # then, apply cosine scheduler
        self.min_lr = min_lr
        self.max_lr = lr
        self.period = init_period
        self.period_mult = period_mult
        self.lr_shrink = lr_shrink

        # total number of updates
        for param_group in self.param_groups:
            param_group["num_updates"] = 0

    def get_lr_for_step(self, num_updates):
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates * self.lr_step
        else:
            t = num_updates - self.warmup_updates
            if self.period_mult == 1:
                pid = math.floor(t / self.period)
                t_i = self.period
                t_curr = t - (self.period * pid)
            else:
                pid = math.floor(
                    math.log(
                        1 - t / self.period * (1 - self.period_mult), self.period_mult
                    )
                )
                t_i = self.period * (self.period_mult ** pid)
                t_curr = (
                    t
                    - (1 - self.period_mult ** pid)
                    / (1 - self.period_mult)
                    * self.period
                )
            lr_shrink = self.lr_shrink ** pid
            min_lr = self.min_lr * lr_shrink
            max_lr = self.max_lr * lr_shrink
            return min_lr + 0.5 * (max_lr - min_lr) * (
                1 + math.cos(math.pi * t_curr / t_i)
            )

    def step(self, closure=None):
        super().step(closure)
        for param_group in self.param_groups:
            param_group["num_updates"] += 1
            param_group["lr"] = self.get_lr_for_step(param_group["num_updates"])


def get_optimizer(parameters, s: str) -> optim.Optimizer:
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    optim_params: Dict[str, Union[float, Tuple[float, float]]] = {}

    if "," in s:
        method = s[: s.find(",")]
        for x in s[s.find(",") + 1 :].split(","):
            split = x.split("=")
            assert len(split) == 2
            assert re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s

    optim_fn: Any = None

    if method.startswith("adam") and method != "adamax":
        beta1 = optim_params.pop("beta1", 0.9)
        beta2 = optim_params.pop("beta2", 0.999)
        assert type(beta1) is float and type(beta2) is float
        optim_params["betas"] = (beta1, beta2)

    if method == "adadelta":
        optim_fn = optim.Adadelta
    elif method == "adagrad":
        optim_fn = optim.Adagrad
    elif method == "adam":
        optim_fn = Adam
    elif method == "adam_warmup":
        optim_fn = AdamWithWarmup
    elif method == "adam_inverse_sqrt":
        optim_fn = AdamInverseSqrtWithWarmup
    elif method == "adam_cosine":
        optim_fn = AdamCosineWithWarmup
    elif method == "adamax":
        optim_fn = optim.Adamax
    elif method == "asgd":
        optim_fn = optim.ASGD
    elif method == "rmsprop":
        optim_fn = optim.RMSprop
    elif method == "rprop":
        optim_fn = optim.Rprop
    elif method == "sgd":
        optim_fn = optim.SGD
        assert "lr" in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    assert optim_fn is not None
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ["self", "params"]
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception(
            'Unexpected parameters: expected "%s", got "%s"'
            % (str(expected_args[2:]), str(optim_params.keys()))
        )
    return optim_fn(parameters, **optim_params)
