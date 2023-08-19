"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None

from collections import defaultdict
class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = defaultdict(float)
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if self.weight_decay > 0.0:
                grad = param.grad.data + param.data * self.weight_decay
            else:
                grad = param.grad.data
            if param not in self.u:
                self.u[param] = 0.0
            self.u[param] = self.momentum * self.u[param] + (1 - self.momentum) * grad
            param.data -= self.lr * self.u[param]
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = defaultdict(float)
        self.v = defaultdict(float)

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for w in self.params:
            if self.weight_decay > 0:
                grad = w.grad.data + self.weight_decay * w.data
            else:
                grad = w.grad.data
            if w not in self.m:
                self.m[w] = 0.0
            if w not in self.v:
                self.v[w] = 0.0
            self.m[w] = self.beta1 * self.m[w] + (1 - self.beta1) * grad
            self.v[w] = self.beta2 * self.v[w] + (1 - self.beta2) * (grad ** 2)
            unbiased_m = self.m[w] / (1 - self.beta1**self.t)
            unbiased_v = self.v[w] / (1 - self.beta2**self.t)
            w.data = w.data - self.lr * unbiased_m / (unbiased_v**0.5 + self.eps)
        ### END YOUR SOLUTION
