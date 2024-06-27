import paddle
from paddle.optimizer import Optimizer
from paddle.optimizer.lr import PolynomialDecay

class PolyOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        assert max_step > 0, "max_step should be greater than zero."
        defaults = dict(lr=lr, weight_decay=weight_decay, max_step=max_step, momentum=momentum)
        super(PolyOptimizer, self).__init__(parameter_list=params, **defaults)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum
        self.__initial_lr = [group['lr'] for group in self._parameter_list]

    def step(self):
        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i, params in enumerate(self._parameter_list):
                params['lr'] = self.__initial_lr[i] * lr_mult

        self.global_step += 1

        for group in self._parameter_list:
            group['lr'] = params['lr']

        super(PolyOptimizer, self).step()
