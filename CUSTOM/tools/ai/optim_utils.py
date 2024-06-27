import paddle
import paddle.optimizer as optim
from .torch_utils import *

class PolyOptimizer(optim.SGD):
    def __init__(self, parameters, learning_rate, weight_decay, max_step, momentum=0.9, nesterov=False):
        super().__init__(learning_rate=learning_rate, parameters=parameters, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum
        
        self.__initial_lr = [group['learning_rate'] for group in self._parameter_list]
    
    def step(self, closure=None):
        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i, param_group in enumerate(self._parameter_list):
                param_group['learning_rate'] = self.__initial_lr[i] * lr_mult

        super().step(closure=closure)

        self.global_step += 1
