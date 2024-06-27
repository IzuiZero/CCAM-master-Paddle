import paddle
import paddle.optimizer as optim

class PolyOptimizer(optim.Optimizer):
    def __init__(self, parameters, learning_rate, weight_decay, max_step, momentum=0.9, nesterov=False):
        defaults = dict(learning_rate=learning_rate, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
        super(PolyOptimizer, self).__init__(parameters, **defaults)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum
        
        self.__initial_lr = [group['learning_rate'] for group in self._parameter_list]

    def step(self):
        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i, param_group in enumerate(self._parameter_list):
                param_group['learning_rate'] = self.__initial_lr[i] * lr_mult

        super(PolyOptimizer, self).step()

        self.global_step += 1
