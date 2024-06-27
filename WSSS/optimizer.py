import paddle
import paddle.optimizer as optim

class PolyOptimizer(optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(learning_rate=lr, parameters=params, weight_decay=weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['learning_rate'] for group in self._parameter_list]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self._parameter_list)):
                self._parameter_list[i]['learning_rate'] = self.__initial_lr[i] * lr_mult

        super().step()

        self.global_step += 1
