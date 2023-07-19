import math
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np




class WarmupAndOrderedScheduler(_LRScheduler):
    def __init__(self,
                 optimizer,
                 num_steps_per_epoch,
                 num_warmup_epochs=0.0,
                 min_delta_to_update_lr=1e-6,
                 epoch_start=0,
                ):
        self._num_steps_per_epoch = num_steps_per_epoch
        self._num_warmup_epochs = num_warmup_epochs
        self._min_delta_to_update_lr = min_delta_to_update_lr
        self._previous_lr = -1
        self._max_lr = optimizer.param_groups[0]['lr']
        super(WarmupAndOrderedScheduler, self).__init__(optimizer)
        self._step_count = self._num_steps_per_epoch * epoch_start

    def _epoch(self):
        return self._step_count // self._num_steps_per_epoch

    def _is_warmup_epoch(self):
        return self._epoch() < math.ceil(self._num_warmup_epochs)

    def get_lr(self):
        epoch = self._epoch()
        lr = 0.0
        decay_epoch_list = [3, 6, 10]
#         decay_epoch_list = [5,8,12]
        if self._is_warmup_epoch():
            # 预热阶段将学习速率从 0.0 线性增加到 self._max_lr
            num_warmup_steps = self._num_warmup_epochs * self._num_steps_per_epoch
            lr = min(
                self._max_lr,
                self._max_lr * ((self._step_count + 1.0) / num_warmup_steps))
        else:
            lr = self._max_lr * 0.1 ** (sum(epoch >= np.array(decay_epoch_list)))

        return [lr for _ in self.base_lrs]

    def step(self, epoch=None):
        current_lr = self.get_lr()[0]

        # 除预热阶段外，之后的每轮训练中，每一步的学习速率相同
        if abs(current_lr - self._previous_lr) >= self._min_delta_to_update_lr:
            super(WarmupAndOrderedScheduler, self).step()
            self._previous_lr = current_lr
        else:
            self._step_count += 1
