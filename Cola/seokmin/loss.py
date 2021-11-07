import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        # target = torch.argmax(target, dim=-1)
        loss = self.CE(inputs, target)
        return loss


_criterion_entrypoints = {
    'cross_entropy': CrossEntropy,
}

def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]

def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints

def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return criterion