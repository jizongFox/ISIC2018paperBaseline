import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None,reduce=True,size_average=True):
        super().__init__()

        self.loss = nn.NLLLoss(weight,reduce=reduce,size_average=size_average)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs,dim=1), targets)