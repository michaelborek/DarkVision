import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        """
        Focal Loss for multi-class classification.
        
        Args:
            gamma (float): Focusing parameter. Defaults to 2.
            alpha (float, int, or list, optional): Weighting factor(s) for classes.
                If a single float/int is provided, it is assumed to be the weight for class 0,
                and (1 - alpha) will be used for the other class (if binary). You can also pass
                a list of weights (one per class). Defaults to None.
            reduction (str): Specifies the reduction to apply to the output: 'mean' | 'sum' | 'none'.
                Defaults to 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.Tensor([alpha, 1 - alpha])
            elif isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                raise TypeError("alpha must be float, int, or list")
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        logpt = F.log_softmax(inputs, dim=1)  
        pt = torch.exp(logpt)                 

        logpt = logpt.gather(1, targets.unsqueeze(1))  
        pt = pt.gather(1, targets.unsqueeze(1))        

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            alpha_t = alpha_t.view(-1, 1)
            logpt = logpt * alpha_t

        loss = - (1 - pt) ** self.gamma * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        Mean Squared Error Loss for classification tasks.
        If the inputs and targets shapes do not match (i.e., targets are class indices),
        this loss converts targets to one-hot encoded vectors.
        
        Args:
            reduction (str): Specifies the reduction method: 'mean' | 'sum' | 'none'. Defaults to 'mean'.
        """
        super(MSELoss, self).__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, inputs, targets):
        if inputs.size() != targets.size():
            num_classes = inputs.size(1)
            targets = torch.nn.functional.one_hot(targets, num_classes=num_classes).float().to(inputs.device)
        return self.mse(inputs, targets)
