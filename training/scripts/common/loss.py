import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MSE_Loss(nn.Module):
    def __init__(self, output_key=0, target_key=0):
        super(MSE_Loss, self).__init__()
        self.output_key = output_key
        self.target_key = target_key

    def forward(self, target, predict):
        assert (target[self.target_key].shape == predict[self.output_key].shape)

        pred = predict[self.output_key].view(-1)
        gt = target[self.target_key].view(-1)

        loss = (pred-gt)**2

        return torch.mean(loss)

class IterationRange(nn.Module):
    def __init__(self, loss, start=None, end=None):
        super(IterationRange, self).__init__()
        self.loss = loss
        self.start = start
        self.end = end
        self.counter = 0

    def forward(self, target, predict):
        skip = False

        if (self.start is not None) and (self.counter < self.start):
            skip = True

        if (self.end is not None) and (self.counter > self.end):
            skip = True

        self.counter = self.counter + 1

        if skip:
            return None

        loss = self.loss(target,predict)

        return loss

class BCE_Loss(nn.Module):
    def __init__(self, output_key=0, target_key=0, bg_weight=1):
        super(BCE_Loss, self).__init__()
        self.output_key = output_key
        self.target_key = target_key
        self.bg_weight=bg_weight

    def forward(self, target, predict):

        assert (target[self.target_key].shape == predict[self.output_key].shape)

        pred = predict[self.output_key].view(-1)
        gt = target[self.target_key].view(-1)

        loss = gt*torch.log(pred+1e-6) + \
        self.bg_weight*(1. - gt)* torch.log((1.+1e-6) - pred)
        #loss = F.binary_cross_entropy(pred, gt)

        smooth = 0.1
        eps = 1e-6

        #loss_smooth = -torch.log(pred+eps)

        return -torch.mean(loss)#torch.mean((1-smooth)*loss + smooth*loss_smooth)

class Dice_loss_joint(nn.Module):
    def __init__(self, output_key=0, target_key=0):
        super(Dice_loss_joint, self).__init__()
        self.output_key = output_key
        self.target_key = target_key

    def forward(self, target, predict):

        assert (target[self.target_key].shape == predict[self.output_key].shape)
        shape = target[self.target_key].shape

        pred = target[self.target_key].view(shape[0], shape[1], -1)
        gt = predict[self.output_key].view(shape[0], shape[1], -1)

        intersection = (pred*gt).sum(dim=(0,2))
        union = (pred**2 + gt).sum(dim=(0,2))
        eps = 1e-4
        dice = (2.0*intersection + eps) / (union + eps)

        return (1.0 - torch.mean(dice))


class Mix(nn.Module):
    def __init__(self, losses, coefficients=None):
        super(Mix, self).__init__()
        self.losses = losses
        self.coefficients = coefficients

        if self.coefficients is None:
            self.coefficients = { k:1 for k in self.losses}

    def forward(self, target, predict):

        losses_results = {k:self.losses[k](target, predict) for k in self.losses }

        loss = sum([losses_results[k]*self.coefficients[k] for k in losses_results if losses_results[k] is not None]) / (len(losses_results))

        return loss, losses_results
