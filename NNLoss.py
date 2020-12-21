import torch
import torch.nn as nn
import torch.nn.functional as F
# ==============================================================================


def dice_loss(input, target):
    smooth = 1
    # input = F.softmax(input, dim=1)
    # input = torch.sigmoid(input) #for binary
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum()
    # union = (torch.mul(iflat, iflat) + torch.mul(tflat, tflat)).sum()
    dice_score = (2.*intersection + smooth)/(union + smooth)
    return 1-dice_score


class focal_loss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def kt_loss(student_output_digits, teacher_output_digits, tempearture):
    # The KL Divergence for PyTorch comparing the softmaxs of teacher and student expects the input tensor to be log probabilities!
    knowledge_transfer_loss = nn.KLDivLoss()(F.logsigmoid(student_output_digits / tempearture), torch.sigmoid(teacher_output_digits / tempearture)) * (tempearture * tempearture)
    return knowledge_transfer_loss