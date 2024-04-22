import torch
import torch.nn as nn
import torch.nn.functional as F

import math

def cross_entropy( pred, target_prob ):
    log_prob = F.log_softmax(pred, dim=1)
    return F.kl_div(input=log_prob, target=target_prob, reduction='none').sum(-1)

class AdaptiveAdversrialCrossEntropy(nn.Module):
    def __init__(self, th=0, mode='ada' ): # mode: 'ada', 'rand', 'const'
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.th = th
        self.mode = mode.lower()

    def forward( self, pred, target ):
        with torch.no_grad():
            onehot = F.one_hot( target, num_classes=pred.shape[1] )
            pred_prob = self.softmax( pred.detach() )
            pp = ( pred_prob * onehot ).sum(dim=1, keepdim=True)
            np = pred_prob * (1-onehot)

            dp = F.relu( pp - self.th )
            pp_adv = pp-dp

            if( self.mode == 'ada' ):
                ad = np
            elif( self.mode == 'rand' ):
                ad = (1-torch.rand( pred.shape, device=pred.device)) * (1-onehot)
            elif( self.mode == 'const' ):
                ad = (1-onehot)

            np_adv = ad / ad.sum(dim=1, keepdim=True)

            target_prob = pp_adv * onehot + np_adv

        return cross_entropy( pred, target_prob )
