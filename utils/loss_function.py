import torch as th
from torch import nn
from utils.support import *
import numpy as np
import cv2

class SilogLoss(nn.Module):
    def __init__(self):
        super(SilogLoss, self).__init__()

    def forward(self, depth_est, target):
        silog_loss = -psnr(depth_est,target)
        return silog_loss

# --------------------------------------------
# TV loss
# --------------------------------------------
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        """
        Total variation loss
        https://github.com/jxgu1016/Total_Variation_Loss.pytorch
        Args:
            tv_loss_weight (int):
        """
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = th.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = th.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
    
class Canny(nn.Module):
    def __init__(self):
        super(Canny, self).__init__()
        
    def forward(self, test, true):
        test_np = test.detach().numpy()
        true_np = true.detach().numpy()
        test_edge = cv2.Canny((test_np*255).astype(np.uint8), 10, 100)
        true_edge = cv2.Canny((true_np*255).astype(np.uint8), 10, 100)
        test_edge = th.from_numpy(test_edge/255.)
        true_edge = th.from_numpy(true_edge/255.)
        edge_loss = nn.BCELoss(test_edge, true_edge)
        
        return edge_loss