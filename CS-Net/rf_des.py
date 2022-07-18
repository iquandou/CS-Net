# -*- coding: utf-8 -*-
# @Time    : 2018-9-27 15:39
# @Author  : xylon
# @Modified : qd

import torch
import torch.nn as nn

from utils.math_utils import distance_matrix_vector, pairwise_distances


class HardNetNeiMask(nn.Module):
    def __init__(self, MARGIN, C):
        super(HardNetNeiMask, self).__init__()

        self.MARGIN = MARGIN
        self.C = C

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

    @staticmethod
    def input_norm(x):
        eps = 1e-8
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + eps
        return (
            x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        ) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        feature = x / torch.norm(x, p=2, dim=-1, keepdim=True)
        return feature

    def loss(self, anchor, positive, anchor_kp, positive_kp):
        """
        HardNetNeiMask
        margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
        if set C=0 the loss function is same as hard loss.
        """
        "Input sizes between positive and negative must be equal."
        assert anchor.size() == positive.size()
        "Inputd must be a 2D matrix."
        assert anchor.dim() == 2

        dist_matrix = distance_matrix_vector(anchor, positive)
        eye = torch.eye(dist_matrix.size(1)).to(dist_matrix.device)

        # steps to filter out same patches that occur in distance matrix as negatives
        pos = dist_matrix.diag()
        dist_without_min_on_diag = dist_matrix + eye * 10

        # neighbor mask
        coo_dist_matrix = pairwise_distances(
            anchor_kp[:, 1:3].to(torch.float), anchor_kp[:, 1:3].to(torch.float)
        ).lt(self.C)
        dist_without_min_on_diag = (
            dist_without_min_on_diag + coo_dist_matrix.to(torch.float) * 10
        )
        coo_dist_matrix = pairwise_distances(
            positive_kp[:, 1:3].to(torch.float), positive_kp[:, 1:3].to(torch.float)
        ).lt(self.C)
        dist_without_min_on_diag = (
            dist_without_min_on_diag + coo_dist_matrix.to(torch.float) * 10
        )
        col_min = dist_without_min_on_diag.min(dim=1)[0]
        row_min = dist_without_min_on_diag.min(dim=0)[0]
        col_row_min = torch.min(col_min, row_min)

        # triplet loss
        hard_loss = torch.clamp(self.MARGIN + pos - col_row_min, min=0.0)
        hard_loss = hard_loss.mean()

        return hard_loss


    def loss_discrim(self, im1_score, im1_topkmask, anchor, positive, anchor_kp, positive_kp):
        "Input sizes between positive and negative must be equal."
        assert anchor.size() == positive.size()
        "Inputd must be a 2D matrix."
        assert anchor.dim() == 2

        dist_matrix = torch.exp((0.5+0.5*torch.mm(anchor, positive.t()))/0.1)
        eye = torch.eye(dist_matrix.size(1)).to(dist_matrix.device)

        # steps to filter out same patches that occur in distance matrix as negatives
        pos = dist_matrix.diag()

        dist_without_min_on_diag = dist_matrix - eye * 100000

        # neighbor mask
        coo_dist_matrix = pairwise_distances(
            anchor_kp[:, 1:3].to(torch.float), anchor_kp[:, 1:3].to(torch.float)
        ).lt(self.C)
        dist_without_min_on_diag = (
                dist_without_min_on_diag - coo_dist_matrix.to(torch.float) * 100000
        )
        coo_dist_matrix = pairwise_distances(
            positive_kp[:, 1:3].to(torch.float), positive_kp[:, 1:3].to(torch.float)
        ).lt(self.C)
        dist_without_min_on_diag = (
                dist_without_min_on_diag - coo_dist_matrix.to(torch.float) * 100000
        )

        Neg_norm_1_topk, _ = torch.topk(dist_without_min_on_diag, 20, dim=0, largest=True)
        Neg_norm_2_topk, _ = torch.topk(dist_without_min_on_diag, 20, dim=1, largest=True)

        Neg_norm_1 = torch.sum(Neg_norm_1_topk, 0)
        Neg_norm_2 = torch.sum(Neg_norm_2_topk, 1)
       

        discrim_loss = (-torch.log(pos/(pos+Neg_norm_1))-torch.log(pos/(pos+Neg_norm_2)))/2

        im1_score_top = im1_score.masked_select(im1_topkmask)
        loss_discrim_score = (im1_score_top/(torch.sum(im1_score_top)+1e-8)) * discrim_loss

        im1_score_rank = torch.unsqueeze(im1_score_top, 1).repeat(1, 512)
        discrim_rank = torch.unsqueeze(-discrim_loss, 1).repeat(1, 512)
        loss_discrim_rank = (torch.clamp(torch.exp(torch.mul((im1_score_rank - im1_score_rank.t()), (discrim_rank - discrim_rank.t())))-1, min=0).sum(
            dim=0)) / 511
        return loss_discrim_score.mean(), loss_discrim_rank.mean()

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data, gain=0.6)
            try:
                nn.init.constant(m.bias.data, 0.01)
            except:
                pass
