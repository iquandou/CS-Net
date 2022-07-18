# -*- coding: utf-8 -*-
# @Time    : 2018-9-27 15:39
# @Author  : xylon
# @Modified : qd

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.image_utils import filter_border, nms, topk_map, get_gauss_filter_weight
from utils.math_utils import distance_matrix_vector, pairwise_distances

class RFDetModule(nn.Module):
    def __init__(
        self,
        score_com_strength,
        scale_com_strength,
        nms_thresh,
        nms_ksize,
        topk,
        gauss_ksize,
        gauss_sigma,
        ksize,
        padding,
        dilation,
        scale_list,
        C,
    ):
        super(RFDetModule, self).__init__()

        self.score_com_strength = score_com_strength
        self.scale_com_strength = scale_com_strength
        self.NMS_THRESH = nms_thresh
        self.NMS_KSIZE = nms_ksize
        self.TOPK = topk
        self.GAUSSIAN_KSIZE = gauss_ksize
        self.GAUSSIAN_SIGMA = gauss_sigma

        self.conv3_1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=dilation,
        )  
        self.insnorm3_1 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s3_1 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.insnorm_s3_1 = nn.InstanceNorm2d(1, affine=True)

        self.conv3_2 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=dilation,
        )  
        self.insnorm3_2 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s3_2 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.insnorm_s3_2 = nn.InstanceNorm2d(1, affine=True)
        self.conv3_3 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=dilation,
        )  
        self.insnorm3_3 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s3_3 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.insnorm_s3_3 = nn.InstanceNorm2d(1, affine=True)

        self.conv3_4 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=dilation,
        )  
        self.insnorm3_4 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s3_4 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.insnorm_s3_4 = nn.InstanceNorm2d(1, affine=True)

        self.conv3_5 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=dilation,
        )  
        self.insnorm3_5 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s3_5 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.insnorm_s3_5 = nn.InstanceNorm2d(1, affine=True)

        self.conv5_1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2,
            dilation=dilation,
        ) 
        self.insnorm5_1 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s5_1 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.insnorm_s5_1 = nn.InstanceNorm2d(1, affine=True)

        self.conv5_2 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2,
            dilation=dilation,
        )  
        self.insnorm5_2 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s5_2 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.insnorm_s5_2 = nn.InstanceNorm2d(1, affine=True)

        self.conv5_3 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2,
            dilation=dilation,
        ) 
        self.insnorm5_3 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s5_3 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.insnorm_s5_3 = nn.InstanceNorm2d(1, affine=True)

        self.conv7_1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=7,
            stride=1,
            padding=3,
            dilation=dilation,
        ) 
        self.insnorm7_1 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s7_1 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.insnorm_s7_1 = nn.InstanceNorm2d(1, affine=True)

        self.conv7_2 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=7,
            stride=1,
            padding=3,
            dilation=dilation,
        )  
        self.insnorm7_2 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s7_2 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.insnorm_s7_2 = nn.InstanceNorm2d(1, affine=True)

        self.conv9_1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=9,
            stride=1,
            padding=4,
            dilation=dilation,
        ) 
        self.insnorm9_1 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s9_1 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.insnorm_s9_1 = nn.InstanceNorm2d(1, affine=True)

        self.attention_avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        self.attention_conv_1 = nn.Conv2d(
            in_channels=10,
            out_channels=10,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.attention_conv_2 = nn.Conv2d(
            in_channels=10,
            out_channels=10,
            kernel_size=1,
            stride=1,
            padding=0,
        )


        self.scale_list = torch.tensor(scale_list)
        self.C = C

    def forward(self, **kwargs):
        pass

    def process(self, im1w_score):
        """
        nms(n), topk(t), gaussian kernel(g) operation
        :param im1w_score: warped score map
        :return: processed score map, topk mask, topk value
        """
        im1w_score = filter_border(im1w_score)

        # apply nms to im1w_score
        nms_mask = nms(im1w_score, thresh=self.NMS_THRESH, ksize=self.NMS_KSIZE)
        im1w_score = im1w_score * nms_mask
        topk_value = im1w_score

        # apply topk to im1w_score
        topk_mask = topk_map(im1w_score, self.TOPK)
        im1w_score = topk_mask.to(torch.float) * im1w_score

        # apply gaussian kernel to im1w_score
        psf = im1w_score.new_tensor(
            get_gauss_filter_weight(self.GAUSSIAN_KSIZE, self.GAUSSIAN_SIGMA)[
                None, None, :, :
            ]
        )
        im1w_score = F.conv2d(
            input=im1w_score.permute(0, 3, 1, 2),
            weight=psf,
            stride=1,
            padding=self.GAUSSIAN_KSIZE // 2,
        ).permute(
            0, 2, 3, 1
        )  # (B, H, W, 1)

        """
        apply tf.clamp to make sure all value in im1w_score isn't greater than 1
        but this won't happend in correct way
        """
        im1w_score = im1w_score.clamp(min=0.0, max=1.0)

        return im1w_score, topk_mask, topk_value

    @staticmethod
    def loss(left_score, im1gt_score, im1visible_mask):
        im1_score = left_score

        l2_element_diff = (im1_score - im1gt_score) ** 2
        # visualization numbers
        Nvi = torch.clamp(im1visible_mask.sum(dim=(3, 2, 1)), min=2.0)
        loss = (
            torch.sum(l2_element_diff * im1visible_mask, dim=(3, 2, 1)) / (Nvi + 1e-8)
        ).mean()

        return loss
    @staticmethod
    def loss_cosis(left_score, im1gt_score, im1visible_mask, im1_topkmask):
        mask = im1_topkmask * im1_topkmask
        im1_score_top = left_score.masked_select(mask)
        im2_score_top = im1gt_score.masked_select(mask)
        Num_score =  im1_score_top.shape[0]
        im1_score_top = torch.unsqueeze(im1_score_top, 1).repeat(1, Num_score)
        im2_score_top = torch.unsqueeze(im2_score_top, 1).repeat(1, Num_score)
        loss_cosis = (torch.clamp(torch.exp (-torch.mul((im1_score_top - im1_score_top.t()) , (im2_score_top - im2_score_top.t())))-1, min=0).sum(dim=0))/Num_score
       # loss_cosis = (((im1_score_top - im1_score_top.t()) - (im2_score_top - im2_score_top.t())) ** 2).sum(dim=0)/ Num_score
       # loss_cosis = (torch.clamp( 1 -(-torch.mul((im1_score_top - im1_score_top.t()) , (im2_score_top - im2_score_top.t()))), min=0).sum(dim=0))/Num_score
        return loss_cosis.mean()

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
        discrim_rank = torch.unsqueeze(discrim_loss, 1).repeat(1, 512)
        loss_discrim_rank = (torch.clamp(torch.exp(torch.mul((im1_score_rank - im1_score_rank.t()), (discrim_rank - discrim_rank.t())))-1, min=0).sum(dim=0)) / 511
        return loss_discrim_score.mean(), loss_discrim_rank.mean()
    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(
                m.weight.data, gain=nn.init.calculate_gain("leaky_relu")
            )
            try:
                nn.init.xavier_uniform_(m.bias.data)
            except:
                pass
