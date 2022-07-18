# -*- coding: utf-8 -*-
# @Time    : 2018-9-13 16:03
# @Author  : xylon
# @Modified : qd

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.image_utils import soft_nms_3d, soft_max_and_argmax_1d
from utils.math_utils import L2Norm
from model.rf_det_module import RFDetModule


class RFDetSO(RFDetModule):
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
        super(RFDetSO, self).__init__(
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
        )

        self.conv_o3_1 = nn.Conv2d(
            in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0
        )
        self.conv_o3_2 = nn.Conv2d(
            in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0
        )
        self.conv_o3_3 = nn.Conv2d(
            in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0
        )
        self.conv_o3_4 = nn.Conv2d(
            in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0
        )
        self.conv_o3_5 = nn.Conv2d(
            in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0
        )
        self.conv_o5_1 = nn.Conv2d(
            in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0
        )
        self.conv_o5_2 = nn.Conv2d(
            in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0
        )
        self.conv_o5_3 = nn.Conv2d(
            in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0
        )

        self.conv_o7_1 = nn.Conv2d(
            in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0
        )
        self.conv_o7_2 = nn.Conv2d(
            in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0
        )
        #        self.conv_o7_3 = nn.Conv2d(
        #            in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0
        #        )
        #        self.conv_o7_4 = nn.Conv2d(
        #            in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0
        #        )
        #         self.conv_o9_1 = nn.Conv2d(
        #            in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0
        #         )


    def forward(self, photos):

        """
        octave 3*3
        """
        score_featmaps_s3_1 = F.leaky_relu(self.insnorm3_1(self.conv3_1(photos)))
        score_map_s3_1 = self.insnorm_s3_1(self.conv_s3_1(score_featmaps_s3_1)).permute(
            0, 2, 3, 1
        )
        orint_map_s3_1 = (
            L2Norm(self.conv_o3_1(score_featmaps_s3_1), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )

        score_featmaps_s3_2 = F.leaky_relu(self.insnorm3_2(self.conv3_2(score_featmaps_s3_1)))
        score_map_s3_2 = self.insnorm_s3_2(self.conv_s3_2(score_featmaps_s3_2)).permute(
            0, 2, 3, 1
        )
        orint_map_s3_2 = (
            L2Norm(self.conv_o3_2(score_featmaps_s3_2), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )
        score_featmaps_s3_2 = score_featmaps_s3_1 + score_featmaps_s3_2

        score_featmaps_s3_3 = F.leaky_relu(self.insnorm3_3(self.conv3_3(score_featmaps_s3_2)))
        score_map_s3_3 = self.insnorm_s3_3(self.conv_s3_3(score_featmaps_s3_3)).permute(
            0, 2, 3, 1
        )
        orint_map_s3_3 = (
            L2Norm(self.conv_o3_3(score_featmaps_s3_3), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )
        score_featmaps_s3_3 = score_featmaps_s3_2 + score_featmaps_s3_3

        score_featmaps_s3_4 = F.leaky_relu(self.insnorm3_4(self.conv3_4(score_featmaps_s3_3)))
        score_map_s3_4 = self.insnorm_s3_4(self.conv_s3_4(score_featmaps_s3_4)).permute(
            0, 2, 3, 1
        )
        orint_map_s3_4 = (
            L2Norm(self.conv_o3_4(score_featmaps_s3_4), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )
        score_featmaps_s3_4 = score_featmaps_s3_3 + score_featmaps_s3_4

        score_featmaps_s3_5 = F.leaky_relu(self.insnorm3_5(self.conv3_5(score_featmaps_s3_4)))
        score_map_s3_5 = self.insnorm_s3_5(self.conv_s3_5(score_featmaps_s3_5)).permute(
            0, 2, 3, 1
        )
        orint_map_s3_5 = (
            L2Norm(self.conv_o3_5(score_featmaps_s3_5), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )

        # octave 5*5

        score_featmaps_s5_1 = F.leaky_relu(self.insnorm5_1(self.conv5_1(photos)))
        score_map_s5_1 = self.insnorm_s5_1(self.conv_s5_1(score_featmaps_s5_1)).permute(
            0, 2, 3, 1
        )
        orint_map_s5_1 = (
            L2Norm(self.conv_o5_1(score_featmaps_s5_1), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )
 
       # score_featmaps_s5_1 = score_featmaps_s3_1+score_featmaps_s5_1

        score_featmaps_s5_2 = F.leaky_relu(self.insnorm5_2(self.conv5_2(score_featmaps_s5_1)))
        score_map_s5_2 = self.insnorm_s5_2(self.conv_s5_2(score_featmaps_s5_2)).permute(
            0, 2, 3, 1
        )
        orint_map_s5_2 = (
            L2Norm(self.conv_o5_2(score_featmaps_s5_2), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )
        score_featmaps_s5_2 = score_featmaps_s5_1 + score_featmaps_s5_2

        score_featmaps_s5_3 = F.leaky_relu(self.insnorm5_3(self.conv5_3(score_featmaps_s5_2)))
        score_map_s5_3 = self.insnorm_s5_3(self.conv_s5_3(score_featmaps_s5_3)).permute(
            0, 2, 3, 1
        )
        orint_map_s5_3 = (
            L2Norm(self.conv_o5_3(score_featmaps_s5_3), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )
        # score_featmaps_s5_3 = score_featmaps_s5_3 + score_featmaps_s5_2

        #        score_featmaps_s5_4 = F.leaky_relu(self.insnorm5_4(self.conv5_4(score_featmaps_s5_3)))
        #        score_map_s5_4 = self.insnorm_s5_4(self.conv_s5_4(score_featmaps_s5_4)).permute(
        #            0, 2, 3, 1
        #        )
        #        orint_map_s5_4 = (
        #            L2Norm(self.conv_o5_4(score_featmaps_s5_4), dim=1)
        #            .permute(0, 2, 3, 1)
        #            .unsqueeze(-2)
        #        )
        #        score_featmaps_s5_4 = score_featmaps_s5_4 + score_featmaps_s5_3
        #

        score_featmaps_s7_1 = F.leaky_relu(self.insnorm7_1(self.conv7_1(photos)))
        score_map_s7_1 = self.insnorm_s7_1(self.conv_s7_1(score_featmaps_s7_1)).permute(
            0, 2, 3, 1
        )
        orint_map_s7_1 = (
            L2Norm(self.conv_o7_1(score_featmaps_s7_1), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )

        #score_featmaps_s7_1=score_featmaps_s3_1+score_featmaps_s7_1

        score_featmaps_s7_2 = F.leaky_relu(self.insnorm7_2(self.conv7_2(score_featmaps_s7_1)))
        score_map_s7_2 = self.insnorm_s7_2(self.conv_s7_2(score_featmaps_s7_2)).permute(
            0, 2, 3, 1
        )
        orint_map_s7_2 = (
            L2Norm(self.conv_o7_2(score_featmaps_s7_2), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )

        # score_featmaps_s9_1 = F.leaky_relu(self.insnorm9_1(self.conv9_1(photos)))
        # score_map_s9_1 = self.insnorm_s9_1(self.conv_s9_1(score_featmaps_s9_1)).permute(
        #     0, 2, 3, 1
        # )
        # orint_map_s9_1 = (
        #     L2Norm(self.conv_o9_1(score_featmaps_s9_1), dim=1)
        #     .permute(0, 2, 3, 1)
        #     .unsqueeze(-2)
        # )
        score_maps_merge = torch.cat(
            (
                score_map_s3_1,
                score_map_s3_2,
                score_map_s3_3,
                score_map_s3_4,
                score_map_s3_5,
                score_map_s5_1,
                score_map_s5_2,
                score_map_s5_3,
                score_map_s7_1,
                score_map_s7_2,
                # score_map_s9_1,
            ),
            -1,
        )  # (B, H, W, C)
        score_maps_merge_T = score_maps_merge.permute(0, -1, 1, 2)
 
        score_maps_attention_weights = F.sigmoid(self.attention_conv_2(F.relu(self.attention_conv_1(self.attention_avgpool(score_maps_merge_T)))))
        score_maps_attention_weights = score_maps_attention_weights.permute (0, 2, -1, 1)
        score_maps = score_maps_merge * score_maps_attention_weights
 

        orint_maps = torch.cat(
            (
                orint_map_s3_1,
                orint_map_s3_2,
                orint_map_s3_3,
                orint_map_s3_4,
                orint_map_s3_5,
                orint_map_s5_1,
                orint_map_s5_2,
                orint_map_s5_3,
                orint_map_s7_1,
                orint_map_s7_2,
                #orint_map_s9_1,
            ),
            -2,
        )  # (B, H, W, 10, 2)

        # get each pixel probability in all scale
        scale_probs = soft_nms_3d(score_maps, ksize=15, com_strength=3.0)

        # get each pixel probability summary from all scale space and correspond scale value
        score_map, scale_map, orint_map = soft_max_and_argmax_1d(
            input=scale_probs,
            orint_maps=orint_maps,
            dim=-1,
            scale_list=self.scale_list,
            keepdim=True,
            com_strength1=self.score_com_strength,
            com_strength2=self.scale_com_strength,
        )

        return score_map, scale_map, orint_map

    @staticmethod
    def convO_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.zeros_(m.weight.data)
            try:
                nn.init.ones_(m.bias.data)
            except:
                pass
