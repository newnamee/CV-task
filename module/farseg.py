import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from simplecv.interface import CVModule
from simplecv import registry
from simplecv.module import resnet
from simplecv.module import fpn
import math
from module.loss import softmax_focalloss
from module.loss import annealing_softmax_focalloss
from module.loss import cosine_annealing, poly_annealing, linear_annealing
import simplecv.module as scm
from configs.isaid.farseg50 import config as cfg


class SceneRelation(nn.Module):
    '''
        该模块接受一个场景特征和多个特征列表作为输入，对这些特征进行处理，并输出经过关系加权处理后的特征。
    在初始化过程中，根据参数scale_aware_proj的值选择是否使用多个网络进行场景特征的处理。
    如果scale_aware_proj为True，则会使用多个卷积操作对场景特征进行处理；否则仅使用一个卷积操作。
    同时，对于特征列表中的每一个特征，都会有一个内容编码器和特征重编码器用于处理。
    在前向传播过程中，首先对特征列表中的每一个特征进行内容编码，然后根据场景特征和编码后的特征计算关系。
    根据是否使用多个网络处理场景特征，选择相应的关系计算方式。最后，将关系和特征列表中的特征进行点乘，
    并经过特征重编码器处理，得到处理后的特征列表。
    这样设计的目的是利用关系图对特征进行重加权，提高前景特征的区分度，从而提升模型性能。
    '''
    def __init__(self,
                 in_channels,
                 channel_list,
                 out_channels,
                 scale_aware_proj=True):
        super(SceneRelation, self).__init__()
        self.scale_aware_proj = scale_aware_proj

        if scale_aware_proj:
            self.scene_encoder = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.ReLU(True),
                    nn.Conv2d(out_channels, out_channels, 1),
                ) for _ in range(len(channel_list))]
            )
        else:
            # 2mlp
            self.scene_encoder = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 1),
            )
        self.content_encoders = nn.ModuleList()
        self.feature_reencoders = nn.ModuleList()
        for c in channel_list:
            # 通过内容保留机制改进：原始的F-S关系模块利用关系图对非线性变换的特征图进行重新加权，以提高前景特征的区分度
            self.content_encoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                )
            )
            self.feature_reencoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                )
            )

        self.normalizer = nn.Sigmoid()

    def forward(self, scene_feature, features: list):
        content_feats = [c_en(p_feat) for c_en, p_feat in zip(self.content_encoders, features)]
        if self.scale_aware_proj:
            scene_feats = [op(scene_feature) for op in self.scene_encoder]
            relations = [self.normalizer((sf * cf).sum(dim=1, keepdim=True)) for sf, cf in
                         zip(scene_feats, content_feats)]
        else:
            scene_feat = self.scene_encoder(scene_feature)
            relations = [self.normalizer((scene_feat * cf).sum(dim=1, keepdim=True)) for cf in content_feats]

        p_feats = [op(p_feat) for op, p_feat in zip(self.feature_reencoders, features)]

        refined_feats = [r * p for r, p in zip(relations, p_feats)]

        return refined_feats


class AssymetricDecoder(nn.Module):
    '''
        这个模块实现了一个非对称的解码器，用于对输入的特征进行解码处理。
    在初始化方法中，根据输入的参数设置，确定是否使用批量归一化或组归一化作为标准化函数，
    并根据不同的标准化函数进行初始化。接着，根据输入的特征输出步幅设置，构建解码器的网络块。
    每个网络块中包含了若干个卷积层、标准化层、ReLU激活函数以及上采样层。
    同时，根据输入和输出特征的步幅计算需要的上采样次数，并决定网络块中的层数。
    在前向传播方法中，遍历所有网络块，分别对输入的特征列表中的特征进行解码处理，
    得到解码后的特征列表。最后，将所有解码后的特征列表相加并除以4，得到最终的输出特征。
    这个解码器的设计是为了将不同步幅的特征进行解码合并，最终得到统一步幅的输出特征，以提高网络的性能和效果。
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_feat_output_strides=(4, 8, 16, 32),
                 out_feat_output_stride=4,
                 norm_fn=nn.BatchNorm2d,
                 num_groups_gn=None):
        super(AssymetricDecoder, self).__init__()
        if norm_fn == nn.BatchNorm2d:
            norm_fn_args = dict(num_features=out_channels)
        elif norm_fn == nn.GroupNorm:
            if num_groups_gn is None:
                raise ValueError('When norm_fn is nn.GroupNorm, num_groups_gn is needed.')
            norm_fn_args = dict(num_groups=num_groups_gn, num_channels=out_channels)
        else:
            raise ValueError('Type of {} is not support.'.format(type(norm_fn)))
        self.blocks = nn.ModuleList()
        for in_feat_os in in_feat_output_strides:
            num_upsample = int(math.log2(int(in_feat_os))) - int(math.log2(int(out_feat_output_stride)))

            num_layers = num_upsample if num_upsample != 0 else 1

            self.blocks.append(nn.Sequential(*[
                nn.Sequential(
                    nn.Conv2d(in_channels if idx == 0 else out_channels, out_channels, 3, 1, 1, bias=False),
                    norm_fn(**norm_fn_args) if norm_fn is not None else nn.Identity(),
                    nn.ReLU(inplace=True),
                    nn.UpsamplingBilinear2d(scale_factor=2) if num_upsample != 0 else nn.Identity(),
                )
                for idx in range(num_layers)]))

    def forward(self, feat_list: list):
        inner_feat_list = []
        for idx, block in enumerate(self.blocks):
            decoder_feat = block(feat_list[idx])
            inner_feat_list.append(decoder_feat)

        out_feat = sum(inner_feat_list) / 4.
        return out_feat


@registry.MODEL.register('FarSeg')
class FarSeg(CVModule):
    '''
    该模块实现了一个图像分割网络，通过使用ResNet编码器、FPN处理特征、非对称解码器进行解码等模块，
    可以实现对图像进行分割任务的预测和训练。
    '''
    def __init__(self, config):
        '''
        在初始化方法中，首先调用父类CVModule的初始化方法，并注册了一个名为buffer_step的缓冲区。
        然后依次创建了ResNet的编码器（self.en）、FPN（self.fpn）、非对称解码器（self.decoder）
        以及用于预测类别的卷积层（self.cls_pred_conv）等模块。根据是否支持场景关系（scene_relation）、
        损失类型等设置，选择性地打印相应信息和配置模块。
        '''
        super(FarSeg, self).__init__(config)
        self.register_buffer('buffer_step', torch.zeros((), dtype=torch.float32))

        self.en = resnet.ResNetEncoder(self.config.resnet_encoder)
        self.fpn = fpn.FPN(**self.config.fpn)
        self.decoder = AssymetricDecoder(**self.config.decoder)
        self.cls_pred_conv = nn.Conv2d(self.config.decoder.out_channels, self.config.num_classes, 1)
        self.upsample4x_op = nn.UpsamplingBilinear2d(scale_factor=4)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if 'scene_relation' in self.config:
            print('scene_relation: on')
            self.gap = scm.GlobalAvgPool2D()
            self.sr = SceneRelation(**self.config.scene_relation)

        if 'softmax_focalloss' in self.config:
            print('loss type: softmax_focalloss')

        if 'cosineannealing_softmax_focalloss' in self.config:
            print('loss type: cosineannealing_softmax_focalloss')

        if 'annealing_softmax_focalloss' in self.config:
            print('loss type: {}'.format(self.config.annealing_softmax_focalloss.annealing_type))

    def forward(self, x, y=None):
        '''
        前向传播方法，接受输入x和可选的标签y，返回预测结果或者训练损失。首先通过编码器获取特征列表，
        然后使用FPN对特征进行处理。根据是否存在场景关系，选择性地进行场景关系处理。
        之后，通过非对称解码器对特征列表进行解码，得到最终特征。将最终特征传入类别预测的卷积层，
        再进行4倍上采样，得到最终类别预测结果。如果处于训练状态，计算并返回损失值。
        '''
        feat_list = self.en(x)
        fpn_feat_list = self.fpn(feat_list)
        if 'scene_relation' in self.config:
            c5 = feat_list[-1]
            c6 = self.gap(c5)
            refined_fpn_feat_list = self.sr(c6, fpn_feat_list)
        else:
            refined_fpn_feat_list = fpn_feat_list

        final_feat = self.decoder(refined_fpn_feat_list)
        cls_pred = self.cls_pred_conv(final_feat)
        cls_pred = self.upsample4x_op(cls_pred)

        if self.training:
            cls_true = y['cls']
            loss_dict = dict()
            self.buffer_step += 1
            cls_loss_v = self.config.loss.cls_weight * self.cls_loss(cls_pred, cls_true)
            loss_dict['cls_loss'] = cls_loss_v

            mem = torch.cuda.max_memory_allocated() // 1024 // 1024
            loss_dict['mem'] = torch.from_numpy(np.array([mem], dtype=np.float32)).to(self.device)
            return loss_dict

        return cls_pred.softmax(dim=1)

    def cls_loss(self, y_pred, y_true):
        '''
        定义了分类损失的计算方法，包括Softmax Focal Loss和Cosine Annealing Softmax Focal Loss等
        不同的计算方式，根据配置选择相应的损失计算方式。
        :param y_pred:
        :param y_true:
        :return:
        '''
        if 'softmax_focalloss' in self.config:
            return softmax_focalloss(y_pred, y_true.long(), ignore_index=self.config.loss.ignore_index,
                                     gamma=self.config.softmax_focalloss.gamma,
                                     normalize=self.config.softmax_focalloss.normalize)
        elif 'annealing_softmax_focalloss' in self.config:
            func_dict = dict(cosine=cosine_annealing,
                             poly=poly_annealing,
                             linear=linear_annealing)
            return annealing_softmax_focalloss(y_pred, y_true.long(),
                                               self.buffer_step.item(),
                                               self.config.annealing_softmax_focalloss.max_step,
                                               self.config.loss.ignore_index,
                                               self.config.annealing_softmax_focalloss.gamma,
                                               func_dict[self.config.annealing_softmax_focalloss.annealing_type])
        return F.cross_entropy(y_pred, y_true.long(), ignore_index=self.config.loss.ignore_index)

    def set_defalut_config(self):
        '''
        设置默认的模型配置，包括ResNet编码器的参数、FPN的参数、解码器的参数、类别数等默认设置。
        '''
        self.config.update(dict(
            resnet_encoder=dict(
                resnet_type='resnet50',
                include_conv5=True,
                batchnorm_trainable=True,
                pretrained=False,
                freeze_at=0,
                # 8, 16 or 32
                output_stride=32,
                with_cp=(False, False, False, False),
                stem3_3x3=False,
                norm_layer=nn.BatchNorm2d,
            ),
            fpn=dict(
                in_channels_list=(256, 512, 1024, 2048),
                out_channels=256,
                conv_block=fpn.default_conv_block,
                top_blocks=None,
            ),
            decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                norm_fn=nn.BatchNorm2d,
                num_groups_gn=None
            ),
            num_classes=16,
            loss=dict(
                cls_weight=1.0,
                ignore_index=255,
            )
        ))


# if __name__ == '__main__':
#     print(FarSeg(config=cfg))
