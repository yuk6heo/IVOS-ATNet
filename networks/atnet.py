import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.deeplab.aspp import ASPP
from networks.deeplab.backbone.resnet import SEResNet50
from networks.correlation_package.correlation import Correlation
from networks.ltm_transfer import LTM_transfer


class ATnet(nn.Module):
    def __init__(self, pretrained=1, resfix=False, corr_displacement=4, corr_stride=2):
        super(ATnet, self).__init__()
        print("Constructing ATnet architecture..")

        self.encoder_6ch = Encoder_6ch(resfix)
        self.encoder_3ch = Encoder_3ch(resfix)
        self.indicator_encoder = ConverterEncoder() #
        self.decoder_iact = Decoder()
        self.decoder_prop = Decoder_prop()

        self.ltm_local_affinity = Correlation(pad_size=corr_displacement * corr_stride, kernel_size=1,
                                               max_displacement=corr_displacement * corr_stride,
                                               stride1=1, stride2=corr_stride, corr_multiply=1)
        self.ltm_transfer = LTM_transfer(md=corr_displacement, stride=corr_stride)

        self.prev_conv1x1 = nn.Conv2d(256, 256, kernel_size=1, padding=0)  # 1/4, 256
        self.conv1x1 = nn.Conv2d(2048*2, 2048, kernel_size=1, padding=0)  # 1/16, 2048

        self.refer_weight = None
        self._initialize_weights(pretrained)

    def forward_ANet(self, x): # Bx4xHxW to Bx1xHxW
        r5, r4, r3, r2 = self.encoder_6ch(x)
        estimated_mask, m2 = self.decoder_iact(r5, r3, r2, only_return_feature=False)
        r5_indicator = self.indicator_encoder(r5, m2)
        return estimated_mask, r5_indicator

    def forward_TNet(self, anno_propEnc_r5_list, targframe_3ch, anno_iactEnc_r5_list, r2_prev, predmask_prev, debug_f_mask = False): #1/16, 2048
        f_targ, _, r3_targ, r2_targ = self.encoder_3ch(targframe_3ch)
        f_mask_r5 = self.correlation_global_transfer(anno_propEnc_r5_list, f_targ, anno_iactEnc_r5_list) # 1/16, 2048

        r2_targ_c = self.prev_conv1x1(r2_targ)
        r2_prev   = self.prev_conv1x1(r2_prev)
        f_mask_r2 = self.correlation_local_transfer(r2_prev, r2_targ_c, predmask_prev) # 1/4, 1 [B,1,H/4,W/4]

        r5_concat = torch.cat([f_targ, f_mask_r5], dim=1) # 1/16, 2048*2
        r5_concat = self.conv1x1(r5_concat)
        estimated_mask, m2 = self.decoder_prop(r5_concat, r3_targ, r2_targ, f_mask_r2)

        if not debug_f_mask:
            return estimated_mask, r2_targ
        else:
            return estimated_mask, r2_targ, f_mask_r2

    def correlation_global_transfer(self, anno_feature_list, targ_feature, anno_indicator_feature_list ):
        '''
        :param anno_feature_list: [B,C,H,W] x list (N values in list)
        :param targ_feature:  [B,C,H,W]
        :param anno_indicator_feature_list:  [B,C,H,W] x list (N values in list)
        :return targ_mask_feature: [B,C,H,W]
        '''

        b, c, h, w = anno_indicator_feature_list[0].size() # b means n_objs
        targ_feature = targ_feature.view(b, c, h * w) # [B, C, HxW]
        n_features = len(anno_feature_list)
        anno_feature = []
        for f_idx in range(n_features):
            anno_feature.append(anno_feature_list[f_idx].view(b, c, h * w).transpose(1, 2)) # [B, HxW', C]
        anno_feature = torch.cat(anno_feature, dim=1) # [B, NxHxW', C]
        sim_feature = torch.bmm(anno_feature, targ_feature) # [B, NxHxW', HxW]
        sim_feature = F.softmax(sim_feature, dim=2) / n_features # [B, NxHxW', HxW]
        anno_indicator_feature = []
        for f_idx in range(n_features):
            anno_indicator_feature.append(anno_indicator_feature_list[f_idx].view(b, c, h * w)) # [B, C, HxW']
        anno_indicator_feature = torch.cat(anno_indicator_feature, dim=-1) # [B, C, NxHxW']
        targ_mask_feature = torch.bmm(anno_indicator_feature, sim_feature) # [B, C, HxW]
        targ_mask_feature = targ_mask_feature.view(b, c, h, w)

        return targ_mask_feature

    def correlation_local_transfer(self, r2_prev, r2_targ, predmask_prev):
        '''

        :param r2_prev: [B,C,H,W]
        :param r2_targ:  [B,C,H,W]
        :param predmask_prev: [B,1,4*H,4*W]
        :return targ_mask_feature_r2: [B,1,H,W]
        '''

        predmask_prev = F.interpolate(predmask_prev, scale_factor=0.25, mode='bilinear',align_corners=True) # B,1,H,W
        sim_feature = self.ltm_local_affinity.forward(r2_targ,r2_prev,) # B,D^2,H,W
        sim_feature = F.softmax(sim_feature, dim=2)  # B,D^2,H,W
        predmask_targ = self.ltm_transfer.forward(sim_feature, predmask_prev, apply_softmax_on_simfeature = False)  # B,1,H,W

        return predmask_targ

    def _initialize_weights(self, pretrained):
        for m in self.modules():
            if pretrained:
                break
            else:
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0, 0.001)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()


class Encoder_3ch(nn.Module):
    # T-Net Encoder
    def __init__(self, resfix):
        super(Encoder_3ch, self).__init__()

        self.conv0_3ch = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)

        resnet = SEResNet50(output_stride=16, BatchNorm=nn.BatchNorm2d, pretrained=True)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/16, 1024
        self.res5 = resnet.layer4  # 1/16, 2048

        # freeze BNs
        if resfix:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    for p in m.parameters():
                        p.requires_grad = False

    def forward(self, x):
        x = self.conv0_3ch(x)  # 1/2, 64
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/16, 1024
        r5 = self.res5(r4)  # 1/16, 2048

        return r5, r4, r3, r2

    def forward_r2(self,x):
        x = self.conv0_3ch(x)  # 1/2, 64
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        return r2


class Encoder_6ch(nn.Module):
    # A-Net Encoder
    def __init__(self, resfix):
        super(Encoder_6ch, self).__init__()

        self.conv0_6ch = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=True)

        resnet = SEResNet50(output_stride=16, BatchNorm=nn.BatchNorm2d, pretrained=True)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/16, 1024
        self.res5 = resnet.layer4  # 1/16, 2048

        # freeze BNs
        if resfix:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    for p in m.parameters():
                        p.requires_grad = False

    def forward(self, x):

        x = self.conv0_6ch(x)  # 1/2, 64
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/16, 1024
        r5 = self.res5(r4)  # 1/16, 2048

        return r5, r4, r3, r2


class Decoder(nn.Module):
    # A-Net Decoder
    def __init__(self):
        super(Decoder, self).__init__()
        mdim = 256

        self.aspp_decoder = ASPP(backbone='res', output_stride=16, BatchNorm=nn.BatchNorm2d, pretrained=1)
        self.convG0 = nn.Conv2d(2048, mdim, kernel_size=3, padding=1)
        self.convG1 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)
        self.convG2 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)

        self.RF3 = Refine(512, mdim)  # 1/16 -> 1/8
        self.RF2 = Refine(256, mdim)  # 1/8 -> 1/4

        self.lastconv = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(),
                                      nn.Dropout(0.1),
                                      nn.Conv2d(256, 1, kernel_size=1, stride=1))

    def forward(self, r5, r3_targ, r2_targ, only_return_feature = False):

        aspp_out = self.aspp_decoder(r5) #1/16 mdim
        aspp_out = F.interpolate(aspp_out, scale_factor=4, mode='bilinear',align_corners=True) #1/4 mdim
        m4 = self.convG0(F.relu(r5))  # out: # 1/16, mdim
        m4 = self.convG1(F.relu(m4))  # out: # 1/16, mdim
        m4 = self.convG2(F.relu(m4)) # out: # 1/16, mdim


        m3 = self.RF3(r3_targ, m4)  # out: 1/8, mdim
        m2 = self.RF2(r2_targ, m3)  # out: 1/4, mdim
        m2 = torch.cat((m2, aspp_out), dim=1) # out: 1/4, mdim*2

        if only_return_feature:
            return m2

        x = self.lastconv(m2)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

        return x, m2


class Decoder_prop(nn.Module):
    # T-Net Decoder
    def __init__(self):
        super(Decoder_prop, self).__init__()
        mdim = 256

        self.aspp_decoder = ASPP(backbone='res', output_stride=16, BatchNorm=nn.BatchNorm2d, pretrained=1)
        self.convG0 = nn.Conv2d(2048, mdim, kernel_size=3, padding=1)
        self.convG1 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)
        self.convG2 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)

        self.RF3 = Refine(512, mdim)  # 1/16 -> 1/8
        self.RF2 = Refine(256, mdim)  # 1/8 -> 1/4

        self.lastconv = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(),
                                      nn.Dropout(0.1),
                                      nn.Conv2d(256, 1, kernel_size=1, stride=1))

    def forward(self, r5, r3_targ, r2_targ, f_mask_r2):

        aspp_out = self.aspp_decoder(r5) #1/16 mdim
        aspp_out = F.interpolate(aspp_out, scale_factor=4, mode='bilinear',align_corners=True) #1/4 mdim
        m4 = self.convG0(F.relu(r5))  # out: # 1/16, mdim
        m4 = self.convG1(F.relu(m4))  # out: # 1/16, mdim
        m4 = self.convG2(F.relu(m4)) # out: # 1/16, mdim

        m3 = self.RF3(r3_targ, m4)  # out: 1/8, mdim
        m3 = m3 + 0.5 * F.interpolate(f_mask_r2, scale_factor=0.5, mode='bilinear',align_corners=True) #1/4 mdim
        m2 = self.RF2(r2_targ, m3)  # out: 1/4, mdim
        m2 = m2 + 0.5 * f_mask_r2
        m2 = torch.cat((m2, aspp_out), dim=1) # out: 1/4, mdim*2

        x = self.lastconv(m2)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

        return x, m2


class ConverterEncoder(nn.Module):
    def __init__(self):
        super(ConverterEncoder, self).__init__()
        # [1/4, 512] to [1/8, 1024]
        downsample1 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=1, stride=2, bias=False),
                                        nn.BatchNorm2d(1024),
                                        )
        self.block1 = SEBottleneck(512, 256, stride = 2, downsample = downsample1)
        # [1/8, 1024] to [1/16, 2048]
        downsample2 = nn.Sequential(nn.Conv2d(1024, 2048, kernel_size=1, stride=2, bias=False),
                                    nn.BatchNorm2d(2048),
                                    )
        self.block2 = SEBottleneck(1024, 512, stride = 2, downsample=downsample2)
        self.conv1x1 = nn.Conv2d(2048 * 2, 2048, kernel_size=1, padding=0)  # 1/16, 2048

    def forward(self, r5, m2):
        '''

        :param r5: 1/16, 2048
        :param m2: 1/4, 512
        :return:
        '''
        x = self.block1(m2)
        x = self.block2(x)
        x = torch.cat((x,r5),dim=1)
        x = self.conv1x1(x)

        return x


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=nn.BatchNorm2d):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # SE
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(
            planes * 4, planes // 4, kernel_size=1, bias=False)
        self.conv_up = nn.Conv2d(
            planes // 4, planes * 4, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()

        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out1 = self.global_pool(out)
        out1 = self.conv_down(out1)
        out1 = self.relu(out1)
        out1 = self.conv_up(out1)
        out1 = self.sig(out1)

        if self.downsample is not None:
            residual = self.downsample(x)

        res = out1 * out + residual
        res = self.relu(res)

        return res


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        self.convFS2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convFS3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convMM1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convMM2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.convFS1(f)
        sr = self.convFS2(F.relu(s))
        sr = self.convFS3(F.relu(sr))
        s = s + sr

        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear',align_corners=True)
        mr = self.convMM1(F.relu(m))
        mr = self.convMM2(F.relu(mr))
        m = m + mr
        return m
