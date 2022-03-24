import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Paper title: <PolypSeg+: a Lightweight Context-aware Network for Real-time Polyp Segmentation>
    Paper link:
    Paper accepted by TCYB 2022
"""

class SingleConvLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(SingleConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding, dilation, groups, bias),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Attention_Block(nn.Module):
    def __init__(self,ch,ratio=1):
        super(Attention_Block,self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch,ch//ratio,kernel_size=1,padding=0,bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch//ratio,ch,kernel_size=1,bias=True),
            nn.Sigmoid()
        )
    def forward(self,x):
        return x*self.attention(x)

class ASCModule(nn.Module):
    """
        The adaptive scale context (ASC) module described in paper.
    """

    def __init__(self, ch_in, ch_out):
        super(ASCModule, self).__init__()
        self.conv1 = SingleConvLayer(ch_in, ch_out // 3, 3, 1, 1, 1, 1, True)
        self.conv2 = SingleConvLayer(ch_in, ch_out // 3, 3, 1, 3, 3, 1, True)
        self.conv3 = SingleConvLayer(ch_in, ch_out // 3, 3, 1, 5, 5, 1, True)
        self.conv123 = SingleConvLayer(ch_out, ch_out, 3, 1, 1, 1, 1, True)
        self.se = Attention_Block(ch_out,ratio=16)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x123 = self.conv123(torch.cat((x1, x2, x3), dim=1))
        return self.se(x123)

class GCBlock(nn.Module):
    """
    Paper title: <GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond>
    Paper link: https://arxiv.org/abs/1904.11492
    Paper accepted by CVPR 2019
    """
    def __init__(self, inplanes, planes, pool, fusions):
        super(GCBlock, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out

class EGCModule(nn.Module):
    """
        The efficient global context (EGC) module described in paper.
    """
    def __init__(self, ch_in, ch_out):
        super(EGCModule, self).__init__()
        self.conv = SingleConvLayer(ch_in, ch_out)
        self.upsample = nn.Upsample(scale_factor=2)
        self.gc = GCBlock(ch_out, ch_out, 'att', ['channel_add', 'channel_mul'])

    def forward(self, low, high):
        high = self.upsample(self.conv(high))
        fusion = torch.mul(low, high)
        return self.gc(fusion)

class DepthwiseConv(nn.Module):
    def __init__(self, ch_in, ch_out, stride, dilation_rate=1):
        super(DepthwiseConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=stride, padding=dilation_rate, dilation=dilation_rate, groups=ch_in, bias=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class DepwiseConv_pyramid(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(DepwiseConv_pyramid,self).__init__()
        self.depwiseConv1 = nn.Sequential(DepthwiseConv(ch_in=ch_in,ch_out=ch_out//3,stride=1,dilation_rate=1))
        self.depwiseConv2 = nn.Sequential(DepthwiseConv(ch_in=ch_in,ch_out=ch_out//3,stride=1,dilation_rate=3))
        self.depwiseConv3 = nn.Sequential(DepthwiseConv(ch_in=ch_in,ch_out=ch_out//3,stride=1,dilation_rate=5))
    def forward(self,x):
        x1 = self.depwiseConv1(x)
        x2 = self.depwiseConv2(x)
        x3 = self.depwiseConv3(x)
        return torch.cat((x1,x2,x3),dim=1)

class Feature_Pyramid_Fusion(nn.Module):
    """
        The feature pyramid fusion (FPF) module described in paper.
    """
    def __init__(self, ch1_in, ch2_in, ch3_in, ch1_out, ch2_out, ch3_out):
        """
        ch1_in: high_level feature map filters
        ch3_in: low_level feature map filters
        """
        super(Feature_Pyramid_Fusion, self).__init__()
        self.conv1 = nn.Conv2d(ch1_in, ch1_out, kernel_size=1, stride=1, padding=0)
        self.up1 = nn.Upsample(scale_factor=4)
        self.conv2 = nn.Conv2d(ch2_in, ch2_out, kernel_size=1, stride=1, padding=0)
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv3 = nn.Conv2d(ch3_in, ch3_out, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Sequential(DepwiseConv_pyramid((ch1_out + ch2_out + ch3_out), (ch1_out + ch2_out + ch3_out)))

    def forward(self, high_feature, inner_feature, low_feature):
        x_high = self.conv1(high_feature)
        x_high = self.up1(x_high)

        x_inner1 = self.conv2(inner_feature)
        x_inner1 = self.up2(x_inner1)

        x_low = self.conv3(low_feature)
        x_concat = torch.cat((x_high,x_inner1,x_low),dim=1)
        x = self.conv4(x_concat)
        return x