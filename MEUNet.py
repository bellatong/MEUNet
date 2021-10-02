import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()


def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)

    return src


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3 * dilation - 1) // 2,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out + x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 3, stride=1, dilation=1)
        self.layer2 = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3 = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4 = self.make_layer(512, 3, stride=2, dilation=1)
        self.initialize()

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out0 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out0, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)

        return out0, out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('/home/bianyetong/saliency_models/ours/res/resnet50-19c8e357.pth'),
                             strict=False)


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1,stride=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate,stride=1*stride)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

    def initialize(self):
        weight_init(self)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = x.mul(avg_out)
        return self.sigmoid(out)

    def initialize(self):
        print("Chanel-Attetion")


"""
class EDGE_PRODUCE(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate = 1):
        super(EDGE_PRODUCE,self).__init__()

        self.conv_e1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dirate = 1*dirate)"""


# RSU-7
class RSU7(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx5dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx5dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


# RSU-6
class RSU6(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx5dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx5dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


# RSU-5
class RSU5(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


# RSU-4
class RSU4(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


# UEN_A
class RSU4F(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


class Decoder(nn.Module):
    def __init__(self, out_ch=1):
        super(Decoder, self).__init__()

        self.stage5d = RSU4F(128, 128, 128)
        self.stage4d = RSU4F(128, 128, 128)
        self.stage3d = RSU4(128, 128, 128)
        self.stage2d = RSU4(128, 128, 128)
        self.stage1d = RSU5(128, 128, 128)

        self.stage1_edge = nn.Sequential(REBNCONV(128, 128, dirate=1), REBNCONV(128, 128, dirate=1),
                                         REBNCONV(128, 128, dirate=1), REBNCONV(128, 128, dirate=1))

        self.side1_edge = nn.Conv2d(128, 1, 3, padding=1, dilation=1)

        self.side1 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(128, out_ch, 3, padding=1)

        self.edge1_down = nn.Sequential(REBNCONV(128, 128, dirate=1), nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.edge2_down = nn.Sequential(REBNCONV(128, 128, dirate=1), nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.edge3_down = nn.Sequential(REBNCONV(128, 128, dirate=1), nn.MaxPool2d(2, stride=2, ceil_mode=True))
        self.edge4_down = nn.Sequential(REBNCONV(128, 128, dirate=1), nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.edgedownto56 = nn.Sequential(REBNCONV(128, 128, dirate=1), nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.outconv = nn.Conv2d(4, out_ch, 1)

        self.edb_conv = nn.Sequential(REBNCONV(128, 128, dirate=1), REBNCONV(128, 128, dirate=1))

    def forward(self, input1, input2, x, edb):

        input2_2_up = _upsample_like(input2[3], input2[4])
        out_edge = self.stage1_edge(torch.add(input2[4], input2_2_up))

        out_edge56 = self.edgedownto56(out_edge)

        out1_edge = self.edge1_down(out_edge56)
        out2_edge = self.edge2_down(out1_edge)
        out3_edge = self.edge3_down(out2_edge)
        out4_edge = self.edge4_down(out3_edge)


        edb = self.edb_conv(torch.add(edb,out4_edge))
        edb = _upsample_like(edb, input1)

        out5_add = torch.add(edb,input1)
        out5d = self.stage5d(out5_add)

        out4_add = torch.add(out5d, input1)
        out4d = self.stage4d(torch.add(out3_edge, out4_add))
        out4d_up = _upsample_like(out4d, input2[0])

        out3_add = torch.add(input2[0], out4d_up)
        out3d = self.stage3d(torch.add(out2_edge, out3_add))
        out3d_up = _upsample_like(out3d, input2[1])

        out2_add = torch.add(input2[1], out3d_up)
        out2d = self.stage2d(torch.add(out1_edge, out2_add))
        out2d_up = _upsample_like(out2d, input2[2])

        out1_add = torch.add(input2[2], out2d_up)
        out1d = self.stage1d(torch.add(out_edge56, out1_add))

        # side output as intermediate prediction
        d0 = self.side1_edge(out_edge)
        d0_edge = _upsample_like(d0, x)

        d1 = self.side1(out1d)
        d1 = _upsample_like(d1, x)

        d2 = self.side2(out2d)
        d2 = _upsample_like(d2, x)

        d3 = self.side3(out3d)
        d3 = _upsample_like(d3, x)

        d4 = self.side4(out4d)
        d4 = _upsample_like(d4, x)

        d_united = torch.cat((d1, d2, d3, d4), 1)
        d_united = self.outconv(d_united)

        return d0_edge, d_united, d1, d2, d3, d4



class EDB(nn.Module):
    def __init__(self, in_ch, ou_ch):
        super(EDB, self).__init__()
        self.conv1 = REBNCONV(128, 128, dirate=1)
        self.conv2 = REBNCONV(128, 128, dirate=1)

        self.max1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, input):
        input1 = self.conv1(input)
        input2 = self.conv2(input1)
        output = self.max1(input2)

        return output


class MEUNet(nn.Module):

    def __init__(self, cfg, out_ch=1):
        super(MEUNet, self).__init__()
        self.cfg = cfg

        self.bkbone = ResNet()

        self.decoder = Decoder(out_ch)

        self.conv5e = nn.Sequential(nn.Conv2d(2048, 128, kernel_size=1), nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv4e = nn.Sequential(nn.Conv2d(1024, 128, kernel_size=1), nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv3e = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1), nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv2e = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1), nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv1e = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1), nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        self.conv0e = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1), nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        self.edb = EDB(128, 128)
        self.CA = ChannelAttention(128)
        self.initialize()

    def forward(self, x, shape=None):
        out0, out1, out2, out3, out4, out5 = self.bkbone(x)

        out5 = self.conv5e(out5)
        out4 = self.conv4e(out4)
        out3 = self.conv3e(out3)
        out2 = self.conv2e(out2)
        out1 = self.conv1e(out1)

        out0 = self.conv0e(out0)

        out_edb = self.edb(out5)
        out_edb = self.CA(out_edb)

        input1 = out5
        input2 = [out4, out3, out2, out1, out0]

        if shape is None:
            shape = x.size()[2:]

        outd_edge, outd_united, outd1, outd2, outd3, outd4 = self.decoder(input1, input2, x, out_edb)

        return outd_edge, outd_united, outd1, outd2, outd3, outd4

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)
