import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.util.conv_1_1 import conv1x1


class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

#  Attention


def get_input(x, direction):
    if direction == "x":
        x = x.permute(0, 2, 4, 1, 3)
    elif direction == "y":
        x = x.permute(0, 2, 3, 1, 4)  # N, W, C, H
    else:  # is seq
        x = x.permute(0, 3, 4, 1, 2)  # N, W, C, H
    return x


def get_output(output, direction):
    if direction == "x":
        output = output.permute(0, 3, 1, 4, 2)
    elif direction == "y":
        output = output.permute(0, 3, 1, 2, 4)  # N, W, C, H
    else:  # is seq
        output = output.permute(0, 3, 4, 1, 2)  # N, W, C, H
    return output


class AxialAttentionWithoutPosition3D(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, direction="x"):  # direction can be x, y , seq
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttentionWithoutPosition3D, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.direction = direction

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups )

        self.bn_output = nn.BatchNorm1d(out_planes * 1)

        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        x = get_input(x, self.direction)
        N, seq, W, C, H = x.shape
        x = x.contiguous().view(N * seq * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * seq * W, self.groups, self.group_planes * 2, H),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        stacked_similarity = self.bn_similarity(qk).reshape(N * seq * W, 1, self.groups, H, H).sum(dim=1).contiguous()

        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)

        sv = sv.reshape(N * seq * W, self.out_planes * 1, H).contiguous()
        output = self.bn_output(sv).reshape(N, seq, W, self.out_planes, 1, H).sum(dim=-2).contiguous()

        output = get_output(output, self.direction)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))


class AxialAttentionWithPosition3D(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, direction="x"):  # direction can be x, y , seq
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttentionWithPosition3D, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.direction = direction

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)

        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        x = get_input(x, self.direction)
        N, seq, W, C, H = x.shape
        x = x.contiguous().view(N * seq * W, C, H)

        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * seq * W, self.groups, self.group_planes * 2, H),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2,
                                                                                       self.kernel_size,
                                                                                       self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.group_planes // 2, self.group_planes // 2,
                                                             self.group_planes], dim=0)

        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)

        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * seq * W, 3, self.groups, H, H).sum(dim=1)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * seq * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, seq, W, self.out_planes, 2, H).sum(dim=-2)

        output = get_output(output, self.direction)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class AxialAttentionWithPositionAndGate3D(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, direction="x"):  # direction can be x, y , seq
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttentionWithPositionAndGate3D, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.direction = direction

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        self.f_qr = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_kr = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_sve = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_sv = nn.Parameter(torch.tensor(1.0),  requires_grad=False)

        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        x = get_input(x, self.direction)
        N, seq, W, C, H = x.shape
        x = x.contiguous().view(N * seq * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * seq * W, self.groups, self.group_planes * 2, H),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2,
                                                                                       self.kernel_size,
                                                                                       self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.group_planes // 2, self.group_planes // 2,
                                                             self.group_planes], dim=0)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        # multiply by factors
        qr = torch.mul(qr, self.f_qr)
        kr = torch.mul(kr, self.f_kr)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * seq * W, 3, self.groups, H, H).sum(dim=1)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

        # multiply by factors
        sv = torch.mul(sv, self.f_sv)
        sve = torch.mul(sve, self.f_sve)

        stacked_output = torch.cat([sv, sve], dim=-1).view(N * seq * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, seq, W, self.out_planes, 2, H).sum(dim=-2)

        output = get_output(output, self.direction)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

#  Attention Block ***********************


class Axial3DWithoutPositionBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, down_sample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(Axial3DWithoutPositionBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        self.conv_down = conv1x1(inplanes, width)
        self.conv1 = nn.Conv2d(width, width, kernel_size=1)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttentionWithoutPosition3D(width, width, groups=groups, kernel_size=kernel_size,
                                                           direction="x")
        self.width_block = AxialAttentionWithoutPosition3D(width, width, groups=groups, kernel_size=kernel_size,
                                                           direction="y")
        self.seq_block = AxialAttentionWithoutPosition3D(width, width, groups=groups, kernel_size=kernel_size,
                                                         stride=stride, direction="seq")
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.seq_block(out)

        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class Axial3DPositionBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, down_sample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(Axial3DPositionBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.height_block = AxialAttentionWithPosition3D(width, width, groups=groups, kernel_size=kernel_size,
                                                         direction="x")
        self.width_block = AxialAttentionWithPosition3D(width, width, groups=groups, kernel_size=kernel_size,
                                                        direction="y")
        self.seq_block = AxialAttentionWithPosition3D(width, width, groups=groups, kernel_size=kernel_size,
                                                      stride=stride, direction="sqe")
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.height_block(out)
        out = self.width_block(out)
        out = self.seq_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class Axial3DPositionGateBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, down_sample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(Axial3DPositionGateBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.height_block = AxialAttentionWithPositionAndGate3D(width, width, groups=groups, kernel_size=kernel_size,
                                                                direction="x")
        self.width_block = AxialAttentionWithPositionAndGate3D(width, width, groups=groups, kernel_size=kernel_size,
                                                               stride=stride, direction="Y")
        self.seq_block = AxialAttentionWithPositionAndGate3D(width, width, groups=groups, kernel_size=kernel_size,
                                                             stride=stride, direction="seq")
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.height_block(out)
        out = self.width_block(out)
        out = self.seq_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


