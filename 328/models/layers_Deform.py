from models.doconv_pytorch import *
from torchvision.ops import DeformConv2d


# =============================================
# 可变形卷积模块 (Deformable Convolution Modules)
# =============================================

class DeformableConv2d(nn.Module):
    """
    可变形卷积封装类
    包含 offset 预测网络和可变形卷积层
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False, groups=1):
        super(DeformableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # offset 预测网络：预测每个采样点的偏移量 (2 * kernel_size * kernel_size 个值)
        self.offset_conv = nn.Conv2d(
            in_channel,
            2 * kernel_size * kernel_size,  # x和y方向的偏移
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )

        # 可选的 modulation mask (DCNv2)
        self.mask_conv = nn.Conv2d(
            in_channel,
            kernel_size * kernel_size,  # 每个采样点的权重
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )

        # 可变形卷积层
        self.deform_conv = DeformConv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups
        )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        # offset 初始化为0，使得初始时等价于标准卷积
        nn.init.constant_(self.offset_conv.weight, 0)  # 修正：去掉多余的空格
        nn.init.constant_(self.offset_conv.bias, 0)  # 修正：去掉多余的空格
        nn.init.constant_(self.mask_conv.weight, 0)  # 修正：去掉多余的空格
        nn.init.constant_(self.mask_conv.bias, 0)  # 修正：去掉多余的空格

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        out = self.deform_conv(x, offset, mask)
        return out


class DeformableConv2d_Simple(nn.Module):
    """
    简化版可变形卷积 (DCNv1，不带 mask)
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False, groups=1):
        super(DeformableConv2d_Simple, self).__init__()
        self.kernel_size = kernel_size

        # offset 预测网络
        self.offset_conv = nn.Conv2d(
            in_channel,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )

        # 可变形卷积层
        self.deform_conv = DeformConv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups
        )

        # 初始化 offset 为 0
        nn.init.constant_(self.offset_conv.weight, 0)  # 修正：去掉多余的空格
        nn.init.constant_(self.offset_conv.bias, 0)  # 修正：去掉多余的空格

    def forward(self, x):
        offset = self.offset_conv(x)
        out = self.deform_conv(x, offset)  # 修正：去掉多余的空格
        return out


# =============================================
# 基于可变形卷积的 BasicConv 模块
# =============================================

class BasicConv_Deform(nn.Module):
    """
    使用可变形卷积的基础卷积模块 (训练模式)
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, bias=False, norm=False, relu=True,
                 transpose=False,
                 relu_method=nn.ReLU, groups=1, norm_method=nn.BatchNorm2d, use_dcnv2=True):
        super(BasicConv_Deform, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()

        if transpose:
            # 转置卷积不使用可变形卷积
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            # 使用可变形卷积替代普通卷积
            if kernel_size >= 3:  # 可变形卷积通常用于 3x3 或更大的卷积核
                if use_dcnv2:
                    layers.append(
                        DeformableConv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride,
                                         bias=bias, groups=groups))
                else:
                    layers.append(  # 修正：去掉多余的空格
                        DeformableConv2d_Simple(in_channel, out_channel, kernel_size, padding=padding, stride=stride,
                                                bias=bias, groups=groups))
            else:
                # 1x1 卷积使用普通卷积
                layers.append(  # 修正：去掉多余的空格
                    nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                              groups=groups))

        if norm:
            layers.append(norm_method(out_channel))
        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(relu_method())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BasicConv_Deform_eval(nn.Module):
    """
    使用可变形卷积的基础卷积模块 (评估模式)
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, bias=False, norm=False, relu=True,
                 transpose=False,
                 relu_method=nn.ReLU, groups=1, norm_method=nn.BatchNorm2d, use_dcnv2=True):
        super(BasicConv_Deform_eval, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()

        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            if kernel_size >= 3:
                if use_dcnv2:
                    layers.append(
                        DeformableConv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride,
                                         bias=bias, groups=groups))
                else:
                    layers.append(  # 修正：去掉多余的空格
                        DeformableConv2d_Simple(in_channel, out_channel, kernel_size, padding=padding, stride=stride,
                                                bias=bias, groups=groups))
            else:
                layers.append(
                    nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                              groups=groups))

        if norm:
            layers.append(norm_method(out_channel))
        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(relu_method())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)  # 修正：去掉多余的空格


# =============================================
# 基于可变形卷积的 ResBlock 模块
# =============================================

class ResBlock_Deform(nn.Module):
    """
    使用可变形卷积的残差块
    """

    def __init__(self, out_channel):
        super(ResBlock_Deform, self).__init__()
        self.main = nn.Sequential(
            BasicConv_Deform(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_Deform(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class ResBlock_Deform_fft_bench(nn.Module):
    """
    使用可变形卷积 + FFT 的残差块 (训练模式)
    """

    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_Deform_fft_bench, self).__init__()
        # 空间域使用可变形卷积
        self.main = nn.Sequential(
            BasicConv_Deform(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_Deform(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        # 频域分支保持 1x1 卷积
        self.main_fft = nn.Sequential(
            BasicConv_Deform(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=True),
            BasicConv_Deform(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=False)
        )
        self.dim = out_channel
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y


class ResBlock_Deform_fft_bench_eval(nn.Module):
    """
    使用可变形卷积 + FFT 的残差块 (评估模式)
    """

    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_Deform_fft_bench_eval, self).__init__()
        self.main = nn.Sequential(
            BasicConv_Deform_eval(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_Deform_eval(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.main_fft = nn.Sequential(
            BasicConv_Deform_eval(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=True),
            BasicConv_Deform_eval(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=False)
        )
        self.dim = out_channel
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y


# =============================================
# 原有模块保持不变
# =============================================

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False,
                 channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                                   groups=groups))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                          groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        elif relu:
            layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BasicConv_do(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, bias=False, norm=False, relu=True,
                 transpose=False,
                 relu_method=nn.ReLU, groups=1, norm_method=nn.BatchNorm2d):
        super(BasicConv_do, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(  # 修正：去掉多余的空格
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(  # 修正：去掉多余的空格
                DOConv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                         groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))  # 修正：去掉多余的空格
            else:
                layers.append(relu_method())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BasicConv_do_eval(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False,
                 relu_method=nn.ReLU, groups=1, norm_method=nn.BatchNorm2d):
        super(BasicConv_do_eval, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                DOConv2d_eval(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                              groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))  # 修正：去掉多余的空格
            else:
                layers.append(relu_method())  # 修正：去掉多余的空格
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)  # 修正：去掉多余的空格


###########################################
###########################################
###########################################
class ResBlock(nn.Module):
    def __init__(self, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=True, norm=False),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False, norm=False)
        )

    def forward(self, x):
        return self.main(x) + x


class ResBlock_do(nn.Module):
    def __init__(self, out_channel):
        super(ResBlock_do, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class ResBlock_do_eval(nn.Module):
    def __init__(self, out_channel):
        super(ResBlock_do_eval, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do_eval(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do_eval(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


###########################################
###########################################
###########################################


class ResBlock_fft_bench(nn.Module):
    def __init__(self, n_feat, norm='backward'):
        super(ResBlock_fft_bench, self).__init__()
        self.main = nn.Sequential(
            BasicConv(n_feat, n_feat, kernel_size=3, stride=1, relu=True),
            BasicConv(n_feat, n_feat, kernel_size=3, stride=1, relu=False)
        )
        self.main_fft = nn.Sequential(
            BasicConv(n_feat * 2, n_feat * 2, kernel_size=1, stride=1, relu=True),
            BasicConv(n_feat * 2, n_feat * 2, kernel_size=1, stride=1, relu=False)
        )
        self.dim = n_feat
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape  # 修正：去掉多余的空格
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)  # 修正：去掉多余的空格
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)  # 修正：去掉多余的空格
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)  # 修正：去掉多余的空格
        return self.main(x) + x + y


class ResBlock_do_fft_bench(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_do_fft_bench, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.main_fft = nn.Sequential(
            BasicConv_do(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=True),
            BasicConv_do(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=False)
        )
        self.dim = out_channel
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y


class ResBlock_do_fft_bench_eval(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_do_fft_bench_eval, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do_eval(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do_eval(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.main_fft = nn.Sequential(
            BasicConv_do_eval(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=True),
            BasicConv_do_eval(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=False)
        )
        self.dim = out_channel
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y


###########################################
###########################################
###########################################


class ResBlock_do_nofft_bench(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_do_nofft_bench, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.main_fft = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=1, stride=1, relu=False)
        )
        self.dim = out_channel
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = self.main_fft(x)
        return self.main(x) + x + y


class ResBlock_do_nofft_bench_eval(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_do_nofft_bench_eval, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do_eval(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do_eval(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.main_fft = nn.Sequential(
            BasicConv_do_eval(out_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv_do_eval(out_channel, out_channel, kernel_size=1, stride=1, relu=False)
        )
        self.dim = out_channel
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = self.main_fft(x)  # 修正：去掉多余的空格
        return self.main(x) + x + y


###########################################
###########################################
###########################################


def window_partitions(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows


def window_reverses(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    C = windows.shape[1]
    x = windows.view(-1, H // window_size, W // window_size, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    return x


def window_partitionx(x, window_size):
    _, _, H, W = x.shape
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_partitions(x[:, :, :h, :w], window_size)  # 修正：去掉多余的空格
    b_main = x_main.shape[0]
    if h == H and w == W:
        return x_main, [b_main]
    if h != H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)  # 修正：去掉多余的空格
        b_r = x_r.shape[0] + b_main
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)  # 修正：去掉多余的空格
        b_d = x_d.shape[0] + b_r
        x_dd = x[:, :, -window_size:, -window_size:]  # 修正：去掉多余的空格
        b_dd = x_dd.shape[0] + b_d
        return torch.cat([x_main, x_r, x_d, x_dd], dim=0), [b_main, b_r, b_d, b_dd]
    if h == H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)  # 修正：去掉多余的空格
        b_r = x_r.shape[0] + b_main
        return torch.cat([x_main, x_r], dim=0), [b_main, b_r]
    if h != H and w == W:
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)  # 修正：去掉多余的空格
        b_d = x_d.shape[0] + b_main
        return torch.cat([x_main, x_d], dim=0), [b_main, b_d]  # 修正：去掉多余的空格


def window_reversex(windows, window_size, H, W, batch_list):
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_reverses(windows[:batch_list[0], ...], window_size, h, w)  # 修正：去掉多余的空格
    B, C, _, _ = x_main.shape
    res = torch.zeros([B, C, H, W], device=windows.device)  # 修正：去掉多余的空格
    res[:, :, :h, :w] = x_main
    if h == H and w == W:
        return res
    if h != H and w != W and len(batch_list) == 4:
        x_dd = window_reverses(windows[batch_list[2]:, ...], window_size, window_size, window_size)  # 修正：去掉多余的空格
        res[:, :, h:, w:] = x_dd[:, :, h - H:, w - W:]  # 修正：去掉多余的空格
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)  # 修正：去掉多余的空格
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]  # 修正：去掉多余的空格
        x_d = window_reverses(windows[batch_list[1]:batch_list[2], ...], window_size, window_size, w)  # 修正：去掉多余的空格
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]  # 修正：去掉多余的空格
        return res
    if w != W and len(batch_list) == 2:
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)  # 修正：去掉多余的空格
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]  # 修正：去掉多余的空格
    if h != H and len(batch_list) == 2:
        x_d = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, window_size, w)  # 修正：去掉多余的空格
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]  # 修正：去掉多余的空格
    return res