import torch
import torch.nn as nn
from torch.nn import functional as F
import sys

# 当前模块，用于动态引用本文件中的类或函数
current_module = sys.modules[__name__]


# =========================================
# FBCNet 主体网络
# =========================================
class FBCNet(nn.Module):
    """
    FBCNet（Filter Bank Convolutional Network）
    基于 FBCSP 思想的网络结构：
    - 进行空间卷积（SCB）
    - 然后在时间维上计算特征（例如方差或对数方差）
    - 最后通过全连接层分类
    
    输入数据格式： (batch, 1, chan, time, filterBand)
    """

    def SCB(self, m, nChan, nBands, doWeightNorm=True, *args, **kwargs):
        """
        构建空间卷积块（Spatial Convolution Block）
        参数:
            m: 每个滤波器组的空间滤波器数量
            nChan: 通道数（电极数）
            nBands: 滤波器组数量（滤波频带）
            doWeightNorm: 是否启用权重归一化
        """
        return nn.Sequential(
            # 分组卷积: 每个频带单独卷积，不在频带间共享权重
            Conv2dWithConstraint(
                in_channels=nBands,
                out_channels=m * nBands,
                kernel_size=(nChan, 1),     # 跨所有通道
                groups=nBands,              # 每个频带单独卷积
                max_norm=2,                 # 权重范数约束
                doWeightNorm=doWeightNorm,
                padding=0
            ),
            nn.BatchNorm2d(m * nBands),    # 批归一化
            swish()                        # 非线性激活函数
        )

    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        """
        最后的全连接分类模块
        输入特征 -> LogSoftmax 输出分类概率
        """
        return nn.Sequential(
            LinearWithConstraint(
                in_features=inF,
                out_features=outF,
                max_norm=0.5,
                doWeightNorm=doWeightNorm,
                *args, **kwargs
            ),
            nn.LogSoftmax(dim=1)  # 对数概率输出 (多分类常用)
        )

    def __init__(self, nChan, nTime, nClass=2, nBands=2, m=32,
                 temporalLayer='LogVarLayer', strideFactor=2, doWeightNorm=True, *args, **kwargs):
        """
        模型初始化函数
        参数：
            nChan: 通道数
            nTime: 时间长度
            nClass: 分类类别数
            nBands: 滤波频带数
            m: 每个频带的空间滤波器数
            temporalLayer: 时间聚合层名称（如 LogVarLayer）
            strideFactor: 时间分块因子
            doWeightNorm: 是否启用权重归一化
        """
        super(FBCNet, self).__init__()

        self.nBands = nBands
        self.m = m
        self.strideFactor = strideFactor

        # 构建空间卷积块
        self.scb = self.SCB(m, nChan, self.nBands, doWeightNorm=doWeightNorm)

        # 动态实例化时间层
        # 从当前模块命名空间取出 temporalLayer 类（如 LogVarLayer）
        self.temporalLayer = current_module.__dict__[temporalLayer](dim=3)

        # 构建最终全连接层
        self.lastLayer = self.LastBlock(
            inF=self.m * self.nBands * self.strideFactor,
            outF=nClass,
            doWeightNorm=doWeightNorm
        )

    def forward(self, x):
        """
        前向传播过程
        输入形状: (batch, 1, chan, time, band)
        """
        x = x.squeeze()                  # 去除单通道维度 -> (batch, band, chan, time)
        x = self.scb(x)                  # 空间卷积 + BN + 激活

        # 时间分块处理，例如 strideFactor=2 时，将时间维分为两个时间窗口
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3] / self.strideFactor)])

        # 计算时间维度上的对数方差
        x = self.temporalLayer(x)

        # 展平为向量
        x = torch.flatten(x, start_dim=1)

        # 全连接层输出分类结果
        x = self.lastLayer(x)
        return x


# =========================================
# FBCNet 改进版（更模块化）
# =========================================
class FBCNet_2(nn.Module):
    def __init__(self,
                 n_classes,
                 input_shape,
                 m,
                 temporal_stride,
                 weight_init_method=None,
                 ):
        """
        参数:
            n_classes: 类别数
            input_shape: 输入形状 (batch, n_band, n_electrode, time_points)
            m: 每个频带的空间滤波器数
            temporal_stride: 时间分块数
            weight_init_method: 权重初始化方式
        """
        super().__init__()
        self.temporal_stride = temporal_stride

        batch_size, n_band, n_electrode, time_points = input_shape

        # --- 空间卷积块 (SCB) ---
        self.scb = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=n_band,
                out_channels=m * n_band,
                kernel_size=(n_electrode, 1),
                groups=n_band,       # 每个频带独立
                max_norm=2
            ),
            nn.BatchNorm2d(m * n_band),
            swish()
        )

        # --- 时间统计层（对数方差层） ---
        self.temporal_layer = LogVarLayer(dim=-1)

        # --- 分类层（全连接层）---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(
                in_features=n_band * m * temporal_stride,
                out_features=n_classes,
                max_norm=0.5
            )
        )

        # --- 权重初始化 ---
        initialize_weight(self, weight_init_method)

    def forward(self, x):
        """
        前向传播
        输入形状: (batch, n_band, n_electrode, time_points)
        """
        out = self.scb(x)

        # 将时间维按 temporal_stride 分块
        out = out.reshape([*out.shape[:2], self.temporal_stride, int(out.shape[-1] / self.temporal_stride)])

        # 计算每个时间块的 log(var)
        out = self.temporal_layer(out)

        # 扁平化并通过全连接层分类
        out = self.classifier(out)
        return out


# =========================================
# 带有权重范数约束的卷积层
# =========================================
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        """
        带约束的二维卷积：
        在每次 forward 前对权重进行 renorm 限制
        """
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            # 重新归一化权重模长，防止梯度爆炸/塌陷
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


# =========================================
# 带有权重范数约束的全连接层
# =========================================
class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


# =========================================
# swish 激活函数层
# f(x) = x * sigmoid(x)
# =========================================
class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


# =========================================
# LogVarLayer: 计算给定维度上的对数方差
# 常用于 EEG 等生物信号的能量特征提取
# =========================================
class LogVarLayer(nn.Module):
    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        # 防止数值不稳定，使用 clamp 限定范围
        return torch.log(torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-6, 1e6))


# =========================================
# initialize_weight: 模型权重初始化函数
# =========================================
def initialize_weight(model, method):
    """
    根据指定方法初始化模型权重
    可选方法：
        - 'normal'：正态分布（均值0，标准差0.01）
        - 'xavier_uni'：Xavier 均匀分布
        - 'xavier_normal'：Xavier 正态分布
        - 'he_uni'：He 均匀分布
        - 'he_normal'：He 正态分布
    """
    method = dict(
        normal=['normal_', dict(mean=0, std=0.01)],
        xavier_uni=['xavier_uniform_', dict()],
        xavier_normal=['xavier_normal_', dict()],
        he_uni=['kaiming_uniform_', dict()],
        he_normal=['kaiming_normal_', dict()]
    ).get(method)

    if method is None:
        return None

    for module in model.modules():
        # 若为 LSTM 层，按权重类型单独初始化
        if module.__class__.__name__ in ['LSTM']:
            for param in module._all_weights[0]:
                if param.startswith('weight'):
                    getattr(nn.init, method[0])(getattr(module, param), **method[1])
                elif param.startswith('bias'):
                    nn.init.constant_(getattr(module, param), 0)
        else:
            if hasattr(module, "weight"):
                # 非 BatchNorm 层
                if not ("BatchNorm" in module.__class__.__name__):
                    getattr(nn.init, method[0])(module.weight, **method[1])
                # BatchNorm 权重初始化为 1
                else:
                    nn.init.constant_(module.weight, 1)
                # 偏置项初始化为 0
                if hasattr(module, "bias"):
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
