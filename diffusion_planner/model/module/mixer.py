import torch.nn as nn
from timm.models.layers import Mlp


# 这是一个MLP-Mixer模块的实现，它是Vision Transformer的一个变体，专门用于处理序列数据。让我详细解释：

# 核心设计理念
# MLP-Mixer 放弃了传统的注意力机制，纯粹使用MLP来混合信息，分为两个维度的混合：

# Token Mixing：在序列长度维度上混合信息
# Channel Mixing：在特征维度上混合信息

class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, drop_path_rate):
        super().__init__()

        self.norm1 = nn.LayerNorm(channels_mlp_dim) # 用于token mixing前
          # Channel MLP：处理特征维度
        self.channels_mlp = Mlp(in_features=channels_mlp_dim, hidden_features=channels_mlp_dim, act_layer=nn.GELU, drop=drop_path_rate)
        self.norm2 = nn.LayerNorm(channels_mlp_dim) # 用于channel mixing前
        # Token MLP：处理序列维度  
        self.tokens_mlp = Mlp(in_features=tokens_mlp_dim, hidden_features=tokens_mlp_dim, act_layer=nn.GELU, drop=drop_path_rate)
        
    def forward(self, x):
            # 输入 x: (batch_size, sequence_length, feature_dim)
    
    # === Token Mixing 阶段 ===
        y = self.norm1(x)           # LayerNorm归一化
        y = y.permute(0, 2, 1)      # 转置: (B, feature_dim, sequence_length)
        y = self.tokens_mlp(y)      # 在序列维度上应用MLP
        y = y.permute(0, 2, 1)      # 转回: (B, sequence_length, feature_dim)
        x = x + y                   # 残差连接

            # === Channel Mixing 阶段 ===
        y = self.norm2(x)
        return x + self.channels_mlp(y)   # 在特征维度上应用MLP + 残差连接
    





# 关键特性
# 1. 双重混合机制
# Token Mixing: 通过转置让MLP在序列维度上操作，使不同位置的token能够交换信息
# Channel Mixing: 在特征维度上操作，增强特征表达能力
# 2. 残差连接
# 3. LayerNorm归一化
# 在每个MLP操作前进行归一化，稳定训练
# 在自动驾驶中的应用
# 对于扩散规划器，这个模块可能用于：

# 多智能体信息融合: 在agent序列维度混合信息
# 时序特征处理: 在时间步维度上混合信息
# 多模态特征融合: 在不同特征类型间混合信息
# 优势
# 计算效率: 比Self-Attention更高效，复杂度为O(n)而非O(n²)
# 简单有效: 架构简单但表达能力强
# 可扩展性: 易于扩展到大规模序列
# 这种设计特别适合处理自动驾驶中的多元素序列数据（如多个智能体、多个时间步、多种特征类型），能有效地在不同维度间传递和融合信息。

