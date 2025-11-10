import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import os

try:
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_
    from timm.models.vision_transformer import VisionTransformer
except ImportError:
    print("错误: timm模块导入失败，请确保已正确安装")
    raise

try:
    from modules import (
        CNN, SFIMBlock, deform_inputs, deform_inputs_only_one,
        ConvFFN, MSCEM, MSDeformAttn
    )
    from coral_utils import CoralLoss, coral_ordinal_regression
except ImportError as e:
    print(f"错误: 无法导入自定义模块: {e}")
    raise


class EVAPoplarBlock(nn.Module):
    """
    PoplarFormer 混合块
    在EVA02 Transformer Block基础上集成SFIM交互机制
    """
    def __init__(self, original_block, dim, num_heads=6, n_points=4, 
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), drop=0., drop_path=0.,
                 with_cffn=True, cffn_ratio=0.25, init_values=0., deform_ratio=1.0,
                 use_SFIM_toV=True, use_SFIM_toC=True, cnn_feature_interaction=False):
        super().__init__()
        
        self.original_block = original_block
        
        # SFIM交互模块
        self.sfim_block = SFIMBlock(
            dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
            drop=drop, drop_path=drop_path, with_cffn=with_cffn, cffn_ratio=cffn_ratio,
            init_values=init_values, deform_ratio=deform_ratio, use_SFIM_toV=use_SFIM_toV,
            use_SFIM_toC=use_SFIM_toC, cnn_feature_interaction=cnn_feature_interaction
        )
        
        self.use_sfim = use_SFIM_toV or use_SFIM_toC
    
    def forward(self, x, c=None, H=None, W=None):
        if self.use_sfim and c is not None and H is not None and W is not None:
            # 准备可变形注意力输入
            deform_inputs1 = deform_inputs(x, H*16, W*16)
            deform_inputs2 = deform_inputs_only_one(c, H*16, W*16)
            
            # SFIM交互
            x, c = self.sfim_block(x, c, [self.original_block], deform_inputs1, deform_inputs2, H, W)
        else:
            # 仅使用原始EVA02块
            x = self.original_block(x)
        
        return x, c


class EVAPoplar(nn.Module):
    """
    PoplarFormer 混合架构模型
    融合EVA02-small主干网络与MSCEM和SFIM模块
    """
    def __init__(self, num_classes=5, dropout_rate=0.1, pretrained=True, 
                 local_weights_path=None, img_size=336, patch_size=14, embed_dim=384,
                 depth=12, num_heads=6, mlp_ratio=2.6667, qkv_bias=True, drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 # SFIM相关参数
                 conv_inplane=64, n_points=4, deform_num_heads=6, init_values=0.,
                 interaction_indexes=[0, 3, 6, 9],  # 4次交互，将12个Block分为4个Stage
                  with_cffn=True, cffn_ratio=0.25,
                 deform_ratio=1.0, add_vit_feature=True, use_extra_SFIM=False,
                 use_SFIM_toV=True, use_SFIM_toC=True, cnn_feature_interaction=True):
        super().__init__()

        if pretrained and not local_weights_path:
            raise ValueError(
                "错误: pretrained=True 时必须提供 local_weights_path 参数。\n"
            )

        if not pretrained:
            raise ValueError(
                "错误: pretrained=False 不允许使用。\n"
            )
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        
        # 创建EVA02主干网络
        import timm
        self.eva_backbone = timm.create_model(
            'eva02_small_patch14_336.mim_in22k_ft_in1k',
            pretrained=False,
            num_classes=0,
            drop_rate=drop_rate
        )
        
        # 更新embed_dim为实际的特征维度
        self.embed_dim = self.eva_backbone.num_features
        
        # CNN并行分支 (MSCEM)
        self.spm = CNN(inplanes=conv_inplane, embed_dim=embed_dim)
        
        # SFIM交互模块
        self.interactions = nn.ModuleList()
        for i in range(len(interaction_indexes)):
            layer = EVAPoplarBlock(
                original_block=self.eva_backbone.blocks[interaction_indexes[i]],
                dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                norm_layer=norm_layer, drop=drop_rate, drop_path=drop_path_rate,
                with_cffn=with_cffn, cffn_ratio=cffn_ratio, init_values=init_values,
                deform_ratio=deform_ratio, use_SFIM_toV=use_SFIM_toV, use_SFIM_toC=use_SFIM_toC,
                cnn_feature_interaction=cnn_feature_interaction
            )
            self.interactions.append(layer)
        
        # 特征融合层
        if add_vit_feature:
            self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
            self.spm_norm = norm_layer(embed_dim)
        
        # CORAL分类头
        self.norm = norm_layer(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(embed_dim, num_classes - 1)
        
        # 加载预训练权重
        if pretrained:
            self._load_pretrained_weights(local_weights_path)
        
        # 初始化权重
        self._init_weights()
    
    def _load_pretrained_weights(self, local_weights_path=None):
        if not local_weights_path:
            raise ValueError(
                "请设置 local_weights_path 参数指向预训练权重文件。"
            )
        
        if not os.path.exists(local_weights_path):
            raise FileNotFoundError(
                "请确保权重文件路径正确。"
            )
        
        print(f"正在从本地加载预训练权重: {local_weights_path}")
        try:
            # 加载预训练权重
            state_dict = torch.load(local_weights_path, map_location='cpu')
            
            # 使用 strict=True 确保所有权重都必须加载
            try:
                missing_keys, unexpected_keys = self.eva_backbone.load_state_dict(state_dict, strict=True)
                
                # 统计加载情况
                total_params = len(state_dict)
                loaded_params = total_params - len(missing_keys)
                
                print(f"预训练权重加载完成:")
                print(f"  - 总参数数: {total_params}")
                print(f"  - 成功加载: {loaded_params}")
                print(f"  - 跳过参数: {len(unexpected_keys)}")
                
                if unexpected_keys:
                    print(f"  跳过的参数: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
                    
            except RuntimeError as e:
                missing_keys, unexpected_keys = self.eva_backbone.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    error_msg = (
                        f"错误: 预训练权重加载不完整！\n"
                        f"缺失的参数数量: {len(missing_keys)}\n"
                        f"缺失的参数列表: {missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}\n"
                    )
                    raise RuntimeError(error_msg) from e
                else:
                    raise
                    
        except Exception as e:
            error_msg = (
                f"错误: 加载预训练权重失败: {e}\n"
            )
            raise RuntimeError(error_msg) from e
    
    def _resize_pos_embed(self, pos_embed, target_length):
        """调整位置嵌入的长度"""
        if pos_embed.shape[1] == target_length:
            return pos_embed
        
        B, N, D = pos_embed.shape
        pos_embed_resized = F.interpolate(
            pos_embed.transpose(1, 2),  
            size=target_length,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)  
        
        return pos_embed_resized
    
    def _init_weights(self):
        """初始化新增模块的权重"""
        if hasattr(self, 'level_embed'):
            trunc_normal_(self.level_embed, std=.02)
        
        if hasattr(self.classifier, 'weight'):
            trunc_normal_(self.classifier.weight, std=.02)
            nn.init.constant_(self.classifier.bias, 0)
        
        # 初始化SFIM模块中的gamma参数为0
        for module in self.modules():
            if hasattr(module, 'gamma') and isinstance(module.gamma, nn.Parameter):
                nn.init.constant_(module.gamma, 0.)
    
    def forward_features(self, x):
        """特征提取前向传播"""
        B, C, H, W = x.shape
        
        # ViT主干分支 - Patch Embedding
        x_vit = self.eva_backbone.patch_embed(x)
        
        # 处理位置嵌入
        if hasattr(self.eva_backbone, 'pos_embed') and self.eva_backbone.pos_embed is not None:
            if x_vit.shape[1] != self.eva_backbone.pos_embed.shape[1]:
                pos_embed = self.eva_backbone.pos_embed
                if pos_embed.shape[1] == x_vit.shape[1] + 1: 
                    pos_embed = pos_embed[:, 1:, :] 
                
                if pos_embed.shape[1] != x_vit.shape[1]:
                    pos_embed = self._resize_pos_embed(pos_embed, x_vit.shape[1])
                
                x_vit = x_vit + pos_embed
            else:
                x_vit = x_vit + self.eva_backbone.pos_embed
        
        x_vit = self.eva_backbone.pos_drop(x_vit)
        
        # CNN并行分支 - MSCEM
        c1, c2, c3, c4 = self.spm(x)
        # 保存原始特征的长度，用于后续分割
        c2_len, c3_len, c4_len = c2.shape[1], c3.shape[1], c4.shape[1]
        c = torch.cat([c2, c3, c4], dim=1)  # 拼接多尺度特征
        
        # 计算特征图尺寸
        patch_H, patch_W = H // self.patch_size[0], W // self.patch_size[1]
        
        # 逐层处理，在指定位置插入SFIM交互
        interaction_idx = 0
        for i, blk in enumerate(self.eva_backbone.blocks):
            if i in self.interaction_indexes and interaction_idx < len(self.interactions):
                # 注意：这里使用patch_H和patch_W，但在某些情况下可能需要调整
                x_vit, c = self.interactions[interaction_idx](x_vit, c, patch_H, patch_W)
                interaction_idx += 1
            else:
                x_vit = blk(x_vit)
                if i == len(self.eva_backbone.blocks) - 1:
                    if hasattr(self, 'check_feature_dim') and c.shape[-1] != self.embed_dim:
                        pass
        
        # 最终归一化
        x_vit = self.eva_backbone.norm(x_vit)
        
        # 特征融合
        if self.add_vit_feature:
            # 使用保存的长度信息分割特征
            c2 = c[:, :c2_len, :]
            c3 = c[:, c2_len:c2_len+c3_len, :]
            c4 = c[:, c2_len+c3_len:, :]
            
            # 添加level embedding
            c2 = c2 + self.level_embed[0]
            c3 = c3 + self.level_embed[1] 
            c4 = c4 + self.level_embed[2]
            
            # 归一化
            c2 = self.spm_norm(c2)
            c3 = self.spm_norm(c3)
            c4 = self.spm_norm(c4)
            
            c2_len_new, c3_len_new, c4_len_new = c2.shape[1], c3.shape[1], c4.shape[1]
            c2_len, c3_len, c4_len = c2_len_new, c3_len_new, c4_len_new
            
            def get_hw(length):
                h = int(length**0.5)
                w = length // h
                if h * w != length:
                    factors = [(i, length // i) for i in range(1, int(length**0.5) + 1) if length % i == 0]
                    h, w = min(factors, key=lambda x: abs(x[0] - x[1]))
                return h, w
            
            c2_h, c2_w = get_hw(c2_len)
            c3_h, c3_w = get_hw(c3_len)
            c4_h, c4_w = get_hw(c4_len)
            
            c2_2d = c2.transpose(1, 2).view(B, self.embed_dim, c2_h, c2_w)
            c3_2d = c3.transpose(1, 2).view(B, self.embed_dim, c3_h, c3_w)
            c4_2d = c4.transpose(1, 2).view(B, self.embed_dim, c4_h, c4_w)
            
            # 上采样到统一尺寸
            c3_up = F.interpolate(c3_2d, size=(c2_h, c2_w), mode='bilinear', align_corners=False)
            c4_up = F.interpolate(c4_2d, size=(c2_h, c2_w), mode='bilinear', align_corners=False)
            
            # 融合CNN特征
            c_fused = c2_2d + c3_up + c4_up
            c_fused = c_fused.flatten(2).transpose(1, 2)
            
            # 与ViT特征融合
            if c_fused.shape[1] == x_vit.shape[1]:
                x_vit = x_vit + c_fused
            else:
                c_fused_pooled = F.adaptive_avg_pool1d(c_fused.transpose(1, 2), x_vit.shape[1]).transpose(1, 2)
                x_vit = x_vit + c_fused_pooled
        
        return x_vit
    
    def forward(self, x):
        """完整前向传播"""
        features = self.forward_features(x)
        
        # 全局平均池化
        if len(features.shape) == 3: 
            features = features.mean(dim=1)  
        
        # 分类头
        features = self.norm(features)
        features = self.dropout(features)
        logits = self.classifier(features)
        
        return logits
    
    def get_classifier(self):
        return self.classifier
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.embed_dim, num_classes - 1 if num_classes > 1 else 1)


class CustomClassifierEVAPoplar(nn.Module):
    """
    自定义PoplarFormer分类器
    与原始CustomClassifierEVA02接口保持一致
    """
    def __init__(self, num_classes=5, dropout_rate=0.1, pretrained=True, local_weights_path=None):
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = "PoplarFormer"
        
        # 创建PoplarFormer模型
        self.model = EVAPoplar(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            pretrained=pretrained,
            local_weights_path=local_weights_path,
            img_size=336,
            patch_size=14,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=2.6667,
            # SFIM参数
            conv_inplane=64,
            n_points=4,
            deform_num_heads=6,
            init_values=0.0,
            interaction_indexes=[0, 3, 6, 9],  
            with_cffn=True,
            cffn_ratio=0.25,
            deform_ratio=1.0,
            add_vit_feature=True,
            use_SFIM_toV=True,
            use_SFIM_toC=True,
            cnn_feature_interaction=True
        )
        
        print(f"创建 {self.model_name} 模型:")
        print(f"  - 类别数: {num_classes}")
        print(f"  - Dropout率: {dropout_rate}")
        print(f"  - 预训练: {pretrained}")
        print(f"  - SFIM交互位置: {[0, 3, 6, 9]}")
        print(f"  - 初始化值: {0.0} (保护预训练权重)")
    
    def forward(self, x):
        return self.model(x)
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'architecture': 'EVA02-small + MSCEM + SFIM'
        }



from torchvision import transforms


def get_leaf_transforms(input_size=336, is_training=True):
    """
    获取叶片图像的数据变换
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize(int(input_size * 1.1)),
            transforms.CenterCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(input_size * 1.1)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


if __name__ == "__main__":
    # 测试模型创建
    print("测试PoplarFormer模型创建...")
    
    # 测试权重路径
    test_weights_path = "../pretrained/eva02_small_patch14_336.mim_in22k_ft_in1"
    
    if not os.path.exists(test_weights_path):
        print(f"警告: 测试权重文件不存在: {test_weights_path}")
    else:
        model = CustomClassifierEVAPoplar(
            num_classes=5,
            dropout_rate=0.1,
            pretrained=True,
            local_weights_path=test_weights_path
        )
        
        # 打印模型信息
        info = model.get_model_info()
        print("\n模型信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 测试前向传播
        x = torch.randn(2, 3, 336, 336)
        try:
            with torch.no_grad():
                output = model(x)
            print(f"\n前向传播测试成功!")
            print(f"输入形状: {x.shape}")
            print(f"输出形状: {output.shape}")
            print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        except Exception as e:
            print(f"\n前向传播测试失败: {e}")

