import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath
import torch.utils.checkpoint as cp
import math



def get_reference_points(spatial_shapes, device):
    """
    生成参考点坐标
    """
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            indexing='ij'
        )
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x, H, W):
    """
    为可变形注意力准备输入
    """
    bs, n, c = x.shape
    spatial_shapes = torch.as_tensor([(H // 8, W // 8), (H // 16, W // 16), (H // 32, W // 32)], 
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points(spatial_shapes, x.device)
    deform_inputs = [reference_points, spatial_shapes, level_start_index]
    return deform_inputs


def deform_inputs_only_one(x, H, W):
    """
    为单一尺度可变形注意力准备输入
    """
    bs, n, c = x.shape
    spatial_shapes = torch.as_tensor([(H // 16, W // 16)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points(spatial_shapes, x.device)
    deform_inputs = [reference_points, spatial_shapes, level_start_index]
    return deform_inputs


class ConvFFN(nn.Module):
    """
    卷积前馈网络
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    """
    深度可分离卷积
    """
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        
        if N != H * W:
            sqrt_n = int(N**0.5)
            if sqrt_n * sqrt_n == N:
                h_temp = w_temp = sqrt_n
            else:
                h_temp = int(N**0.5)
                w_temp = N // h_temp
                while h_temp * w_temp != N and h_temp > 1:
                    h_temp -= 1
                    w_temp = N // h_temp
            
            x_reshaped = x.transpose(1, 2).view(B, C, h_temp, w_temp)
            x_reshaped = F.adaptive_avg_pool2d(x_reshaped, (H, W))
        else:
            x_reshaped = x.transpose(1, 2).view(B, C, H, W)
        
        x_conv = self.dwconv(x_reshaped)
        x_conv = x_conv.flatten(2).transpose(1, 2) 
        
        if x_conv.shape[1] != N:
            x_conv_reshaped = x_conv.transpose(1, 2) 
            x_conv_reshaped = F.interpolate(x_conv_reshaped, size=N, mode='linear', align_corners=False)
            x_conv = x_conv_reshaped.transpose(1, 2) 
        
        return x_conv


class MultiDWConv(nn.Module):
    """
    多尺度深度可分离卷积
    基于最佳实践，只使用3x3和5x5两种卷积核
    """
    def __init__(self, dim):
        super(MultiDWConv, self).__init__()
        self.dwconv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv2 = nn.Conv2d(dim, dim, 5, 1, 2, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        if N != H * W:
            sqrt_n = int(N**0.5)
            if sqrt_n * sqrt_n != N:
                h_temp = int(N**0.5)
                w_temp = N // h_temp
                while h_temp * w_temp != N and h_temp > 1:
                    h_temp -= 1
                    w_temp = N // h_temp
            else:
                h_temp = w_temp = sqrt_n
            
            x_reshaped = x.transpose(1, 2).view(B, C, h_temp, w_temp)
            x_reshaped = F.adaptive_avg_pool2d(x_reshaped, (H, W))
        else:
            x_reshaped = x.transpose(1, 2).view(B, C, H, W)
        
        x1 = self.dwconv1(x_reshaped)
        x2 = self.dwconv2(x_reshaped)
        
        x = torch.cat([x1, x2], dim=1)
        x_conv = x.flatten(2).transpose(1, 2) 
        
        # 确保输出序列长度与输入一致
        if x_conv.shape[1] != N:
            x_conv_reshaped = x_conv.transpose(1, 2)  
            x_conv_reshaped = F.interpolate(x_conv_reshaped, size=N, mode='linear', align_corners=False)
            x_conv = x_conv_reshaped.transpose(1, 2) 
        
        return x_conv


class MSCEM(nn.Module):
    def __init__(self, dim, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or dim
        hidden_features = hidden_features or dim
        
        self.fc1 = nn.Linear(dim, hidden_features)
        self.dwconv = MultiDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features * 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MSDeformAttn(nn.Module):
    """
    多尺度可变形注意力
    """
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, ratio=1.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')
        
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)
    
    def forward(self, query, reference_points, input_flatten, spatial_shapes, level_start_index, attention_mask=None):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        
        value = self.value_proj(input_flatten)
        if attention_mask is not None:
            value = value.masked_fill(attention_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, -1)
        
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        
        
        if reference_points.shape[-1] == 2:
            query_h = int(Len_q**0.5)
            query_w = Len_q // query_h
            if query_h * query_w != Len_q:
                factors = []
                for i in range(1, int(Len_q**0.5) + 1):
                    if Len_q % i == 0:
                        factors.append((i, Len_q // i))
                query_h, query_w = min(factors, key=lambda x: abs(x[0] - x[1]))
            
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, query_h - 0.5, query_h, dtype=torch.float32, device=query.device),
                torch.linspace(0.5, query_w - 0.5, query_w, dtype=torch.float32, device=query.device),
                indexing='ij'
            )
            ref_y = ref_y.reshape(-1) / query_h
            ref_x = ref_x.reshape(-1) / query_w
            ref_points_2d = torch.stack((ref_x, ref_y), -1) 
            
            ref_points_expanded = ref_points_2d[None, :, None, :].expand(N, Len_q, self.n_levels, 2)
            
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1).float()
            sampling_locations = ref_points_expanded[:, :, None, :, None, :] + \
                               sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            query_h = int(Len_q**0.5)
            query_w = Len_q // query_h
            if query_h * query_w != Len_q:
                factors = []
                for i in range(1, int(Len_q**0.5) + 1):
                    if Len_q % i == 0:
                        factors.append((i, Len_q // i))
                query_h, query_w = min(factors, key=lambda x: abs(x[0] - x[1]))
            
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, query_h - 0.5, query_h, dtype=torch.float32, device=query.device),
                torch.linspace(0.5, query_w - 0.5, query_w, dtype=torch.float32, device=query.device),
                indexing='ij'
            )
            ref_y = ref_y.reshape(-1) / query_h
            ref_x = ref_x.reshape(-1) / query_w
            ref_w = torch.ones_like(ref_x) * 0.1 
            ref_h = torch.ones_like(ref_y) * 0.1  
            ref_points_4d = torch.stack((ref_x, ref_y, ref_w, ref_h), -1) 
            
            ref_points_expanded = ref_points_4d[None, :, None, :].expand(N, Len_q, self.n_levels, 4)
            sampling_locations = ref_points_expanded[:, :, None, :, None, :2] + \
                               sampling_offsets / self.n_points * ref_points_expanded[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError('Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        
        output = self._ms_deform_attn_core_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        
        return output
    
    def _ms_deform_attn_core_pytorch(self, value, value_spatial_shapes, sampling_locations, attention_weights):
        """
        多尺度可变形注意力的PyTorch核心实现
        """
        N_, S_, M_, D_ = value.shape
        _, Lq_, M_, L_, P_, _ = sampling_locations.shape
        
        expected_total = sum([H_ * W_ for H_, W_ in value_spatial_shapes])
        actual_total = value.shape[1]
        
        if expected_total != actual_total:
            feat_h = int(actual_total**0.5)
            feat_w = actual_total // feat_h
            if feat_h * feat_w != actual_total:
                factors = []
                for i in range(1, int(actual_total**0.5) + 1):
                    if actual_total % i == 0:
                        factors.append((i, actual_total // i))
                feat_h, feat_w = min(factors, key=lambda x: abs(x[0] - x[1]))
            
            value_spatial_shapes = torch.tensor([[feat_h, feat_w]], dtype=torch.long, device=value.device)
            L_ = 1  
        
        value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
        
        sampling_grids = 2 * sampling_locations - 1
        
        sampling_value_list = []
        for lid_, (H_, W_) in enumerate(value_spatial_shapes):
            value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
            sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
            sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear',
                                              padding_mode='zeros', align_corners=False)
            sampling_value_list.append(sampling_value_l_)

        attention_weights_transposed = attention_weights.transpose(1, 2)  
        
        actual_L = attention_weights_transposed.shape[3]
        actual_P = attention_weights_transposed.shape[4]
        
        target_shape = (N_ * M_, 1, Lq_, actual_L * actual_P)
        attention_weights = attention_weights_transposed.reshape(target_shape)
        
        if len(sampling_value_list) > 0:
            target_shape = sampling_value_list[0].shape
            for i in range(len(sampling_value_list)):
                if sampling_value_list[i].shape != target_shape:
                    sampling_value_list[i] = F.interpolate(
                        sampling_value_list[i], 
                        size=target_shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
        
        if len(sampling_value_list) == 1:
            stacked_values = sampling_value_list[0].unsqueeze(-2) 
        else:
            stacked_values = torch.stack(sampling_value_list, dim=-2)
        
        if len(stacked_values.shape) == 5: 
            stacked_values = stacked_values.permute(0, 1, 2, 4, 3)
            stacked_values = stacked_values.flatten(-2)
        else: 
            pass
        
        
        if stacked_values.shape[-1] != attention_weights.shape[-1]:
            target_size = stacked_values.shape[-1]
            attention_weights = F.interpolate(
                attention_weights.squeeze(1).unsqueeze(1),  
                size=(attention_weights.shape[-2], target_size),
                mode='bilinear',
                align_corners=False
            )
        
        output = (stacked_values * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
        
        return output.transpose(1, 2).contiguous()


class MultiscaleExtractor(nn.Module):
    """
    多尺度特征提取器
    """
    def __init__(self, dim, n_levels=3, num_heads=6, n_points=4, 
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0., with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                n_points=n_points, ratio=deform_ratio)
        
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        def _inner_forward(query, feat):
            B, N_query, C = query.shape
            
            attn = self.attn(self.query_norm(query), reference_points,
                           self.feat_norm(feat), spatial_shapes,
                           level_start_index, None)
            
            if attn.shape[1] != N_query:
                attn_reshaped = attn.transpose(1, 2) 
                attn_reshaped = F.interpolate(attn_reshaped, size=N_query, mode='linear', align_corners=False)
                attn = attn_reshaped.transpose(1, 2)  
            
            query = query + attn
            
            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            
            return query
        
        return _inner_forward(query, feat)


class SFIM_toV(nn.Module):
    """
    CNN到ViT的特征交互模块
    """
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., 
                 with_cp=False, drop=0., drop_path=0., cffn_ratio=0.25):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                               n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        
        self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
        self.ffn_norm = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        def _inner_forward(query, feat, H, W):
            B, N, C = feat.shape
            c1 = self.attn(self.query_norm(feat), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)

            c1 = c1 + self.drop_path(self.ffn(self.ffn_norm(c1), H, W)) 

            if N >= H*W*6: 
                c_select1, c_select2, c_select3 = c1[:,:H*W*4, :], c1[:, H*W*4:H*W*4+H*W, :], c1[:, H*W*4+H*W:, :]
                
                if c_select3.shape[1] > 0:
                    c_select1 = F.interpolate(c_select1.permute(0,2,1).reshape(B, C, H*2, W*2), scale_factor=0.5, mode='bilinear', align_corners=False).flatten(2).permute(0,2,1)
                    c_select3 = F.interpolate(c_select3.permute(0,2,1).reshape(B, C, H//2, W//2), scale_factor=2, mode='bilinear', align_corners=False).flatten(2).permute(0,2,1)
                    return query + self.gamma * (c_select1 + c_select2 + c_select3)
                else:
                    c_select1 = F.interpolate(c_select1.permute(0,2,1).reshape(B, C, H*2, W*2), scale_factor=0.5, mode='bilinear', align_corners=False).flatten(2).permute(0,2,1)
                    return query + self.gamma * (c_select1 + c_select2)
            else:
                 if c1.shape[1] != query.shape[1]:
                     c1_reshaped = c1.transpose(1, 2) 
                     c1_reshaped = F.interpolate(c1_reshaped, size=query.shape[1], mode='linear', align_corners=False)
                     c1 = c1_reshaped.transpose(1, 2)  
                 return query + self.gamma * c1
        
        if self.with_cp and query.requires_grad:
            return cp.checkpoint(_inner_forward, query, feat, H, W)
        else:
            return _inner_forward(query, feat, H, W)


class SFIM_toC(nn.Module):
    """
    ViT到CNN的特征交互模块
    """
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False,
                 cnn_feature_interaction=True):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        
        self.cnn_feature_interaction = cnn_feature_interaction
        if cnn_feature_interaction:
            self.cfinter = MultiscaleExtractor(dim=dim, n_levels=3, num_heads=num_heads,
                                             n_points=n_points, norm_layer=norm_layer,
                                             deform_ratio=deform_ratio, with_cffn=with_cffn,
                                             cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path,
                                             with_cp=with_cp)
    
    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        def _inner_forward(query, feat, H, W):
            B, N, C = query.shape
            n = N // 21
            x1 = query[:, 0:16 * n, :].contiguous()
            x2 = query[:, 16 * n:20 * n, :].contiguous()
            x3 = query[:, 20 * n:, :].contiguous()
            if feat.shape[1] != x2.shape[1]:
                feat_reshaped = feat.transpose(1, 2) 
                feat_reshaped = F.interpolate(feat_reshaped, size=x2.shape[1], mode='linear', align_corners=False)
                feat = feat_reshaped.transpose(1, 2) 
            x2 = x2 + feat
            query = torch.cat([x1, x2, x3], dim=1)

            if self.cnn_feature_interaction:               
                deform_input = deform_inputs_only_one(query, H*16, W*16)
                query = self.cfinter(query=self.query_norm(query), reference_points=deform_input[0],
                          feat=self.feat_norm(query), spatial_shapes=deform_input[1],
                          level_start_index=deform_input[2],
                          H=H, W=W)               
            
            return query
        
        return _inner_forward(query, feat, H, W)


class Extractor_SFIM(nn.Module):
    """
    额外的SFIM提取器模块
    """
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False,
                 cnn_feature_interaction=True):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.cnn_feature_interaction = cnn_feature_interaction
        if cnn_feature_interaction:
            self.cfinter = MultiscaleExtractor(dim=dim, n_levels=3, num_heads=num_heads,
                                             n_points=n_points, norm_layer=norm_layer,
                                             deform_ratio=deform_ratio, with_cffn=with_cffn,
                                             cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path,
                                             with_cp=with_cp)
    
    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        def _inner_forward(query, feat, H, W):
            B, N, C = query.shape
            n = N // 21
            x1 = query[:, 0:16 * n, :].contiguous()
            x2 = query[:, 16 * n:20 * n, :].contiguous()
            x3 = query[:, 20 * n:, :].contiguous()
            x2 = x2 + feat
            query = torch.cat([x1, x2, x3], dim=1)

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))

            if self.cnn_feature_interaction:
                deform_input = deform_inputs_only_one(query, H*16, W*16)
                query = self.cfinter(query=self.query_norm(query), reference_points=deform_input[0],
                          feat=self.feat_norm(query), spatial_shapes=deform_input[1],
                          level_start_index=deform_input[2],
                          H=H, W=W)
            
            return query
        
        if self.with_cp and query.requires_grad:
            return cp.checkpoint(_inner_forward, query, feat, H, W)
        else:
            return _inner_forward(query, feat, H, W)


class SFIMBlock(nn.Module):
    """
    SFIM交互块
    """
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_SFIM=False, with_cp=False,
                 use_SFIM_toV=True, use_SFIM_toC=True, cnn_feature_interaction=False,
                 extra_num=4):
        super().__init__()
        
        self.use_SFIM_toV = use_SFIM_toV
        self.use_SFIM_toC = use_SFIM_toC
        
        if use_SFIM_toV:
            self.sfim_tov = SFIM_toV(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp, drop=drop, drop_path=drop_path, cffn_ratio=cffn_ratio)
            self.mscem = MSCEM(dim, hidden_features=int(dim * 0.5))
        
        if use_SFIM_toC:
            self.sfim_toc = SFIM_toC(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                 norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                 cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp,
                                 cnn_feature_interaction=cnn_feature_interaction)
        
        if extra_SFIM:
            self.extra_SFIMs = nn.Sequential(*[
                Extractor_SFIM(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                             norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp,
                             cnn_feature_interaction=cnn_feature_interaction)
                for _ in range(extra_num)
            ])
        else:
            self.extra_SFIMs = None
    
    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W):
        B, N, C = x.shape
        deform_inputs = deform_inputs_only_one(x, H*16, W*16)
        
        if self.use_SFIM_toV:
            c = self.mscem(c, H, W)
            c_select1, c_select2, c_select3 = c[:,:H*W*4, :], c[:, H*W*4:H*W*4+H*W, :], c[:, H*W*4+H*W:, :]
            
            if c_select2.shape[1] != x.shape[1]:
                c_select2_reshaped = c_select2.transpose(1, 2) 
                c_select2_reshaped = F.interpolate(c_select2_reshaped, size=x.shape[1], mode='linear', align_corners=False)
                c_select2 = c_select2_reshaped.transpose(1, 2)  
            
            c = torch.cat([c_select1, c_select2 + x, c_select3], dim=1)

            x = self.sfim_tov(query=x, reference_points=deform_inputs[0],
                          feat=c, spatial_shapes=deform_inputs[1],
                          level_start_index=deform_inputs[2], H=H, W=W)

        for idx, blk in enumerate(blocks):
            x = blk(x)

        if self.use_SFIM_toC:
            c = self.sfim_toc(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
                           
        if self.extra_SFIMs is not None:
            for sfim in self.extra_SFIMs:
                c = sfim(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        
        return x, c

class CNN(nn.Module):
    """
    CNN并行分支 - MSCEM模块
    """
    def __init__(self, inplanes=64, embed_dim=384):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * inplanes),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
            nn.ReLU(inplace=True)
        )
        
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
    
    def forward(self, x):
        c1 = self.stem(x)    
        c2 = self.conv2(c1)    
        c3 = self.conv3(c2)    
        c4 = self.conv4(c3)  
        
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)
        
        bs, dim, _, _ = c1.shape
        c2 = c2.view(bs, dim, -1).transpose(1, 2)
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  
        c4 = c4.view(bs, dim, -1).transpose(1, 2) 
        
        return c1, c2, c3, c4

