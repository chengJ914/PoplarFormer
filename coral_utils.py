import json
import torch
import torch.nn as nn


class CoralLoss(nn.Module):
    def __init__(self, num_classes=5, weight=None):
        super(CoralLoss, self).__init__()
        self.num_classes = num_classes
        self.weight = weight
        # 使用二元交叉熵作为基础损失函数
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits, targets):
        # 将目标类别索引转换为二进制标签矩阵
        extended_targets = self._extend_targets(targets)
        
        # 计算每个二分类任务的损失
        loss = self.bce(logits, extended_targets)
        
        # 如果提供了权重，则应用权重
        if self.weight is not None:
            # 获取每个样本对应的权重
            sample_weights = self.weight[targets]
            loss = loss * sample_weights.unsqueeze(1)
        
        # 返回平均损失
        return loss.mean()
    
    def _extend_targets(self, targets):
        batch_size = targets.size(0)
        extended_targets = torch.zeros(batch_size, self.num_classes - 1, device=targets.device)
        
        for i in range(batch_size):
            target = targets[i]
            # 对于类别k，前k个位置为1，其余为0
            if target > 0:  # 避免索引越界
                extended_targets[i, :target] = 1
        
        return extended_targets


def coral_ordinal_regression(logits):
    # 计算每个阈值的概率
    probs = torch.sigmoid(logits)
    
    # 计算每个类别的概率
    n_classes = logits.shape[1] + 1
    probs_class = torch.zeros(logits.shape[0], n_classes, device=logits.device)
    
    # 第一个类别的概率
    probs_class[:, 0] = 1 - probs[:, 0]
    
    # 中间类别的概率
    for k in range(1, n_classes - 1):
        probs_class[:, k] = probs[:, k-1] - probs[:, k]
    
    # 最后一个类别的概率
    probs_class[:, -1] = probs[:, -1]
    
    # 修正负概率问题：钳位到[0, 1]范围
    probs_class = torch.clamp(probs_class, min=0.0, max=1.0)
    
    # 重新归一化确保概率和为1
    probs_sum = probs_class.sum(dim=1, keepdim=True)
    probs_class = probs_class / (probs_sum + 1e-8) 
    
    return probs_class



def save_class_names(class_names, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(class_names, f, indent=4)


def load_class_names(input_path, num_classes=None):
    with open(input_path, 'r', encoding='utf-8') as f:
        class_names = json.load(f)
    
    # 如果指定了类别数量，则检查是否匹配
    if num_classes is not None and len(class_names) != num_classes:
        print(f"警告: 类别名称数量 ({len(class_names)}) 与指定的类别数量 ({num_classes}) 不匹配")
    
    return class_names


def extract_numeric_value_for_coral(class_name):
    """
    从类别名称中提取数值用于CORAL MAE计算
    使用简单的1-5映射，与CORAL损失函数保持一致
    """
    if '1_Very_Mild' in class_name:
        return 1.0
    elif '2_Mild' in class_name:
        return 2.0
    elif '3_Mild_Moderate' in class_name:
        return 3.0
    elif '4_Moderate' in class_name:
        return 4.0
    elif '5_Severe' in class_name:
        return 5.0
    else:
        # 尝试从字符串中提取数字作为备选方案
        import re
        numbers = re.findall(r'\d+', class_name)
        if numbers:
            return float(numbers[0])
        return 0.0