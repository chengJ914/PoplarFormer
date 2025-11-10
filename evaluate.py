import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import classification_report, confusion_matrix

# 导入自定义分类器
from model import CustomClassifierEVAPoplar, get_leaf_transforms
from coral_utils import CoralLoss, coral_ordinal_regression, extract_numeric_value_for_coral


def get_args_parser():
    parser = argparse.ArgumentParser('PoplarFormer 杨树叶片病斑分类器评估', add_help=False)
    
    # 模型参数
    parser.add_argument('--model_path', default='./output_eva02/best_model.pth', type=str,
                        help='模型路径')
    parser.add_argument('--input_size', default=336, type=int,
                        help='输入图像大小 (默认: 336)')
    
    # 评估参数
    parser.add_argument('--batch_size', default=64, type=int,
                        help='批量大小 (默认: 64)')
    parser.add_argument('--data_path', default='../data_new', type=str,
                        help='数据集路径')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='输出目录，如果为None则根据模型路径自动生成')
    
    # 其他参数
    parser.add_argument('--num_workers', default=8, type=int,
                        help='数据加载线程数 (默认: 8)')
    parser.add_argument('--device', default='cuda',
                        help='设备 (默认: cuda)')
    
    return parser


# 使用统一的extract_numeric_value_for_coral函数，保持与训练脚本一致
extract_numeric_value = extract_numeric_value_for_coral


def plot_confusion_matrix(cm, class_names, output_path):
    """
    绘制混淆矩阵，保存为PDF格式
    """
    # 设置字体为Times New Roman，全局字体大小设为25
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 25
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 25})  
    plt.xlabel('Predicted Label', fontfamily='Times New Roman', fontsize=25, labelpad=20)
    plt.ylabel('True Label', fontfamily='Times New Roman', fontsize=25, labelpad=20)
    plt.title('Confusion Matrix', fontfamily='Times New Roman', fontsize=25, pad=25)
    
    plt.xticks(fontsize=25, rotation=0, ha='center')
    plt.yticks(fontsize=25, rotation=90, va='center')
    
    plt.tight_layout(pad=2.5)
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.3)
    plt.close()


def load_model(model_path, device):
    """
    加载模型
    """
    print(f"加载模型: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    except Exception as e:
        print(f"警告: 使用weights_only=True加载失败，回退到默认方式: {e}")
        checkpoint = torch.load(model_path, map_location='cpu')
    
    # 获取类别数量
    if 'num_classes' in checkpoint:
        num_classes = checkpoint['num_classes']
    else:
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 查找分类器的权重形状 - CORAL模型使用classifier而不是head
        classifier_keys = [k for k in state_dict.keys() if 'classifier' in k and 'weight' in k]
        if classifier_keys:
            classifier_weight = state_dict[classifier_keys[-1]]
            # 对于CORAL模型，分类器输出是num_classes-1
            num_classes = classifier_weight.shape[0] + 1
        else:
            print("警告: 无法确定类别数量，使用默认值 (5)")
            num_classes = 5
    
    # 获取类别名称
    if 'class_names' in checkpoint:
        class_names = checkpoint['class_names']
    else:
        class_names = [f'class_{i}' for i in range(num_classes)]
        print("警告: 检查点中未找到类别名称，使用默认名称")
    
    # 设置预训练权重路径
    local_weights = "../pretrained/eva02_small_patch14_336.mim_in22k_ft_in1"
    
    # 检查预训练权重文件
    if not os.path.exists(local_weights):
        raise FileNotFoundError(
            f"错误: 预训练权重文件不存在: {local_weights}\n"
        )
    
    print(f"找到本地预训练权重: {local_weights}")
    print("正在加载预训练权重...")
    
    # 创建PoplarFormer模型
    model = CustomClassifierEVAPoplar(
        num_classes=num_classes,
        dropout_rate=0.0,
        pretrained=True,
        local_weights_path=local_weights
    )
    
    # 加载权重
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, num_classes, class_names


def calculate_class_mae(targets, predictions, class_names):
    """
    计算每个类别的MAE
    """
    class_mae = {}
    class_counts = {}
    
    # 初始化每个类别的MAE和计数
    for class_name in class_names:
        class_mae[class_name] = 0.0
        class_counts[class_name] = 0
    
    # 计算每个样本的MAE并按类别累加
    for i in range(len(targets)):
        true_class = class_names[targets[i]]
        pred_class = class_names[predictions[i]]
        true_value = extract_numeric_value(true_class)
        pred_value = extract_numeric_value(pred_class)
        mae = abs(true_value - pred_value)
        
        class_mae[true_class] += mae
        class_counts[true_class] += 1
    
    # 计算每个类别的平均MAE
    for class_name in class_names:
        if class_counts[class_name] > 0:
            class_mae[class_name] /= class_counts[class_name]
        else:
            class_mae[class_name] = 0.0
    
    return class_mae


def evaluate_model(model, dataloader, device, class_names):
    """
    评估模型
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    total_loss = 0.0
    correct = 0
    total = 0
    total_mae = 0.0
    
    # 使用CORAL损失函数
    criterion = CoralLoss(num_classes=len(class_names))
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="评估中"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # 前向传播
            logits = model(images)
            loss = criterion(logits, targets)
            
            # 使用CORAL预测
            probs = coral_ordinal_regression(logits)
            preds = torch.argmax(probs, dim=1)
            
            # 计算MAE - 使用统一的数值映射
            batch_mae = 0.0
            for i in range(len(targets)):
                true_class = class_names[targets[i].item()]
                pred_class = class_names[preds[i].item()]
                true_value = extract_numeric_value(true_class)
                pred_value = extract_numeric_value(pred_class)
                batch_mae += abs(true_value - pred_value)
            total_mae += batch_mae / len(targets)
            
            total_loss += loss.item()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    mae = total_mae / len(dataloader)
    
    # 计算每个类别的MAE
    class_mae = calculate_class_mae(all_targets, all_preds, class_names)
    
    return np.array(all_preds), np.array(all_targets), avg_loss, accuracy, mae, class_mae


def main(args):
    # 设置设备
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if str(device).startswith('cuda'):
        device_idx = torch.cuda.current_device() if torch.cuda.is_available() else None
    print(f"使用设备: {device}")
    
    # 设置输出目录
    if args.output_dir is None:
        model_dir = os.path.dirname(args.model_path)
        args.output_dir = os.path.join(model_dir, 'evaluation')
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    model, num_classes, class_names = load_model(args.model_path, device)
    print(f"模型加载完成，类别数量: {num_classes}")
    print(f"类别名称: {class_names}")
    
    # 获取数据变换
    transform = get_leaf_transforms(args.input_size, is_training=False)
    
    # 加载测试集
    test_dir = os.path.join(args.data_path, 'test')
    if not os.path.exists(test_dir):
        print(f"错误: 测试集目录不存在: {test_dir}")
        return
    
    # 创建测试数据集
    test_dataset = ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"测试集: {len(test_dataset)} 个样本")
    print(f"测试集类别: {test_dataset.classes}")
    
    # 验证类别一致性
    if test_dataset.classes != class_names:
        print("警告: 测试集类别与模型训练时的类别不一致")
        print(f"模型类别: {class_names}")
        print(f"测试集类别: {test_dataset.classes}")
    
    # 评估模型
    all_preds, all_targets, avg_loss, accuracy, mae, class_mae = evaluate_model(
        model, test_loader, device, class_names
    )
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(cm, class_names, os.path.join(args.output_dir, 'confusion_matrix.pdf'))
    
    # 生成分类报告
    report = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    print("\n分类报告:")
    print(report)
    
    # 新增：打印每类MAE
    print("\n各类别MAE:")
    for class_name, mae_val in class_mae.items():
        print(f"  {class_name}: {mae_val:.4f}")
    
    # 保存分类报告（包含MAE信息）
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
        f.write("\n\n各类别MAE:\n")
        for class_name, mae_val in class_mae.items():
            f.write(f"  {class_name}: {mae_val:.4f}\n")
    
    # 打印结果
    print(f"\n评估结果:")
    print(f"平均损失: {avg_loss:.4f}")
    print(f"准确率: {accuracy * 100:.2f}%")
    print(f"MAE (平均绝对误差): {mae:.4f}")
    
    # 保存评估结果
    results = {
        'accuracy': float(accuracy),
        'mae': float(mae),
        'avg_loss': float(avg_loss),
        'class_mae': class_mae,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }
    
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n评估完成！结果保存在: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('评估脚本', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

