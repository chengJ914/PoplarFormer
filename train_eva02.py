import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import math
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from eva_comer_model import CustomClassifierEVACoMer, get_leaf_transforms
from coral_utils import (
    CoralLoss, 
    coral_ordinal_regression, 
    save_class_names, 
    extract_numeric_value_for_coral
)


def get_args_parser():
    parser = argparse.ArgumentParser('PoplarFormer杨树叶片病斑分类器训练', add_help=False)

    # 模型参数
    parser.add_argument('--num_classes', default=5, type=int,
                        help='类别数量')
    parser.add_argument('--input_size', default=336, type=int,
                        help='输入图像大小 (默认: 336)')

    # 训练参数
    parser.add_argument('--batch_size', default=32, type=int,
                        help='物理批量大小 (默认: 32)')
    parser.add_argument('--accumulation_steps', default=2, type=int,
                        help='梯度累积步数 (默认: 2, 逻辑batch_size = batch_size * accumulation_steps)')
    parser.add_argument('--epochs', default=50, type=int,
                        help='训练轮次数 (默认: 50)')
    parser.add_argument('--lr', default=2e-5, type=float,
                        help='EVA02主干网络学习率 (默认: 2e-5, 差分学习率策略)')
    parser.add_argument('--comer_lr', default=2e-4, type=float,
                        help='新增模块学习率 (默认: 2e-4, 差分学习率策略)')
    parser.add_argument('--use_differential_lr', action='store_true', default=True,
                        help='是否使用差分学习率策略 (默认: True)')
    parser.add_argument('--weight_decay', default=0.05, type=float,
                        help='权重衰减 (默认: 0.05)')
    parser.add_argument('--dropout_rate', default=0.1, type=float,
                        help='Dropout比率 (默认: 0.1)')
    parser.add_argument('--max_grad_norm', default=1.0, type=float,
                        help='梯度裁剪的最大范数 (默认: 1.0)')

    # 数据参数
    parser.add_argument('--data_path', default='../data_new', type=str,
                        help='数据集路径')
    parser.add_argument('--output_dir', default='./output_eva02', type=str,
                        help='输出目录')
    parser.add_argument('--save_all_epochs', action='store_true', default=False,
                        help='是否保存每个轮次的模型权重')
    parser.add_argument('--val_split', default=0.2, type=float,
                        help='验证集分割比例，当验证集不存在时使用 (默认: 0.2)')
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='是否使用类别权重处理不平衡问题')

    # 其他参数
    parser.add_argument('--num_workers', default=4, type=int,
                        help='数据加载线程数 (默认: 8)')
    parser.add_argument('--device', default='cuda',
                        help='设备 (默认: cuda)')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='是否使用混合精度训练')

    # 添加恢复训练的参数
    parser.add_argument('--resume', default=None, type=str,
                        help='恢复训练的检查点路径')

    return parser


def extract_numeric_value(class_name):
    """
    从类别名称中提取数值用于MAE计算
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


def calculate_coral_mae(targets, predictions, class_names):
    """
    计算CORAL模型的MAE
    """
    total_mae = 0.0
    class_mae = {}
    class_counts = {}
    
    # 初始化
    for class_name in class_names:
        class_mae[class_name] = 0.0
        class_counts[class_name] = 0
    
    # 计算MAE
    for i in range(len(targets)):
        true_class = class_names[targets[i]]
        pred_class = class_names[predictions[i]]
        
        true_value = extract_numeric_value_for_coral(true_class)
        pred_value = extract_numeric_value_for_coral(pred_class)
        
        mae = abs(true_value - pred_value)
        total_mae += mae
        
        class_mae[true_class] += mae
        class_counts[true_class] += 1
    
    # 计算平均MAE
    avg_mae = total_mae / len(targets)
    
    for class_name in class_names:
        if class_counts[class_name] > 0:
            class_mae[class_name] /= class_counts[class_name]
    
    return avg_mae, class_mae


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch, args, class_names, use_amp=True, accumulation_steps=2, max_grad_norm=1.0):
    """
    训练一个轮次，支持梯度累积和混合精度训练
    """
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    progress_bar = tqdm(dataloader, desc=f'训练 Epoch {epoch+1}')
    
    # 梯度累积相关变量
    accumulated_loss = 0.0
    # 滑动平均loss用于进度条显示
    smoothed_loss = 0.0
    alpha = 0.1  # 滑动平均系数
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)
                # 梯度累积：损失需要除以累积步数
                loss = loss / accumulation_steps
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            # 梯度累积：损失需要除以累积步数
            loss = loss / accumulation_steps
        
        # 反向传播
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        accumulated_loss += loss.item()
        
        # 梯度累积：每accumulation_steps步或最后一个batch时更新参数
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            if use_amp:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            
            optimizer.zero_grad()
            scheduler.step()
        
        # 计算预测结果
        with torch.no_grad():
            probs = coral_ordinal_regression(logits)
            preds = torch.argmax(probs, dim=1)
        
        # 统计
        # 只在累积步骤完成时更新total_loss
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            total_loss += accumulated_loss * accumulation_steps  # 恢复原始损失大小
            accumulated_loss = 0.0
        
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
        # 更新滑动平均loss
        current_loss = loss.item() * accumulation_steps
        if batch_idx == 0:
            smoothed_loss = current_loss
        else:
            smoothed_loss = alpha * current_loss + (1 - alpha) * smoothed_loss
        
        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{smoothed_loss:.4f}',  # 显示滑动平均loss
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    # 计算指标
    num_updates = len(dataloader) // accumulation_steps + (1 if len(dataloader) % accumulation_steps != 0 else 0)
    avg_loss = total_loss / num_updates
    accuracy = 100. * correct / total
    mae, class_mae = calculate_coral_mae(all_targets, all_preds, class_names)
    
    return {
        'loss': avg_loss,
        'acc': accuracy,
        'mae': mae,
        'class_mae': class_mae
    }


def validate(model, dataloader, criterion, device, class_names):
    """
    验证模型
    """
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='验证'):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # 前向传播
            logits = model(images)
            loss = criterion(logits, targets)
            
            # 计算预测结果
            probs = coral_ordinal_regression(logits)
            preds = torch.argmax(probs, dim=1)
            
            # 统计
            total_loss += loss.item()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 计算指标
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    mae, class_mae = calculate_coral_mae(all_targets, all_preds, class_names)
    
    return {
        'loss': avg_loss,
        'acc': accuracy,
        'mae': mae,
        'class_mae': class_mae
    }


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, train_maes, val_maes, output_dir):
    """
    绘制训练和验证的损失、准确率和MAE曲线，保存为PDF格式
    """
    epochs = range(1, len(train_losses) + 1)

    # 绘制综合曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_maes, 'b-', label='Training MAE')
    plt.plot(epochs, val_maes, 'r-', label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    # 单独保存损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    # 单独保存准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_curve.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    
    # 单独保存MAE曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_maes, 'b-', label='Training MAE')
    plt.plot(epochs, val_maes, 'r-', label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'mae_curve.pdf'), format='pdf', bbox_inches='tight')
    plt.close()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    余弦退火学习率调度器
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def main(args):
    # 记录开始时间
    total_start_time = time.time()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存参数
    with open(os.path.join(args.output_dir, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    
    # 获取数据变换
    train_transform = get_leaf_transforms(args.input_size, is_training=True)
    val_transform = get_leaf_transforms(args.input_size, is_training=False)
    
    # 加载训练集
    train_dir = os.path.join(args.data_path, 'train')
    if not os.path.exists(train_dir):
        print(f"错误: 训练集目录不存在: {train_dir}")
        return
    
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    class_names = train_dataset.classes
    print(f"类别名称: {class_names}")
    print(f"类别数量: {len(class_names)}")
    
    # 保存类别名称
    save_class_names(class_names, os.path.join(args.output_dir, 'class_names.json'))
    
    # 检查验证集
    val_dir = os.path.join(args.data_path, 'val')
    if os.path.exists(val_dir):
        val_dataset = ImageFolder(val_dir, transform=val_transform)
        print(f"使用独立验证集: {len(val_dataset)} 个样本")
    else:
        # 分割训练集创建验证集
        print(f"未找到验证集，从训练集分割 {args.val_split*100:.0f}% 作为验证集")
        train_size = int((1 - args.val_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"训练集: {len(train_dataset)} 个样本")
        print(f"验证集: {len(val_dataset)} 个样本")
    
    # 计算类别权重
    if args.use_class_weights:
        if hasattr(train_dataset, 'targets'):
            train_targets = train_dataset.targets
        else:
            train_targets = [train_dataset.dataset.targets[i] for i in train_dataset.indices]
        
        class_counts = torch.bincount(torch.tensor(train_targets))
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        print(f"类别权重: {class_weights}")
    else:
        class_weights = None
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 打印训练配置信息
    logical_batch_size = args.batch_size * args.accumulation_steps
    print(f"\n=== 训练配置 ===")
    print(f"物理批量大小: {args.batch_size}")
    print(f"梯度累积步数: {args.accumulation_steps}")
    print(f"逻辑批量大小: {logical_batch_size}")
    print(f"梯度裁剪范数: {args.max_grad_norm}")
    print(f"混合精度训练: {'启用' if args.use_amp else '禁用'}")
    print(f"学习率策略: {'差分学习率' if args.use_differential_lr else '统一学习率'}")
    if args.use_differential_lr:
        print(f"  - EVA02主干网络学习率: {args.lr}")
        print(f"  - 新增模块学习率: {args.comer_lr}")
    else:
        print(f"  - 统一学习率: {args.lr}")
    
    # 创建混合模型
    print("\n创建模型...")
    local_weights = "../pretrained/eva02_small_patch14_336.mim_in22k_ft_in1k/pytorch_model.bin"
    
    # 检查预训练权重文件
    if os.path.exists(local_weights):
        print(f"找到本地预训练权重: {local_weights}")
        print("正在加载预训练权重...")
    else:
        print(f"未找到本地预训练权重: {local_weights}")
        print("将使用在线下载的预训练权重")
    
    model = CustomClassifierEVACoMer(
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        pretrained=True,
        local_weights_path=local_weights if os.path.exists(local_weights) else None
    )
    model = model.to(device)
    
    # 打印模型信息
    model_info = model.get_model_info()
    print(f"\n使用模型: {model_info['model_name']}")
    print(f"模型架构: {model_info['architecture']}")
    print(f"总参数量: {model_info['total_params']:,}")
    print(f"可训练参数: {model_info['trainable_params']:,}")
    print(f"参数利用率: {model_info['trainable_params']/model_info['total_params']*100:.2f}%")
    
    # CORAL损失函数
    criterion = CoralLoss(
        num_classes=args.num_classes,
        weight=class_weights.to(device) if class_weights is not None else None
    )
    
    # 差分学习率策略
    def get_parameter_groups(model, backbone_lr=2e-5, new_modules_lr=2e-4, weight_decay=0.05):
        """
        为不同模块设置不同的学习率
        backbone_lr: EVA02主干网络学习率 (2e-5)
        new_modules_lr: 新增模块学习率 (2e-4)
        """
        backbone_params = []
        new_module_params = []
        
        # EVA02主干网络参数 (较低学习率)
        if hasattr(model, 'model') and hasattr(model.model, 'eva_backbone'):
            for name, param in model.model.eva_backbone.named_parameters():
                if param.requires_grad:
                    backbone_params.append(param)
        
        # 新增模块参数 (较高学习率)
        for name, param in model.named_parameters():
            if param.requires_grad:
                # 跳过已经在backbone中的参数
                if 'eva_backbone' not in name:
                    new_module_params.append(param)
        
        param_groups = [
            {
                'params': backbone_params,
                'lr': backbone_lr,
                'weight_decay': weight_decay,
                'name': 'eva_backbone'
            },
            {
                'params': new_module_params,
                'lr': new_modules_lr,
                'weight_decay': weight_decay,
                'name': 'comer_modules'
            }
        ]
        
        print(f"\n=== 差分学习率配置 ===")
        print(f"EVA02主干网络参数数: {len(backbone_params):,}")
        print(f"EVA02主干网络学习率: {backbone_lr}")
        print(f"新增模块参数数: {len(new_module_params):,}")
        print(f"新增模块学习率: {new_modules_lr}")
        print(f"权重衰减: {weight_decay}")
        
        return param_groups
    
    # 优化器配置
    if args.use_differential_lr:
        # 差分学习率策略
        param_groups = get_parameter_groups(
            model, 
            backbone_lr=args.lr, 
            new_modules_lr=args.comer_lr, 
            weight_decay=args.weight_decay
        )
        optimizer = optim.AdamW(
            param_groups,
            betas=(0.9, 0.999)
        )
    else:
        # 统一学习率策略
        print(f"\n=== 统一学习率配置 ===")
        print(f"所有参数学习率: {args.lr}")
        print(f"权重衰减: {args.weight_decay}")
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    
    # 学习率调度器
    num_training_steps = len(train_loader) // args.accumulation_steps * args.epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_mae': [],
        'val_mae': [],
        'train_class_mae': [],
        'val_class_mae': [],
        'lr': []
    }
    
    # 最佳模型记录
    best_val_acc = 0.0
    best_epoch = 0
    start_epoch = 0
    
    # 恢复训练
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"恢复训练: {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            best_val_acc = checkpoint['best_val_acc']
            history = checkpoint['history']
            print(f"从第 {start_epoch} 轮次开始恢复训练")
        else:
            print(f"未找到检查点文件: {args.resume}")
    
    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        print(f"\n开始第 {epoch + 1} 轮次训练...")
        
        # 训练
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, args, class_names,
            use_amp=args.use_amp, accumulation_steps=args.accumulation_steps, max_grad_norm=args.max_grad_norm
        )
        
        # 验证
        val_stats = validate(model, val_loader, criterion, device, class_names)
        
        # 记录历史
        history['train_loss'].append(train_stats['loss'])
        history['val_loss'].append(val_stats['loss'])
        history['train_acc'].append(train_stats['acc'])
        history['val_acc'].append(val_stats['acc'])
        history['train_mae'].append(train_stats['mae'])
        history['val_mae'].append(val_stats['mae'])
        history['train_class_mae'].append(train_stats['class_mae'])
        history['val_class_mae'].append(val_stats['class_mae'])
        # 记录学习率
        if args.use_differential_lr:
            # 差分学习率模式
            backbone_lr = optimizer.param_groups[0]['lr'] if len(optimizer.param_groups) > 0 else 0
            comer_lr = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else 0
            history['lr'].append({'backbone_lr': backbone_lr, 'comer_lr': comer_lr})
        else:
            # 统一学习率模式
            current_lr = optimizer.param_groups[0]['lr']
            history['lr'].append({'backbone_lr': current_lr, 'comer_lr': current_lr})
        
        # 打印统计信息
        current_lr = history['lr'][-1]
        print(f"\n训练 - 损失: {train_stats['loss']:.4f}, 准确率: {train_stats['acc']:.2f}%, MAE: {train_stats['mae']:.4f}")
        print(f"验证 - 损失: {val_stats['loss']:.4f}, 准确率: {val_stats['acc']:.2f}%, MAE: {val_stats['mae']:.4f}")
        print(f"学习率 - EVA02主干: {current_lr['backbone_lr']:.2e}, \新增模块: {current_lr['comer_lr']:.2e}")
        
        # 打印每类MAE
        print(f"\n训练集各类MAE:")
        for class_name, mae_val in train_stats['class_mae'].items():
            print(f"  {class_name}: {mae_val:.4f}")
        
        print(f"\n验证集各类MAE:")
        for class_name, mae_val in val_stats['class_mae'].items():
            print(f"  {class_name}: {mae_val:.4f}")
        
        # 保存最佳模型
        is_best = val_stats['acc'] > best_val_acc
        if is_best:
            best_val_acc = val_stats['acc']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history,
                'num_classes': args.num_classes,
                'class_names': class_names
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"保存最佳模型 (验证准确率: {best_val_acc:.2f}%)")
        
        # 保存最新模型
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'history': history,
            'num_classes': args.num_classes,
            'class_names': class_names
        }, os.path.join(args.output_dir, 'latest_model.pth'))
        
        # 保存每个轮次的模型（可选）
        if args.save_all_epochs:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history,
                'num_classes': args.num_classes,
                'class_names': class_names
            }, os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth'))
        
        # 绘制训练曲线
        if len(history['train_loss']) > 0:
            plot_training_curves(
                history['train_loss'], history['val_loss'],
                history['train_acc'], history['val_acc'],
                history['train_mae'], history['val_mae'],
                args.output_dir
            )
        
        # 保存训练历史
        with open(os.path.join(args.output_dir, 'history.json'), 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    # 训练完成
    total_time = time.time() - total_start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f"\n{'='*50}")
    print(f"训练完成！")
    print(f"总用时: {hours}小时{minutes}分{seconds}秒")
    print(f"最佳验证准确率: {best_val_acc:.2f}% (第 {best_epoch + 1} 轮次)")
    print(f"模型保存在: {args.output_dir}")
    print(f"{'='*50}")
    
    # 保存最终模型
    torch.save({
        'epoch': args.epochs - 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_val_acc': best_val_acc,
        'history': history,
        'num_classes': args.num_classes,
        'class_names': class_names
    }, os.path.join(args.output_dir, 'final_model.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('训练脚本', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)