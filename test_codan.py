"""
CODaN 测试脚本 — 独立测试 LLE 对夜间分类的提升效果

流程：
  1. 加载 test_night 图像
  2. 用 EnlightenGAN 增强
  3. 用预训练 ResNet-18 分类，对比 day / night / enhanced 准确率
"""

import sys
import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np
import json

# ─── 项目路径 ───
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'datasets' / 'CODaN'))
from codan import CODaN

# ─── 配置 ───
CLASSES = ['Bicycle', 'Car', 'Motorbike', 'Bus', 'Boat',
           'Cat', 'Dog', 'Bottle', 'Cup', 'Chair']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ══════════════════════════════════════════════════════════
#  分类器
# ══════════════════════════════════════════════════════════

# ImageNet 标准化参数（EfficientNet 与 ResNet 相同）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASSIFIER_REGISTRY = {
    'resnet18': {
        'build': lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
        'num_features': 512,
        'set_head': lambda m, n: setattr(m, 'fc', nn.Linear(512, n)),
        'get_params': lambda m: m.fc.parameters(),
    },
    'efficientnet_b0': {
        'build': lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1),
        'num_features': 1280,
        'set_head': lambda m, n: setattr(m.classifier, '1', nn.Linear(1280, n)),
        'get_params': lambda m: m.classifier[1].parameters(),
    },
}


def build_classifier(name='resnet18'):
    """加载预训练分类器，替换最后一层为 10 分类"""
    if name not in CLASSIFIER_REGISTRY:
        raise ValueError(f"未知分类器: {name}，可选: {list(CLASSIFIER_REGISTRY.keys())}")
    cfg = CLASSIFIER_REGISTRY[name]
    model = cfg['build']()
    cfg['set_head'](model, len(CLASSES))
    model = model.to(DEVICE)
    model.eval()
    return model


def train_classifier(model, train_loader, epochs=5, lr=1e-3, classifier_name='resnet18'):
    """在 CODaN train split 上微调"""
    cfg = CLASSIFIER_REGISTRY[classifier_name]
    model.train()
    optimizer = torch.optim.Adam(cfg['get_params'](model), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total, correct = 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"  Train epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total += labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
        print(f"    acc: {correct/total*100:.1f}%")

    model.eval()
    return model


@torch.no_grad()
def evaluate(model, loader, desc="Eval"):
    """评估分类准确率"""
    total, correct = 0, 0
    for imgs, labels in tqdm(loader, desc=desc):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        total += labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    return correct / total * 100


# ══════════════════════════════════════════════════════════
#  LLE 增强
# ══════════════════════════════════════════════════════════

def load_enlightengan():
    """加载 EnlightenGAN 模型"""
    eg_dir = PROJECT_ROOT / 'benchmark' / 'EnlightenGAN'
    sys.path.insert(0, str(eg_dir))
    from models.models import create_model
    from options.test_options import TestOptions

    opt_obj = TestOptions()
    opt_obj.initialize()
    opt = opt_obj.parser.parse_args([
        '--dataroot', str(eg_dir.parent / 'test_dataset'),
        '--name', 'enlightening',
        '--model', 'single',
        '--which_direction', 'AtoB',
        '--no_dropout',
        '--dataset_mode', 'unaligned',
        '--which_model_netG', 'sid_unet_resize',
        '--skip', '1',
        '--use_norm', '1',
        '--self_attention',
        '--times_residual',
        '--instance_norm', '0',
        '--vgg', '0',
        '--resize_or_crop', 'resize',
        '--loadSize', '512',
        '--fineSize', '512',
        '--which_epoch', '200',
        '--gpu_ids', '0',
        '--nThreads', '1',
        '--batchSize', '1',
        '--serial_batches',
        '--no_flip',
    ])
    opt.isTrain = False
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = [int(s) for s in str_ids if int(s) >= 0]
    if opt.gpu_ids:
        torch.cuda.set_device(opt.gpu_ids[0])
    model = create_model(opt)
    return model


@torch.no_grad()
def enhance_enlightengan(model, img_tensor):
    """用 EnlightenGAN 增强单张图 (tensor: C,H,W, [0,1])"""
    img = img_tensor * 2 - 1
    img = img.unsqueeze(0).to(DEVICE)
    gray = img.mean(dim=1, keepdim=True)
    model.set_input({
        'A': img, 'B': img,
        'input_img': img,
        'A_gray': gray,
        'A_paths': [''],
    })
    model.test()
    fake = model.fake_B
    fake = (fake + 1) / 2
    return fake.squeeze(0).clamp(0, 1)


# LLE 模型注册表 {name: (load_fn, enhance_fn, input_size)}
LLE_REGISTRY = {
    'enlightengan': (load_enlightengan, enhance_enlightengan, 512),
}


# ══════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════

def get_classifier_ckpt_path(classifier_name):
    return PROJECT_ROOT / 'checkpoints' / f'codan_{classifier_name}.pth'


def main():
    parser = argparse.ArgumentParser(description='CODaN LLE Benchmark')
    parser.add_argument('--classifier', type=str, default='resnet18',
                        choices=list(CLASSIFIER_REGISTRY.keys()),
                        help='分类器架构')
    parser.add_argument('--lle_model', type=str, default='enlightengan',
                        choices=list(LLE_REGISTRY.keys()),
                        help='低光增强模型')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5, help='分类器微调轮数')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--skip_train', action='store_true', help='跳过训练，用零样本分类器')
    parser.add_argument('--enhance_only', action='store_true',
                        help='跳过 day/night 评估，直接从增强开始（若已有训练好的分类器则自动加载）')
    args = parser.parse_args()

    classifier_ckpt = get_classifier_ckpt_path(args.classifier)
    lle_load, lle_enhance, lle_input_size = LLE_REGISTRY[args.lle_model]

    codan_root = PROJECT_ROOT / 'datasets' / 'CODaN'

    # ─── 数据预处理 ───
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    print("=" * 60)
    print(f"  CODaN LLE Benchmark  |  classifier={args.classifier}  lle={args.lle_model}")
    print("=" * 60)

    # ─── 分类器 ───
    print(f"\n[1] 构建分类器 ({args.classifier})...")
    classifier = build_classifier(args.classifier)

    if classifier_ckpt.exists() and (args.enhance_only or args.skip_train):
        print(f"  加载已训练的分类器: {classifier_ckpt}")
        classifier.load_state_dict(torch.load(classifier_ckpt, map_location=DEVICE))
        classifier.eval()
    elif not args.skip_train:
        print("  加载训练数据...")
        train_ds = CODaN(root=str(codan_root), split='train', transform=train_transform)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)
        print(f"  train: {len(train_ds)}")

        print("  微调分类器...")
        classifier = train_classifier(classifier, train_loader, epochs=args.epochs,
                                      classifier_name=args.classifier)
        classifier_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(classifier.state_dict(), classifier_ckpt)
        print(f"  分类器已保存至: {classifier_ckpt}")

    # ─── 评估：day / night ───
    day_acc = night_acc = None
    if not args.enhance_only:
        print("\n[2] 加载评估数据...")
        test_day_ds = CODaN(root=str(codan_root), split='test_day', transform=eval_transform)
        test_night_ds = CODaN(root=str(codan_root), split='test_night', transform=eval_transform)
        day_loader = DataLoader(test_day_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
        night_loader = DataLoader(test_night_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers)
        print(f"  test_day: {len(test_day_ds)}, test_night: {len(test_night_ds)}")

        print("\n[3] 评估...")
        day_acc = evaluate(classifier, day_loader, desc="  Day")
        print(f"  Day accuracy:     {day_acc:.2f}%")
        night_acc = evaluate(classifier, night_loader, desc="  Night")
        print(f"  Night accuracy:   {night_acc:.2f}%")

    # ─── LLE 增强 + 评估 ───
    step_label = "[4]" if not args.enhance_only else "[2]"
    print(f"\n{step_label} 加载 {args.lle_model.upper()} + 增强夜间图 + 评估...")
    lle_model = lle_load()

    enhance_transform = transforms.Compose([
        transforms.Resize((lle_input_size, lle_input_size)),
        transforms.ToTensor(),
    ])
    test_night_raw = CODaN(root=str(codan_root), split='test_night',
                           transform=enhance_transform)
    test_night_eval = CODaN(root=str(codan_root), split='test_night',
                            transform=eval_transform)
    night_raw_loader = DataLoader(test_night_raw, batch_size=1,
                                  shuffle=False, num_workers=args.num_workers)
    night_eval_loader = DataLoader(test_night_eval, batch_size=1,
                                   shuffle=False, num_workers=args.num_workers)

    enhanced_correct, enhanced_total = 0, 0
    for (img, _), (_, label) in zip(
        tqdm(night_raw_loader, desc="  Enhancing+Classify", total=len(test_night_raw)),
        night_eval_loader
    ):
        label = label.to(DEVICE)
        enhanced = lle_enhance(lle_model, img.squeeze(0))
        enhanced = transforms.Resize(256)(enhanced)
        enhanced = transforms.CenterCrop(224)(enhanced)
        enhanced = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(enhanced)
        enhanced = enhanced.unsqueeze(0).to(DEVICE)
        outputs = classifier(enhanced)
        enhanced_total += label.size(0)
        enhanced_correct += (outputs.argmax(1) == label).sum().item()

    enhanced_acc = enhanced_correct / enhanced_total * 100
    print(f"  Enhanced accuracy: {enhanced_acc:.2f}%")

    # ─── 汇总 ───
    print(f"\n{'='*60}")
    print(f"  结果汇总")
    print(f"{'='*60}")
    if day_acc is not None:
        print(f"  Day:       {day_acc:6.2f}%")
        print(f"  Night:     {night_acc:6.2f}%  (drop: {day_acc-night_acc:.1f}%)")
        print(f"  Enhanced:  {enhanced_acc:6.2f}%  (recovery: {enhanced_acc-night_acc:.1f}%)")
    else:
        print(f"  Enhanced:  {enhanced_acc:6.2f}%")


if __name__ == '__main__':
    main()
