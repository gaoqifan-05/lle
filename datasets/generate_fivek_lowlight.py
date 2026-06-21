"""
从 MIT-FiveK Expert C 图像生成合成低光数据。

使用方法:
    1. 将 Expert C 图像放入 datasets/FiveK/gt/ 目录
    2. 运行: python datasets/generate_fivek_lowlight.py
    3. 生成的结构:
       datasets/FiveK/train/input/   (4500张低光)
       datasets/FiveK/train/gt/      (4500张Expert C)
       datasets/FiveK/test/input/    (500张低光)
       datasets/FiveK/test/gt/       (500张Expert C)

低光生成策略（多种可选）:
    - gamma:      I_low = I_gt ^ gamma  (默认 gamma=2.2)
    - linear:     I_low = I_gt * scale + noise
    - poisson:    I_low ~ Poisson(I_gt * scale) / scale  (更真实的噪声模型)
"""

import os
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image
from tqdm import tqdm


def apply_gamma(img, gamma=2.2):
    """Gamma校正压暗 (I_out = I_in ^ gamma)"""
    img = img.astype(np.float32) / 255.0
    img = np.power(np.clip(img, 0, 1), gamma)
    return (img * 255).astype(np.uint8)


def apply_linear(img, scale=0.15, noise_std=0.02):
    """线性压暗+高斯噪声"""
    img = img.astype(np.float32) / 255.0
    img = img * scale
    noise = np.random.normal(0, noise_std, img.shape)
    img = img + noise
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def apply_poisson(img, scale=0.2):
    """泊松噪声模型（更真实模拟CMOS噪声）"""
    img = img.astype(np.float32) / 255.0
    img_scaled = img * scale * 255
    noisy = np.random.poisson(np.clip(img_scaled, 0, 255)).astype(np.float32)
    img = noisy / (scale * 255)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def generate_dataset(src_dir, dst_dir, num_train=4500, method='gamma',
                     gamma=2.2, scale=0.15, noise_std=0.02, seed=42,
                     dry_run=False):
    """
    src_dir: Expert C 图像所在目录
    dst_dir: 输出根目录 (如 datasets/FiveK)
    num_train: 训练集数量，剩余为测试集
    """
    np.random.seed(seed)

    # 收集所有图像文件
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    img_files = sorted([
        f for f in os.listdir(src_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])

    if len(img_files) == 0:
        raise ValueError(f"在 {src_dir} 中未找到图像文件")

    print(f"找到 {len(img_files)} 张 Expert C 图像")

    # 随机打乱
    indices = np.random.permutation(len(img_files))

    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    # --dry_run 预览模式：只处理1张，保存到 preview/ 目录
    if dry_run:
        preview_dir = os.path.join(dst_dir, 'preview')
        os.makedirs(preview_dir, exist_ok=True)

        idx = indices[0]
        fname = img_files[idx]
        src_path = os.path.join(src_dir, fname)
        img = np.array(Image.open(src_path).convert('RGB'))

        if method == 'gamma':
            low_img = apply_gamma(img, gamma)
            desc = f"gamma={gamma}"
        elif method == 'linear':
            low_img = apply_linear(img, scale, noise_std)
            desc = f"linear_scale={scale}_noise={noise_std}"
        elif method == 'poisson':
            low_img = apply_poisson(img, scale)
            desc = f"poisson_scale={scale}"

        Image.fromarray(low_img).save(os.path.join(preview_dir, 'low_' + fname))
        shutil.copy2(src_path, os.path.join(preview_dir, 'gt_' + fname))
        print(f"\n预览已生成: {preview_dir}/")
        print(f"  gt_{fname}  →  Expert C 原图")
        print(f"  low_{fname} →  低光合成图 ({desc})")
        return

    # 方法选择
    if method == 'gamma':
        def transform(img):
            return apply_gamma(img, gamma)
        method_desc = f"gamma={gamma}"
    elif method == 'linear':
        def transform(img):
            return apply_linear(img, scale, noise_std)
        method_desc = f"linear_scale={scale}_noise={noise_std}"
    elif method == 'poisson':
        def transform(img):
            return apply_poisson(img, scale)
        method_desc = f"poisson_scale={scale}"
    else:
        raise ValueError(f"未知方法: {method}")

    # 创建目录结构
    for split, idxs in [('train', train_indices), ('test', test_indices)]:
        os.makedirs(os.path.join(dst_dir, split, 'input'), exist_ok=True)
        os.makedirs(os.path.join(dst_dir, split, 'gt'), exist_ok=True)

    # 处理并保存（多线程并行 I/O）
    for split, idxs in [('train', train_indices), ('test', test_indices)]:
        print(f"\n生成 {split} 集 ({len(idxs)} 对)...")
        input_dir = os.path.join(dst_dir, split, 'input')
        gt_dir = os.path.join(dst_dir, split, 'gt')

        def process_one(idx):
            fname = img_files[idx]
            src_path = os.path.join(src_dir, fname)
            low_path = os.path.join(input_dir, fname)
            gt_path = os.path.join(gt_dir, fname)

            img = np.array(Image.open(src_path).convert('RGB'))
            low_img = transform(img)
            Image.fromarray(low_img).save(low_path)
            shutil.copy2(src_path, gt_path)  # GT 直接复制，不重新编码

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(process_one, idx): idx for idx in idxs}
            for _ in tqdm(as_completed(futures), total=len(idxs)):
                pass

    print(f"\n完成! 方法: {method_desc}")
    print(f"  训练集: {len(train_indices)} 对 -> {dst_dir}/train/")
    print(f"  测试集: {len(test_indices)} 对 -> {dst_dir}/test/")

    # 保存生成参数
    info_path = os.path.join(dst_dir, 'generation_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"method: {method}\n")
        f.write(f"method_desc: {method_desc}\n")
        f.write(f"total_images: {len(img_files)}\n")
        f.write(f"train_pairs: {len(train_indices)}\n")
        f.write(f"test_pairs: {len(test_indices)}\n")
        f.write(f"seed: {seed}\n")
    print(f"  生成参数已保存至: {info_path}")


def main():
    parser = argparse.ArgumentParser(
        description='从 MIT-FiveK Expert C 生成合成低光数据'
    )
    parser.add_argument('--src_dir', type=str,
                        default='datasets/FiveK/gt',
                        help='Expert C 图像目录')
    parser.add_argument('--dst_dir', type=str,
                        default='datasets/FiveK',
                        help='输出根目录')
    parser.add_argument('--num_train', type=int, default=4500,
                        help='训练集数量 (标准划分: 4500)')
    parser.add_argument('--method', type=str, default='gamma',
                        choices=['gamma', 'linear', 'poisson'],
                        help='低光生成方法')
    parser.add_argument('--gamma', type=float, default=2.2,
                        help='gamma 值 (method=gamma)')
    parser.add_argument('--scale', type=float, default=0.15,
                        help='线性缩放系数 (method=linear/poisson)')
    parser.add_argument('--noise_std', type=float, default=0.02,
                        help='噪声标准差 (method=linear)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry_run', action='store_true',
                        help='预览模式：只生成1张到 preview/ 目录，用于调参')

    args = parser.parse_args()

    generate_dataset(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        num_train=args.num_train,
        method=args.method,
        gamma=args.gamma,
        scale=args.scale,
        noise_std=args.noise_std,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()
