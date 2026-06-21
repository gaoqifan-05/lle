"""
Unified Benchmark Runner — 统一推理 + 指标计算

Usage:
    python run_benchmark.py --model enlightengan --dataset fivek
    python run_benchmark.py --model colie --dataset all
    python run_benchmark.py --model all --dataset fivek
"""

import os
import sys
import shutil
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm

# ──────────────────── 项目根目录 ────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ──────────────────── Python 解释器 ────────────────────
PYTHON = '/home/intern2/anaconda3/envs/lle/bin/python'  # conda lle 环境
BENCHMARK_DIR = PROJECT_ROOT / 'benchmark'
DATASETS_DIR = PROJECT_ROOT / 'datasets'
RESULTS_DIR = PROJECT_ROOT / 'results'

# ──────────────────── 数据集配置 ────────────────────
DATASET_CONFIG = {
    'fivek': {
        'input_dir': DATASETS_DIR / 'FiveK' / 'test' / 'input',
        'gt_dir':    DATASETS_DIR / 'FiveK' / 'test' / 'gt',
        'name': 'MIT-FiveK',
    },
    'huawei': {
        'input_dir': DATASETS_DIR / 'Huawei' / 'test' / 'input',
        'gt_dir':    DATASETS_DIR / 'Huawei' / 'test' / 'gt',
        'name': 'Huawei',
    },
    'nikon': {
        'input_dir': DATASETS_DIR / 'Nikon' / 'test' / 'input',
        'gt_dir':    DATASETS_DIR / 'Nikon' / 'test' / 'gt',
        'name': 'Nikon',
    },
}

# ──────────────────── 模型配置 ────────────────────
MODEL_CONFIG = {
    'enlightengan': {
        'name': 'EnlightenGAN',
        'dir': BENCHMARK_DIR / 'EnlightenGAN',
        'prepare': 'prepare_enlightengan',
        'run': 'run_enlightengan',
        'collect': 'collect_enlightengan',
    },
    'colie': {
        'name': 'CoLIE',
        'dir': BENCHMARK_DIR / 'colie',
        'prepare': 'prepare_colie',
        'run': 'run_colie',
        'collect': 'collect_colie',
    },
}


# ══════════════════════════════════════════════════════════
#  指标计算
# ══════════════════════════════════════════════════════════

def compute_psnr(img1, img2):
    """PSNR (Peak Signal-to-Noise Ratio)"""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim(img1, img2):
    """SSIM (Structural Similarity), simplified implementation"""
    from scipy.ndimage import uniform_filter

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    K1, K2 = 0.01, 0.03
    L = 255.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # Use uniform filter for mean/std
    mu1 = uniform_filter(img1, size=11)
    mu2 = uniform_filter(img2, size=11)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = uniform_filter(img1 ** 2, size=11) - mu1_sq
    sigma2_sq = uniform_filter(img2 ** 2, size=11) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, size=11) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def compute_lpips_batch(img_pairs, device='cuda'):
    """
    LPIPS 批处理：一次计算多对图像
    img_pairs: list of (out_img, gt_img) as numpy arrays
    """
    try:
        import lpips
        import torch

        if not hasattr(compute_lpips_batch, '_model'):
            model = lpips.LPIPS(net='alex', verbose=False)
            if device == 'cuda' and torch.cuda.is_available():
                model = model.cuda()
            else:
                device = 'cpu'
            compute_lpips_batch._model = model
            compute_lpips_batch._device = device

        model = compute_lpips_batch._model
        device = compute_lpips_batch._device

        results = []
        for out_img, gt_img in img_pairs:
            out_t = torch.from_numpy(out_img).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
            gt_t = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
            if device == 'cuda':
                out_t, gt_t = out_t.cuda(), gt_t.cuda()
            results.append(float(model(out_t, gt_t).item()))

        return results
    except ImportError:
        return [None] * len(img_pairs)


def evaluate_pair(output_path, gt_path):
    """计算单对图像的 PSNR 和 SSIM（快速）"""
    result = {}
    try:
        out_img = np.array(Image.open(output_path).convert('RGB'))
        gt_img = np.array(Image.open(gt_path).convert('RGB'))

        # 确保尺寸一致
        if out_img.shape != gt_img.shape:
            gt_img = np.array(Image.fromarray(gt_img).resize(
                (out_img.shape[1], out_img.shape[0]), Image.LANCZOS))

        # 缓存图像给 LPIPS 批处理
        result['_out'] = out_img
        result['_gt'] = gt_img

        # PSNR — 独立计算
        try:
            result['psnr'] = compute_psnr(out_img, gt_img)
        except Exception as e:
            result['psnr'] = None

        # SSIM — 独立计算
        try:
            result['ssim'] = compute_ssim(out_img, gt_img)
        except Exception as e:
            result['ssim'] = None

    except Exception as e:
        result['psnr'] = None
        result['ssim'] = None
        result['error'] = str(e)

    return result


def evaluate_all(output_dir, gt_dir):
    """评估整个输出目录 vs GT 目录"""
    if not output_dir.exists():
        print(f"  ⚠ 输出目录不存在: {output_dir}")
        return None

    # 建立 GT 索引：{stem: path}
    gt_map = {}
    for p in gt_dir.iterdir():
        if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
            gt_map[p.stem] = p
    gt_name_map = {p.name: p for p in gt_dir.iterdir()
                   if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}}

    # 收集 (输出路径, GT路径) 对
    pairs = []
    for out_file in sorted(output_dir.iterdir()):
        if out_file.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
            continue

        stem = out_file.stem
        # 跳过输入图 (_real_A)
        if stem.endswith('_real_A'):
            continue

        # 去掉模型后缀: xxx_fake_B → xxx
        for suffix in ['_fake_B', '_fake_A', '_real_B']:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break

        if stem in gt_map:
            pairs.append((out_file, gt_map[stem]))
        elif out_file.name in gt_name_map:
            pairs.append((out_file, gt_name_map[out_file.name]))

    if not pairs:
        print(f"  ⚠ 没有匹配到任何 GT 文件")
        return None

    # Step 1: PSNR + SSIM（快速，保留图像缓存给 LPIPS）
    print(f"  计算 PSNR/SSIM ({len(pairs)} 对)...")
    results = []
    for out_path, gt_path in tqdm(pairs, desc="  PSNR/SSIM"):
        r = evaluate_pair(out_path, gt_path)
        results.append(r)

    # Step 2: 批量 LPIPS（GPU 加速）
    print(f"  计算 LPIPS ({len(pairs)} 对)...")
    lpips_pairs = [(r.pop('_out'), r.pop('_gt')) for r in results
                   if '_out' in r and '_gt' in r]
    if lpips_pairs:
        lpips_vals = compute_lpips_batch(lpips_pairs)
        for i, r in enumerate(results):
            r['lpips'] = lpips_vals[i] if i < len(lpips_vals) else None
    else:
        for r in results:
            r['lpips'] = None

    psnrs = [r['psnr'] for r in results if r['psnr'] is not None and r['psnr'] != float('inf')]
    ssims = [r['ssim'] for r in results if r['ssim'] is not None]
    lpips_vals = [r['lpips'] for r in results if r.get('lpips') is not None]

    return {
        'count': len(results),
        'psnr_mean': np.mean(psnrs) if psnrs else None,
        'psnr_std': np.std(psnrs) if psnrs else None,
        'ssim_mean': np.mean(ssims) if ssims else None,
        'ssim_std': np.std(ssims) if ssims else None,
        'lpips_mean': np.mean(lpips_vals) if lpips_vals else None,
        'lpips_std': np.std(lpips_vals) if lpips_vals else None,
    }


# ══════════════════════════════════════════════════════════
#  通用预处理：降采样到固定尺寸
# ══════════════════════════════════════════════════════════

def prepare_input_resized(src_dir, dst_dir, size, clear=True):
    """将 src_dir 中的图片降采样到 size×size，存入 dst_dir"""
    if clear and dst_dir.exists():
        shutil.rmtree(str(dst_dir))
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in sorted(src_dir.iterdir())
             if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]

    for f in tqdm(files, desc="  降采样"):
        img = Image.open(f).convert('RGB')
        img = img.resize((size, size), Image.LANCZOS)
        img.save(str(dst_dir / f.name))

    return len(files)


# ══════════════════════════════════════════════════════════
#  EnlightenGAN
# ══════════════════════════════════════════════════════════

def prepare_enlightengan(model_cfg, dataset_cfg, cfg):
    """为 EnlightenGAN 准备 testA/testB 目录（降采样到 resize_size）"""
    resize_size = cfg.get('resize_size', 512)
    eg_dir = model_cfg['dir']
    testA = eg_dir.parent / 'test_dataset' / 'testA'
    testB = eg_dir.parent / 'test_dataset' / 'testB'

    # 降采样输入图像到 testA
    n = prepare_input_resized(dataset_cfg['input_dir'], testA, resize_size)
    print(f"  testA: {n} 文件 (降采样到 {resize_size}×{resize_size})")

    # testB 放一张 GT 占位（EnlightenGAN 要求 testB 非空）
    if testB.exists():
        shutil.rmtree(str(testB))
    testB.mkdir(parents=True, exist_ok=True)
    gt_files = sorted(dataset_cfg['gt_dir'].iterdir())
    if gt_files:
        img = Image.open(gt_files[0]).convert('RGB')
        img = img.resize((resize_size, resize_size), Image.LANCZOS)
        img.save(str(testB / gt_files[0].name))
    print(f"  testB: 1 文件 (占位)")


def run_enlightengan(model_cfg, dataset_cfg, cfg):
    """运行 EnlightenGAN 推理（直接调 predict.py，绕过 script.py）"""
    eg_dir = model_cfg['dir']

    cmd = (
        f"cd {eg_dir} && {PYTHON} predict.py "
        f"--dataroot ../test_dataset "
        f"--name enlightening "
        f"--model single "
        f"--which_direction AtoB "
        f"--no_dropout "
        f"--dataset_mode unaligned "
        f"--which_model_netG sid_unet_resize "
        f"--skip 1 "
        f"--use_norm 1 "
        f"--use_wgan 0 "
        f"--self_attention "
        f"--times_residual "
        f"--instance_norm 0 "
        f"--vgg 0 "
        f"--resize_or_crop resize "
        f"--loadSize 512 "
        f"--fineSize 512 "
        f"--display_id -1 "
        f"--which_epoch 200 "
        f"--gpu_ids 0 "
        f"--nThreads 1 "
        f"--batchSize 1 "
        f"--serial_batches "
        f"--no_flip"
    )

    print("  运行 EnlightenGAN 推理...")
    result = subprocess.run(
        cmd, shell=True, cwd=str(PROJECT_ROOT),
        timeout=cfg.get('timeout', 3600),
    )
    if result.returncode != 0:
        print(f"  ⚠ 推理出错 (exit code: {result.returncode})")
        return False
    print("  推理完成")
    return True


def collect_enlightengan(model_cfg, dataset_cfg, cfg):
    """收集 EnlightenGAN 输出到 results 目录"""
    eg_dir = model_cfg['dir']
    src = eg_dir / 'ablation' / 'enlightening' / 'test_200' / 'images'
    return src


# ══════════════════════════════════════════════════════════
#  CoLIE
# ══════════════════════════════════════════════════════════

def prepare_colie(model_cfg, dataset_cfg, cfg):
    """为 CoLIE 准备 input/ 目录（降采样到 resize_size）"""
    resize_size = cfg.get('resize_size', 512)
    colie_dir = model_cfg['dir']
    input_dir = colie_dir / 'input'

    n = prepare_input_resized(dataset_cfg['input_dir'], input_dir, resize_size)
    print(f"  input: {n} 文件 (降采样到 {resize_size}×{resize_size})")

    # 清理旧输出
    output_dir = colie_dir / 'output'
    if output_dir.exists():
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)


def run_colie(model_cfg, dataset_cfg, cfg):
    """运行 CoLIE 推理（注入 CUDA 优化，不修改 colie.py）"""
    colie_dir = model_cfg['dir']

    # CoLIE 默认参数 (可根据论文调)
    alpha = cfg.get('colie_alpha', 1.0)
    beta = cfg.get('colie_beta', 0.1)
    gamma = cfg.get('colie_gamma', 0.01)
    delta = cfg.get('colie_delta', 1.0)

    # 输入输出目录用绝对路径（因为 cwd 切到了 colie 目录）
    input_dir = (colie_dir / 'input').resolve()
    output_dir = (colie_dir / 'output').resolve()

    # 包装脚本：注入 CUDA 优化
    wrapper = (
        "import torch; "
        "torch.backends.cudnn.benchmark = True; "
        "torch.backends.cudnn.deterministic = False; "
        "torch.backends.cuda.matmul.allow_tf32 = True; "
        "import sys; "
        f"sys.argv = ['colie.py', "
        f"'--input_folder', '{input_dir}/', "
        f"'--output_folder', '{output_dir}/', "
        f"'--alpha', '{alpha}', '--beta', '{beta}', "
        f"'--gamma', '{gamma}', '--delta', '{delta}']; "
        f"exec(open('{colie_dir / 'colie.py'}').read())"
    )

    print("  运行 CoLIE 推理...")
    result = subprocess.run(
        [PYTHON, '-c', wrapper],
        cwd=str(colie_dir),
        timeout=cfg.get('timeout', 7200),
    )
    if result.returncode != 0:
        print(f"  ⚠ 推理出错 (exit code: {result.returncode})")
        return False
    print("  推理完成")
    return True


def collect_colie(model_cfg, dataset_cfg, cfg):
    """收集 CoLIE 输出"""
    return model_cfg['dir'] / 'output'


# ══════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════

def run_single(model_key, dataset_key, cfg):
    """对单个模型+数据集组合执行完整流程"""
    import time
    t_start = time.time()

    model_cfg = MODEL_CONFIG[model_key]
    dataset_cfg = DATASET_CONFIG[dataset_key]

    print(f"\n{'='*60}")
    print(f"  {model_cfg['name']} × {dataset_cfg['name']}")
    print(f"{'='*60}")

    result_dir = RESULTS_DIR / model_key / dataset_key / 'images'

    if cfg.get('skip_inference'):
        print("[跳过] 推理已完成，直接计算指标...")
    else:
        # 1. 准备
        print("[1/4] 准备输入...")
        prepare_fn = globals()[model_cfg['prepare']]
        prepare_fn(model_cfg, dataset_cfg, cfg)

        # 2. 推理
        print("[2/4] 运行推理...")
        run_fn = globals()[model_cfg['run']]
        success = run_fn(model_cfg, dataset_cfg, cfg)
        if not success:
            return None

        # 3. 收集输出
        print("[3/4] 收集输出...")
        collect_fn = globals()[model_cfg['collect']]
        output_dir = collect_fn(model_cfg, dataset_cfg, cfg)

        # 复制到 results
        result_dir.mkdir(parents=True, exist_ok=True)
        if output_dir and output_dir.exists():
            for f in output_dir.iterdir():
                if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                    shutil.copy2(str(f), str(result_dir / f.name))
            print(f"  输出已复制到: {result_dir}")

    # 4. 计算指标
    print("[4/4] 计算指标...")
    metrics = evaluate_all(result_dir, dataset_cfg['gt_dir'])

    if metrics:
        # 保存指标
        metrics_path = RESULTS_DIR / model_key / dataset_key / 'metrics.json'
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elapsed = time.time() - t_start
        metrics['model'] = model_key
        metrics['dataset'] = dataset_key
        metrics['timestamp'] = timestamp
        metrics['time_seconds'] = round(elapsed, 1)
        metrics['time_str'] = f"{int(elapsed//60)}m{elapsed%60:.0f}s"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  指标已保存到: {metrics_path}")

        # 打印摘要
        print(f"\n  ┌{'─'*40}┐")
        print(f"  │  {model_cfg['name']} × {dataset_cfg['name']}  │")
        print(f"  ├{'─'*40}┤")
        psnr_str = f"{metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}" if metrics['psnr_mean'] is not None else "N/A"
        ssim_str = f"{metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}" if metrics['ssim_mean'] is not None else "N/A"
        lpips_str = f"{metrics['lpips_mean']:.4f} ± {metrics['lpips_std']:.4f}" if metrics['lpips_mean'] is not None else "N/A"
        print(f"  │  PSNR:   {psnr_str}     │")
        print(f"  │  SSIM:   {ssim_str}  │")
        print(f"  │  LPIPS:  {lpips_str}  │")
        time_str = metrics.get('time_str', 'N/A')
        print(f"  │  耗时:   {time_str}  │")
        print(f"  └{'─'*40}┘")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Unified Low-Light Enhancement Benchmark')
    parser.add_argument('--model', type=str, default='enlightengan',
                        choices=['enlightengan', 'colie', 'all'],
                        help='选择模型 (default: enlightengan)')
    parser.add_argument('--dataset', type=str, default='fivek',
                        choices=['fivek', 'huawei', 'nikon', 'all'],
                        help='选择数据集 (default: fivek)')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='推理超时秒数 (default: 3600)')
    parser.add_argument('--skip_inference', action='store_true',
                        help='跳过推理，仅计算指标 (需已有结果)')
    parser.add_argument('--resize_size', type=int, default=512,
                        help='推理时统一降采样尺寸 (default: 512)')
    # CoLIE 参数
    parser.add_argument('--colie_alpha', type=float, default=1.0)
    parser.add_argument('--colie_beta', type=float, default=0.1)
    parser.add_argument('--colie_gamma', type=float, default=0.01)
    parser.add_argument('--colie_delta', type=float, default=1.0)

    args = parser.parse_args()

    models = list(MODEL_CONFIG.keys()) if args.model == 'all' else [args.model]
    datasets = list(DATASET_CONFIG.keys()) if args.dataset == 'all' else [args.dataset]

    all_metrics = {}
    for model_key in models:
        for dataset_key in datasets:
            metrics = run_single(model_key, dataset_key, vars(args))
            if metrics:
                all_metrics[f"{model_key}/{dataset_key}"] = metrics

    # 汇总表
    if all_metrics:
        print(f"\n{'='*70}")
        print(f"  汇总结果")
        print(f"{'='*70}")
        print(f"  {'模型×数据集':<30} {'PSNR↑':>8} {'SSIM↑':>8} {'LPIPS↓':>8} {'耗时':>10}")
        print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")
        for key, m in all_metrics.items():
            psnr = f"{m['psnr_mean']:.2f}" if m['psnr_mean'] else 'N/A'
            ssim = f"{m['ssim_mean']:.4f}" if m['ssim_mean'] else 'N/A'
            lpips = f"{m['lpips_mean']:.4f}" if m['lpips_mean'] else 'N/A'
            t = m.get('time_str', 'N/A')
            print(f"  {key:<30} {psnr:>8} {ssim:>8} {lpips:>8} {t:>10}")

        # 保存汇总
        summary_path = RESULTS_DIR / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\n  汇总已保存: {summary_path}")


if __name__ == '__main__':
    main()
