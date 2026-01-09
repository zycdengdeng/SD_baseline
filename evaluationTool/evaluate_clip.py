"""
评测脚本 - 对单个 clip 进行图像质量和目标检测评测

指标：
- PSNR: 峰值信噪比（越高越好）
- SSIM: 结构相似度（越高越好，0-1）
- LPIPS: 感知相似度（越低越好）
- MSE: 均方误差（越低越好）
- YOLO: 目标检测对比
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from datetime import datetime

import torchvision.transforms as T
from torchvision.utils import save_image

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.lightning_depth import LidarDiffusionModule


def prompt_input(prompt, default):
    """交互式输入"""
    user_input = input(f"{prompt} [{default}]: ").strip()
    return user_input if user_input else default


class ImageMetrics:
    """图像质量指标计算"""

    def __init__(self, device='cuda'):
        self.device = device
        self.lpips_model = None

    def _load_lpips(self):
        """延迟加载 LPIPS"""
        if self.lpips_model is None:
            try:
                import lpips
                self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
                self.lpips_model.eval()
            except ImportError:
                print("警告: lpips 未安装，跳过 LPIPS 指标")
                print("安装命令: pip install lpips")
                return False
        return True

    def psnr(self, img1, img2):
        """计算 PSNR"""
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

    def ssim(self, img1, img2):
        """计算 SSIM"""
        try:
            from torchmetrics.image import StructuralSimilarityIndexMeasure
            ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
            return ssim_metric(img1, img2).item()
        except ImportError:
            print("警告: torchmetrics 未安装，使用简化 SSIM")
            # 简化版 SSIM
            c1, c2 = 0.01 ** 2, 0.03 ** 2
            mu1, mu2 = img1.mean(), img2.mean()
            sigma1, sigma2 = img1.var(), img2.var()
            sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
            return ssim.item()

    def lpips(self, img1, img2):
        """计算 LPIPS"""
        if not self._load_lpips():
            return None
        # LPIPS 需要 [-1, 1] 范围
        img1_scaled = img1 * 2 - 1
        img2_scaled = img2 * 2 - 1
        with torch.no_grad():
            return self.lpips_model(img1_scaled, img2_scaled).item()

    def mse(self, img1, img2):
        """计算 MSE"""
        return torch.mean((img1 - img2) ** 2).item()

    def compute_all(self, generated, gt):
        """计算所有指标"""
        results = {
            'PSNR': self.psnr(generated, gt),
            'SSIM': self.ssim(generated, gt),
            'MSE': self.mse(generated, gt),
        }

        lpips_val = self.lpips(generated, gt)
        if lpips_val is not None:
            results['LPIPS'] = lpips_val

        return results


class YOLOEvaluator:
    """YOLO 目标检测评测"""

    def __init__(self, model_name='yolov8n.pt'):
        self.model = None
        self.model_name = model_name

    def _load_model(self):
        """延迟加载 YOLO"""
        if self.model is None:
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_name)
                print(f"YOLO 模型加载: {self.model_name}")
            except ImportError:
                print("警告: ultralytics 未安装，跳过目标检测")
                print("安装命令: pip install ultralytics")
                return False
        return True

    def detect(self, image_tensor):
        """对图像进行目标检测"""
        if not self._load_model():
            return None

        # tensor to numpy
        img = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)

        results = self.model(img, verbose=False)[0]

        detections = []
        for box in results.boxes:
            detections.append({
                'class': int(box.cls.item()),
                'class_name': results.names[int(box.cls.item())],
                'confidence': float(box.conf.item()),
                'bbox': box.xyxy[0].tolist()
            })

        return detections

    def compare(self, gt_detections, gen_detections):
        """比较两组检测结果"""
        if gt_detections is None or gen_detections is None:
            return None

        gt_count = len(gt_detections)
        gen_count = len(gen_detections)

        # 按类别统计
        gt_classes = defaultdict(int)
        gen_classes = defaultdict(int)

        for det in gt_detections:
            gt_classes[det['class_name']] += 1
        for det in gen_detections:
            gen_classes[det['class_name']] += 1

        # 计算置信度
        gt_conf = np.mean([d['confidence'] for d in gt_detections]) if gt_detections else 0
        gen_conf = np.mean([d['confidence'] for d in gen_detections]) if gen_detections else 0

        return {
            'gt_total': gt_count,
            'gen_total': gen_count,
            'count_diff': gen_count - gt_count,
            'gt_classes': dict(gt_classes),
            'gen_classes': dict(gen_classes),
            'gt_avg_conf': gt_conf,
            'gen_avg_conf': gen_conf,
        }


def load_image(path, image_size=(512, 512)):
    """加载图像"""
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img)


def scan_clip(clip_path):
    """扫描 clip 中的所有时间戳"""
    timestamps = {}
    image_names = ['FL.jpg', 'FN.jpg', 'FR.jpg', 'FW.jpg', 'RL.jpg', 'RN.jpg', 'RR.jpg']

    for timestamp_dir in sorted(os.listdir(clip_path)):
        timestamp_path = os.path.join(clip_path, timestamp_dir)
        if not os.path.isdir(timestamp_path):
            continue

        gt_dir = os.path.join(timestamp_path, 'gt')
        if not os.path.exists(gt_dir):
            gt_dir = os.path.join(timestamp_path, 'GT')
        proj_dir = os.path.join(timestamp_path, 'proj')

        if os.path.exists(gt_dir) and os.path.exists(proj_dir):
            views = {}
            for img_name in image_names:
                gt_path = os.path.join(gt_dir, img_name)
                proj_path = os.path.join(proj_dir, img_name)
                if os.path.exists(gt_path) and os.path.exists(proj_path):
                    view = img_name.split('.')[0]
                    views[view] = {
                        'gt_path': gt_path,
                        'proj_path': proj_path
                    }
            if views:
                timestamps[timestamp_dir] = views

    return timestamps


def main():
    print("=" * 60)
    print("  图像质量评测工具")
    print("  指标: PSNR, SSIM, LPIPS, MSE + YOLO 目标检测")
    print("=" * 60)
    print()

    # 默认值
    default_ckpt = "/mnt/zihanw/SD_baseline/experiments/p2g_20251226_155015/checkpoints/best-epoch=epoch=149-val_loss=val/loss=0.314.ckpt"
    default_clip = "/mnt/zihanw/proj_utils_pro/blur投影/001/001"
    default_output = "./evaluation_results"
    default_num_timestamps = "10"
    default_num_steps = "50"
    default_yolo = "y"

    # 交互式输入
    ckpt_path = prompt_input("Checkpoint 路径", default_ckpt)
    clip_path = prompt_input("Clip 路径", default_clip)

    # 扫描数据
    print(f"\n正在扫描: {clip_path}")
    timestamps = scan_clip(clip_path)
    print(f">>> 找到 {len(timestamps)} 个有效时间戳 <<<\n")

    if len(timestamps) == 0:
        print("错误: 没有找到有效数据")
        return

    output_dir = prompt_input("输出目录", default_output)
    num_timestamps = int(prompt_input(f"评测时间戳数量 (-1=全部{len(timestamps)})", default_num_timestamps))
    num_steps = int(prompt_input("推理步数", default_num_steps))
    use_yolo = prompt_input("使用 YOLO 检测 (y/n)", default_yolo).lower() in ['y', 'yes']

    # 确认配置
    print("\n" + "=" * 60)
    print("  配置确认")
    print("=" * 60)
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Clip: {clip_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  时间戳数量: {num_timestamps if num_timestamps > 0 else '全部'}")
    print(f"  推理步数: {num_steps}")
    print(f"  YOLO 检测: {use_yolo}")
    print("=" * 60)

    confirm = input("\n确认开始? (Enter 继续 / n 取消): ").strip().lower()
    if confirm == 'n':
        print("已取消")
        return

    # 创建输出目录
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"eval_{timestamp_str}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    # 加载模型
    print(f"\n正在加载模型...")
    model = LidarDiffusionModule.load_from_checkpoint(ckpt_path)
    model.eval()
    model.cuda()
    print("模型加载完成!")

    # 初始化评测器
    metrics = ImageMetrics(device='cuda')
    yolo = YOLOEvaluator() if use_yolo else None

    # 选择时间戳
    all_ts = sorted(timestamps.keys())
    if num_timestamps > 0:
        # 从中间选择
        mid = len(all_ts) // 2
        half = num_timestamps // 2
        start = max(0, mid - half)
        end = min(len(all_ts), start + num_timestamps)
        selected_ts = all_ts[start:end]
    else:
        selected_ts = all_ts

    print(f"\n将评测 {len(selected_ts)} 个时间戳")

    # 存储所有结果
    all_results = {
        'config': {
            'ckpt': ckpt_path,
            'clip': clip_path,
            'num_steps': num_steps,
            'num_timestamps': len(selected_ts),
        },
        'per_image': [],
        'per_view': defaultdict(list),
        'per_timestamp': [],
    }

    # 开始评测
    for i, ts in enumerate(selected_ts):
        print(f"\n[{i+1}/{len(selected_ts)}] 时间戳: {ts}")

        ts_results = {'timestamp': ts, 'views': {}}
        views = timestamps[ts]

        for view, paths in views.items():
            # 加载图像
            gt_tensor = load_image(paths['gt_path']).unsqueeze(0).cuda()
            proj_tensor = load_image(paths['proj_path']).unsqueeze(0).cuda()

            # 推理
            with torch.no_grad():
                generated = model.generate_from_lidar(proj_tensor, num_inference_steps=num_steps)
                generated = torch.clamp(generated, 0, 1)

            # 计算指标
            view_metrics = metrics.compute_all(generated, gt_tensor)

            # YOLO 检测
            yolo_result = None
            if yolo is not None:
                gt_det = yolo.detect(gt_tensor)
                gen_det = yolo.detect(generated)
                yolo_result = yolo.compare(gt_det, gen_det)
                if yolo_result:
                    view_metrics['YOLO'] = yolo_result

            # 保存对比图
            comparison = torch.cat([proj_tensor.cpu(), generated.cpu(), gt_tensor.cpu()], dim=3)
            img_path = os.path.join(output_dir, "images", f"{ts}_{view}.jpg")
            save_image(comparison, img_path)

            # 记录结果
            ts_results['views'][view] = view_metrics
            all_results['per_image'].append({
                'timestamp': ts,
                'view': view,
                **view_metrics
            })
            all_results['per_view'][view].append(view_metrics)

            # 打印
            print(f"  {view}: PSNR={view_metrics['PSNR']:.2f}, SSIM={view_metrics['SSIM']:.4f}, MSE={view_metrics['MSE']:.6f}", end="")
            if 'LPIPS' in view_metrics:
                print(f", LPIPS={view_metrics['LPIPS']:.4f}", end="")
            if yolo_result:
                print(f", YOLO[GT:{yolo_result['gt_total']}/Gen:{yolo_result['gen_total']}]", end="")
            print()

        all_results['per_timestamp'].append(ts_results)

    # 计算汇总统计
    print("\n" + "=" * 60)
    print("  评测结果汇总")
    print("=" * 60)

    # 整体平均
    avg_metrics = {}
    for key in ['PSNR', 'SSIM', 'MSE', 'LPIPS']:
        values = [r[key] for r in all_results['per_image'] if key in r]
        if values:
            avg_metrics[key] = np.mean(values)

    print("\n【整体平均】")
    print(f"  PSNR:  {avg_metrics.get('PSNR', 'N/A'):.2f} dB")
    print(f"  SSIM:  {avg_metrics.get('SSIM', 'N/A'):.4f}")
    print(f"  MSE:   {avg_metrics.get('MSE', 'N/A'):.6f}")
    if 'LPIPS' in avg_metrics:
        print(f"  LPIPS: {avg_metrics['LPIPS']:.4f}")

    # 按视角统计
    print("\n【按视角统计】")
    view_avg = {}
    for view, results in all_results['per_view'].items():
        view_avg[view] = {
            'PSNR': np.mean([r['PSNR'] for r in results]),
            'SSIM': np.mean([r['SSIM'] for r in results]),
            'MSE': np.mean([r['MSE'] for r in results]),
        }
        lpips_vals = [r['LPIPS'] for r in results if 'LPIPS' in r]
        if lpips_vals:
            view_avg[view]['LPIPS'] = np.mean(lpips_vals)

        print(f"  {view}: PSNR={view_avg[view]['PSNR']:.2f}, SSIM={view_avg[view]['SSIM']:.4f}", end="")
        if 'LPIPS' in view_avg[view]:
            print(f", LPIPS={view_avg[view]['LPIPS']:.4f}", end="")
        print()

    # YOLO 统计
    if use_yolo:
        yolo_stats = [r.get('YOLO') for r in all_results['per_image'] if r.get('YOLO')]
        if yolo_stats:
            total_gt = sum(y['gt_total'] for y in yolo_stats)
            total_gen = sum(y['gen_total'] for y in yolo_stats)
            print(f"\n【YOLO 检测统计】")
            print(f"  GT 总检测数:   {total_gt}")
            print(f"  生成图检测数: {total_gen}")
            print(f"  检测率:        {total_gen/total_gt*100:.1f}%" if total_gt > 0 else "  N/A")

    # 保存结果
    all_results['summary'] = {
        'overall_avg': avg_metrics,
        'per_view_avg': view_avg,
    }

    result_path = os.path.join(output_dir, "results.json")
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n结果已保存到: {output_dir}")
    print(f"  - 对比图: {output_dir}/images/")
    print(f"  - 详细结果: {result_path}")


if __name__ == "__main__":
    main()
