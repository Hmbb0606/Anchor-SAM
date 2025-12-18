# ./inference.py (通用推理脚本 - 完整版)
import os
import sys
import yaml
import logging
import argparse
import importlib
import numpy as np
from PIL import Image
from tqdm import tqdm
from types import SimpleNamespace

import torch
import torchvision
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score

# 导入您自己的依赖
from utils.data_loading import BasicDataset
from utils.evaluation import CustomIoU

Image.MAX_IMAGE_PIXELS = None


def load_config(config_path):
    """加载并解析YAML配置文件为命名空间"""
    try:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        def dict_to_namespace(d):
            if not isinstance(d, dict):
                return d
            for k, v in d.items():
                d[k] = dict_to_namespace(v)
            return SimpleNamespace(**d)

        return dict_to_namespace(yaml_config)
    except FileNotFoundError:
        logging.error(f"错误: 配置文件 '{config_path}' 未找到。")
        sys.exit(1)


def create_output_dirs(config):
    """根据需要创建所有输出文件夹"""
    logging.info("正在创建输出目录...")
    base_dir = "results"
    # 使用 log_project_name 来创建文件夹前缀，更具描述性
    model_name_prefix = config.log_project_name.split('_')[0].lower()

    if config.inference.save_binary_mask:
        os.makedirs(os.path.join(base_dir, f"{model_name_prefix}_seg"), exist_ok=True)
    if config.inference.save_prob_map:
        os.makedirs(os.path.join(base_dir, f"{model_name_prefix}_tensor"), exist_ok=True)
    if config.inference.save_error_map:
        os.makedirs(os.path.join(base_dir, f"{model_name_prefix}_error_map"), exist_ok=True)
    logging.info("目录创建完成。")


def generate_error_map(pred_binary, label_binary):
    """生成并返回错误分析图 (TP, FP, TN, FN)"""
    TP_COLOR, TN_COLOR, FP_COLOR, FN_COLOR = [255, 255, 255], [0, 0, 0], [255, 0, 0], [0, 0, 255]
    h, w = pred_binary.shape
    error_map_img = np.zeros((h, w, 3), dtype=np.uint8)
    TP = (pred_binary == 1) & (label_binary == 1)
    TN = (pred_binary == 0) & (label_binary == 0)
    FP = (pred_binary == 1) & (label_binary == 0)
    FN = (pred_binary == 0) & (label_binary == 1)
    error_map_img[TP], error_map_img[TN], error_map_img[FP], error_map_img[FN] = TP_COLOR, TN_COLOR, FP_COLOR, FN_COLOR
    return Image.fromarray(error_map_img)


def run_inference(config):
    """主推理函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'使用设备: {device}')

    # 1. 创建数据集和加载器
    test_dataset = BasicDataset(
        images_dir=f'{config.root_dir}/{config.dataset_name}/test/image/',
        labels_dir=f'{config.root_dir}/{config.dataset_name}/test/label/',
        train=False
    )
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=4)

    # 2. 【核心】动态加载并设置模型
    logging.info(f"正在从 ./model/{config.model_filename}.py 加载模型...")
    try:
        model_module = importlib.import_module(f'model.{config.model_filename}')
        ModelClass = getattr(model_module, config.model_classname)
    except (ImportError, AttributeError) as e:
        logging.error(f"无法加载模型。请检查配置文件中的 'model_filename' 和 'model_classname' 是否正确。")
        logging.error(f"原始错误: {e}")
        sys.exit(1)

    net = ModelClass(**vars(config.model_args))
    net.to(device=device)

    model_path = config.inference.model_path
    if not os.path.exists(model_path):
        logging.error(f"模型文件未找到: {model_path}")
        sys.exit(1)

    logging.info(f"正在从 {model_path} 加载模型...")
    state_dict = torch.load(model_path, map_location=device)
    # 兼容完整的checkpoint或仅state_dict
    net.load_state_dict(state_dict['net'] if 'net' in state_dict else state_dict)
    logging.info("模型加载成功。")

    # 3. 初始化评估指标
    metric_collection = MetricCollection({
        'accuracy': Accuracy(task="binary", threshold=config.inference.threshold).to(device),
        'precision': Precision(task="binary", threshold=config.inference.threshold).to(device),
        'recall': Recall(task="binary", threshold=config.inference.threshold).to(device),
        'f1score': F1Score(task="binary", threshold=config.inference.threshold).to(device),
        'iou': CustomIoU(threshold=config.inference.threshold).to(device)
    }).to(device)

    if config.inference.save_outputs:
        create_output_dirs(config)

    # 4. 开始推理循环
    net.eval()
    with torch.no_grad():
        for image, label, name in tqdm(test_loader, desc="正在推理"):
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.int32)

            # 5. 【核心】动态构建 forward 函数的参数
            forward_kwargs = {}
            if hasattr(config.inference, 'forward_args'):
                forward_kwargs = vars(config.inference.forward_args)

            pred_logits = net(image, **forward_kwargs)
            pred_probs = torch.sigmoid(pred_logits)
            pred_binary = (pred_probs > config.inference.threshold).int()

            metric_collection.update(pred_probs.squeeze(1), label)

            # 6. 保存输出
            if config.inference.save_outputs:
                base_name = name[0]
                model_name_prefix = config.log_project_name.split('_')[0].lower()

                if config.inference.save_binary_mask:
                    save_path = os.path.join("results", f"{model_name_prefix}_seg", f"{base_name}.png")
                    torchvision.utils.save_image(pred_binary.float().cpu(), save_path)
                if config.inference.save_prob_map:
                    save_path = os.path.join("results", f"{model_name_prefix}_tensor", f"{base_name}.pt")
                    torch.save(pred_probs.cpu(), save_path)
                if config.inference.save_error_map:
                    pred_np = pred_binary.squeeze().cpu().numpy()
                    label_np = label.squeeze().cpu().numpy()
                    error_map = generate_error_map(pred_np, label_np)
                    save_path = os.path.join("results", f"{model_name_prefix}_error_map", f"{base_name}.png")
                    error_map.save(save_path)

    # 7. 计算并打印最终指标
    final_metrics = metric_collection.compute()
    print("\n" + "=" * 30 + f"\n      {config.log_project_name} 测试集性能\n" + "=" * 30)
    for name, value in final_metrics.items():
        print(f"{name.capitalize():<12}: {value.item():.4f}")
    print("=" * 30 + "\n推理完成。\n")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description='通用模型推理脚本')
    parser.add_argument('-c', '--config', default='config/Anchor-SAM.yaml',
                        help='配置文件的路径', dest='config')
    args = parser.parse_args()

    config = load_config(args.config)

    try:
        run_inference(config)
    except Exception as e:
        logging.error(f"推理过程中发生错误: {e}", exc_info=True)
        sys.exit(1)
