import sys
import time
import yaml
import random
import logging
import argparse
import importlib
import numpy as np
from types import SimpleNamespace

from comet_ml import Experiment, OfflineExperiment
import torch
from torch import optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score

from utils.data_loading import BasicDataset
from utils.losses import FCCDN_loss_without_seg
from utils.utils import train_val_test
from utils.evaluation import CustomIoU
import os

def random_seed(SEED):
    """设置随机种子以确保可复现性"""
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

def create_scheduler(optimizer, config):
    """根据配置文件动态创建学习率调度器"""
    if not hasattr(config, 'scheduler') or not hasattr(config.scheduler, 'active_scheduler'):
        logging.warning("配置文件中未指定 active_scheduler，将不使用学习率调度器。")
        return None

    active_scheduler = config.scheduler.active_scheduler

    if active_scheduler == 'ReduceLROnPlateau':
        params_key = 'ReduceLROnPlateau_params'
        if hasattr(config.scheduler, params_key):
            scheduler_args = vars(getattr(config.scheduler, params_key))
            logging.info(f"创建 ReduceLROnPlateau 调度器，参数: {scheduler_args}")
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_args)
        else:
            logging.error(f"激活的调度器为 {active_scheduler} 但未找到参数 '{params_key}'")
            sys.exit(1)

    elif active_scheduler == 'CosineAnnealingLR':
        params_key = 'CosineAnnealingLR_params'
        if hasattr(config.scheduler, params_key):
            scheduler_args = vars(getattr(config.scheduler, params_key))
            # 兼容性：如果T_max未在yaml中设置，则使用总epochs
            if 'T_max' not in scheduler_args:
                scheduler_args['T_max'] = config.epochs
            logging.info(f"创建 CosineAnnealingLR 调度器，参数: {scheduler_args}")
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_args)
        else:
            logging.error(f"激活的调度器为 {active_scheduler} 但未找到参数 '{params_key}'")
            sys.exit(1)

    else:
        logging.error(f"不支持的调度器类型: {active_scheduler}")
        sys.exit(1)


def train_net(config):
    """主训练函数"""
    random_seed(SEED=config.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 创建数据集
    train_dataset = BasicDataset(
        images_dir=f'{config.root_dir}/{config.dataset_name}/train/image/',
        labels_dir=f'{config.root_dir}/{config.dataset_name}/train/label/',
        train=True
    )
    val_dataset = BasicDataset(
        images_dir=f'{config.root_dir}/{config.dataset_name}/val/image/',
        labels_dir=f'{config.root_dir}/{config.dataset_name}/val/label/',
        train=False
    )
    test_dataset = BasicDataset(
        images_dir=f'{config.root_dir}/{config.dataset_name}/test/image/',
        labels_dir=f'{config.root_dir}/{config.dataset_name}/test/label/',
        train=False
    )

    # 2. 创建数据加载器
    loader_args = dict(num_workers=8, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False,
                              batch_size=config.batch_size, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False,
                            batch_size=config.batch_size, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False,
                             batch_size=config.batch_size, **loader_args)

    # 3. 初始化日志记录
    OFFLINE_LOG_DIR = "./comet_logs"
    os.makedirs(OFFLINE_LOG_DIR, exist_ok=True)

    # --- 方案A：在线模式 ---
    # experiment = Experiment(
    #     api_key=config.comet_ml.api_key,
    #     project_name=config.comet_ml.project_name,
    #     auto_metric_logging=False,
    #     auto_param_logging=False
    # )

    # --- 方案B：离线模式 ---
    experiment = OfflineExperiment(
        project_name=config.log_project_name,
        offline_directory=OFFLINE_LOG_DIR,
        auto_metric_logging=False,
        auto_param_logging=False
    )

    logging.info(f'''开始训练:
        配置文件:         {args.config}
        模型:            {config.model_classname}
        本次运行Epoch:    {config.epochs}
        批次大小:         {config.batch_size}
        学习率:           {config.learning_rate}
        设备:            {device.type}
    ''')

    # 4. 动态加载并设置模型
    logging.info(f"正在从 ./model/{config.model_filename}.py 加载模型...")
    model_module = importlib.import_module(f'model.{config.model_filename}')
    ModelClass = getattr(model_module, config.model_classname)
    net = ModelClass(**vars(config.model_args))
    net.to(device=device)

    # 5. 设置优化器, 调度器, 损失函数等
    training_mode = 'train_all'
    if hasattr(config, 'training_strategy') and hasattr(config.training_strategy, 'mode'):
        training_mode = config.training_strategy.mode

    logging.info("--- 设置训练模式 ---")
    if training_mode == 'adapter_only':
        logging.info("模式：冻结SAM编码器主体，但训练Adapter、FPN模块和解码器。")
        for name, param in net.named_parameters():
            if 'encoder' in name and 'dscg_adapter' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:  # 默认 'train_all' 或任何其他值
        logging.info("模式：训练网络中的所有参数。")
        for param in net.parameters():
            param.requires_grad = True

    params_to_update = filter(lambda p: p.requires_grad, net.parameters())

    num_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logging.info(f"总可训练参数量: {num_trainable_params / 1e6:.4f} M")

    optimizer = optim.AdamW(params_to_update, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = create_scheduler(optimizer, config)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
    criterion = FCCDN_loss_without_seg

    # 6. 初始化训练状态变量
    start_epoch = 1
    total_step = 0
    best_metrics = {'best_f1score': 0.0, 'lowest loss': float('inf'), 'best_test_f1score': 0.0}

    if config.load:
        if not os.path.exists(config.load):
            logging.warning(f"警告: 配置文件中指定的模型路径 '{config.load}' 不存在，将从头开始训练。")
        else:
            logging.info(f"正在从 {config.load} 加载模型...")
            checkpoint = torch.load(config.load, map_location=device)

            if 'net' in checkpoint and 'optimizer' in checkpoint:
                net.load_state_dict(checkpoint['net'])
                optimizer.load_state_dict(checkpoint['optimizer'])

                reset_scheduler = getattr(config, 'reset_scheduler_on_load', False)

                if 'scheduler' in checkpoint and not reset_scheduler:
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    logging.info("已成功加载 Scheduler 状态，将继续之前的学习率曲线。")
                else:
                    logging.info(f"重置优化器学习率至初始值: {config.learning_rate}")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = config.learning_rate

                    # 根据配置重新创建调度器
                    scheduler = create_scheduler(optimizer, config)
                    logging.info(f"学习率调度器已重置。")

                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                if 'total_step' in checkpoint:
                    total_step = checkpoint['total_step']

                if 'best_metrics' in checkpoint:
                    best_metrics = checkpoint['best_metrics']
                    logging.info(f"已成功加载历史最佳指标: Best F1={best_metrics['best_f1score']:.4f}")
                else:
                    logging.info("Checkpoint中未找到历史最佳指标，将从0开始记录。")

                if optimizer.param_groups[0].get('capturable', False):
                    optimizer.param_groups[0]['capturable'] = True

                logging.info(f"已成功加载 Checkpoint (模型和优化器)。")
            else:
                net.load_state_dict(checkpoint)
                logging.info(f'已成功加载模型权重 (仅 state_dict)，将从头开始训练优化器。')

    # 7. 设置评估指标
    metric_collection = MetricCollection({
        'accuracy': Accuracy(task="binary").to(device=device),
        'precision': Precision(task="binary").to(device=device),
        'recall': Recall(task="binary").to(device=device),
        'f1score': F1Score(task="binary").to(device=device),
        'iou': CustomIoU().to(device=device)
    })
    to_pilimg = T.ToPILImage()

    # 8. 设置检查点路径
    project_name = f'{config.log_project_name}_{config.image_size}_{config.learning_rate}'
    base_log_path = f'./logs_weights/{project_name}'
    checkpoint_path = f'{base_log_path}_checkpoint/'
    best_f1score_model_path = f'{base_log_path}_best_f1score_model/'
    best_loss_model_path = f'{base_log_path}_best_loss_model/'
    best_f1score_test_model_path = f'{base_log_path}_best_f1score_test_model/'
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(best_f1score_model_path, exist_ok=True)
    os.makedirs(best_loss_model_path, exist_ok=True)
    os.makedirs(best_f1score_test_model_path, exist_ok=True)

    # 9. 准备通用的 forward 参数
    forward_kwargs = {}
    if hasattr(config, 'forward_args'):
        forward_kwargs = vars(config.forward_args)

    # 10. 开始训练循环
    end_epoch = start_epoch + config.epochs
    logging.info(f"训练计划：将从 Epoch {start_epoch} 运行到 Epoch {end_epoch - 1}。")

    for epoch in range(start_epoch, end_epoch):
        # 训练阶段
        experiment, net, optimizer, grad_scaler, total_step, _ = \
            train_val_test(
                mode='train', dataset_name=config.dataset_name,
                dataloader=train_loader, device=device, experiment=experiment, net=net,
                optimizer=optimizer, total_step=total_step, lr=config.learning_rate, criterion=criterion,
                metric_collection=metric_collection, to_pilimg=to_pilimg, epoch=epoch,
                grad_scaler=grad_scaler, config=config,
                forward_kwargs=forward_kwargs
            )

        # 验证与测试阶段
        if epoch >= config.evaluate_epoch and epoch % config.evaluate_inteval == 0:
            logging.info('开始验证...')
            with torch.no_grad():
                experiment, net, optimizer, total_step, _, best_metrics, val_metrics = \
                    train_val_test(
                        mode='val', dataset_name=config.dataset_name,
                        dataloader=val_loader, device=device, experiment=experiment, net=net,
                        optimizer=optimizer, total_step=total_step, lr=config.learning_rate, criterion=criterion,
                        metric_collection=metric_collection, to_pilimg=to_pilimg, epoch=epoch,
                        best_metrics=best_metrics, checkpoint_path=checkpoint_path,
                        best_f1score_model_path=best_f1score_model_path, best_loss_model_path=best_loss_model_path,
                        scheduler=scheduler, config=config,
                        forward_kwargs=forward_kwargs
                    )

                # --- 测试阶段 ---
                logging.info('开始测试...')
                experiment, net, optimizer, total_step, _, best_metrics, _ = \
                    train_val_test(
                        mode='test', dataset_name=config.dataset_name,
                        dataloader=test_loader, device=device, experiment=experiment, net=net,
                        optimizer=optimizer, total_step=total_step, lr=config.learning_rate, criterion=criterion,
                        metric_collection=metric_collection, to_pilimg=to_pilimg, epoch=epoch,
                        best_metrics=best_metrics,
                        best_f1score_model_path=best_f1score_test_model_path,
                        checkpoint_path=None, best_loss_model_path=None,
                        scheduler=scheduler, config=config,
                        forward_kwargs=forward_kwargs
                    )

        if scheduler is not None and total_step >= config.warm_up_step:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if epoch >= config.evaluate_epoch and epoch % config.evaluate_inteval == 0:
                    scheduler.step(val_metrics['f1score'])
                    logging.info(
                        f"Epoch {epoch}: ReduceLROnPlateau step taken with F1-score: {val_metrics['f1score']:.4f}")
            else:
                scheduler.step()
                logging.info(f"Epoch {epoch}: Scheduler step taken. New LR: {scheduler.get_last_lr()[0]:.1e}")
    experiment.end()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser(description='通用模型训练脚本')
    parser.add_argument('-c', '--config', default='config/Anchor-SAM.yaml',
                        help='配置文件的路径', dest='config')
    args = parser.parse_args()
    config = load_config(args.config)

    try:
        train_net(config)
    except KeyboardInterrupt:
        logging.info('Interrupt')
        sys.exit(0)
    except Exception as e:
        logging.error(f"训练过程中发生错误: {e}", exc_info=True)
        sys.exit(1)

