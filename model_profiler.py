# ----------------------------------------------------------------
# 模型性能综合测试脚本
#
# 功能:
# 1. 计算模型的参数量 (总参数量 & 可训练参数量)
# 2. 估算模型的计算量 (GMACs 和 GFLOPs)
# 3. 测试模型在GPU上的FPS (每秒推理帧数)
#
# 使用方法:
# 1. 修改下方 `CONFIG_FILE_PATH` 变量，指向要测试的配置文件。
# 2. 根据需要修改 `INPUT_HEIGHT` 和 `INPUT_WIDTH`。
# 3. 直接运行脚本: `python test_model_performance.py`
#
# 依赖安装:
# pip install pyyaml torch torchvision thop prettytable
# ----------------------------------------------------------------

import sys
import time
import yaml
import logging
import importlib
import inspect
import os

class MockModule:
    def __getattr__(self, name):
        return MockModule()

sys.modules['flax'] = MockModule()
sys.modules['flax.linen'] = MockModule()
sys.modules['jax'] = MockModule()
sys.modules['jax.numpy'] = MockModule()
# -----------------------------

import torch
from types import SimpleNamespace
from thop import profile
from prettytable import PrettyTable

original_torch_load = torch.load
PROJECT_ROOT = os.getcwd()


def patched_torch_load(f, *args, **kwargs):
    if isinstance(f, str) and f.startswith('/pretrain_weight/'):
        corrected_path = os.path.join(PROJECT_ROOT, f.lstrip('/'))
        logging.info(f"检测到硬编码的预训练权重路径。已将其动态修正：")
        logging.info(f"  原始路径: {f}")
        logging.info(f"  修正后路径: {corrected_path}")
        return original_torch_load(corrected_path, *args, **kwargs)
    return original_torch_load(f, *args, **kwargs)


torch.load = patched_torch_load
CALCULATE_PARAMS = True
CALCULATE_FLOPS = True
MEASURE_FPS = False

# --- FPS 测试专用配置 ---
FPS_WARM_UP_RUNS = 10
FPS_TEST_RUNS = 100

# --- 默认输入图像尺寸 ---
INPUT_CHANNELS = 3
INPUT_HEIGHT = 1024
INPUT_WIDTH = 1024
# -----------------------------
CONFIG_FILE_PATH = 'config/CMTFNet.yaml'
# ------------------------------------


def load_config(config_path):
    """
    加载并解析YAML配置文件为命名空间, 与 train.py 中的函数一致。
    """
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
    except Exception as e:
        logging.error(f"加载或解析配置文件 '{config_path}' 时出错: {e}")
        sys.exit(1)


def initialize_model_from_config(config):
    """
    根据配置文件动态加载并初始化模型。
    会自动检查模型的__init__方法，只传入其接受的参数。
    """
    logging.info(f"正在从 ./model/{config.model_filename}.py 加载模型...")
    try:
        model_module = importlib.import_module(f'model.{config.model_filename}')
        ModelClass = getattr(model_module, config.model_classname)
    except ModuleNotFoundError:
        logging.error(f"错误: 找不到模型文件 ./model/{config.model_filename}.py。请检查文件名是否正确。")
        sys.exit(1)
    except AttributeError:
        logging.error(f"错误: 在 {config.model_filename}.py 中找不到名为 '{config.model_classname}' 的类。")
        sys.exit(1)

    # 1. 获取模型构造函数__init__接受的所有参数名
    init_signature = inspect.signature(ModelClass.__init__)
    valid_params = init_signature.parameters.keys()

    # 2. 从配置文件中获取原始参数
    config_model_args = vars(config.model_args) if hasattr(config, 'model_args') else {}

    # 3. 过滤出模型真正接受的参数
    final_model_args = {}
    for arg_name, arg_value in config_model_args.items():
        if arg_name in valid_params:
            final_model_args[arg_name] = arg_value
        else:
            logging.warning(
                f"配置文件中的参数 '{arg_name}' 不被模型 '{config.model_classname}' 的构造函数接受，将被忽略。")

    # 4. 智能地按需提供 n_channels 和 n_classes 的默认值
    # 仅当模型需要这些参数但配置文件中未提供时，才添加
    if 'n_channels' in valid_params and 'n_channels' not in final_model_args:
        logging.warning(f"模型需要 'n_channels' 但配置文件中未提供，将使用默认值: {INPUT_CHANNELS}")
        final_model_args['n_channels'] = INPUT_CHANNELS

    if 'n_classes' in valid_params and 'n_classes' not in final_model_args:
        logging.warning(f"模型需要 'n_classes' 但配置文件中未提供，将使用默认值: 1")
        final_model_args['n_classes'] = 1

    logging.info(f"正在使用以下有效参数实例化模型 '{config.model_classname}': {final_model_args}")

    try:
        model = ModelClass(**final_model_args)
        logging.info("模型实例化成功!")
        return model
    except Exception as e:
        logging.error(f"实例化模型 '{config.model_classname}' 时出错: {e}", exc_info=True)
        sys.exit(1)


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 1. 加载配置和模型
    logging.info(f"正在从脚本中加载指定的配置文件: {CONFIG_FILE_PATH}")
    config = load_config(CONFIG_FILE_PATH)
    model = initialize_model_from_config(config)

    # 2. 设置设备并准备输入
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"测试将在设备 '{device.type}' 上运行。")

    model.to(device)
    model.eval()  # 切换到评估模式

    # 创建一个符合模型输入的虚拟张量
    dummy_input = torch.randn(1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH).to(device)
    logging.info(f"已创建虚拟输入张量, 尺寸: {list(dummy_input.shape)}")

    # 3. 创建结果表格
    table = PrettyTable()
    table.field_names = ["Metric", "Value", "Unit"]
    table.align["Metric"] = "l"
    table.align["Value"] = "r"
    table.align["Unit"] = "l"
    table.title = f"Performance Analysis for {config.model_classname}"

    # 4. 执行选择的测试
    # --- 参数量测试 ---
    if CALCULATE_PARAMS:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        table.add_row(["Total Params", f"{total_params / 1e6:.2f}", "M"])
        table.add_row(["Trainable Params", f"{trainable_params / 1e6:.2f}", "M"])
        logging.info("参数量计算完成。")

    # --- 计算量测试 (MACs & FLOPs) ---
    if CALCULATE_FLOPS:
        try:
            # 使用 thop.profile 计算 MACs (Multiply-Accumulate operations)
            macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
            # FLOPs (Floating Point Operations) 约等于 2 * MACs
            flops = macs * 2
            table.add_row(["GMACs", f"{macs / 1e9:.2f}", "G"])
            table.add_row(["GFLOPs", f"{flops / 1e9:.2f}", "G"])
            logging.info("计算量 (MACs/FLOPs) 计算完成。")
        except Exception as e:
            logging.error(f"使用 thop 计算 FLOPs 时出错: {e}")
            table.add_row(["GMACs", "Calculation Failed", "N/A"])
            table.add_row(["GFLOPs", "Calculation Failed", "N/A"])

    # --- FPS 测试 ---
    if MEASURE_FPS:
        if device.type == 'cpu':
            logging.warning("FPS 测试在 CPU 上可能不准确，建议在 CUDA 环境下运行。")

        with torch.no_grad():
            # 预热阶段
            logging.info(f"正在进行 {FPS_WARM_UP_RUNS} 次预热运行...")
            for _ in range(FPS_WARM_UP_RUNS):
                _ = model(dummy_input)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # 正式测试阶段
            logging.info(f"正在进行 {FPS_TEST_RUNS} 次正式测试...")
            start_time = time.perf_counter()
            for _ in range(FPS_TEST_RUNS):
                _ = model(dummy_input)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            total_time = end_time - start_time
            avg_time_per_frame = total_time / FPS_TEST_RUNS
            fps = 1.0 / avg_time_per_frame

            table.add_row(["Avg Inference Time", f"{avg_time_per_frame * 1000:.2f}", "ms"])
            table.add_row(["FPS", f"{fps:.2f}", "frames/sec"])
            logging.info("FPS 测试完成。")

    # 5. 打印最终结果
    print("\n" + "=" * 80)
    print(table)
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()

