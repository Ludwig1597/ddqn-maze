"""项目配置加载工具。"""

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"


def load_config(config_path=None):
    """读取 YAML 配置并返回 Python 字典。

    Args:
        config_path: YAML 文件路径。省略时读取 configs/base.yaml。
                     相对路径按项目根目录解析。

    Returns:
        dict: 配置内容。
    """
    if config_path is None:
        path = DEFAULT_CONFIG_PATH
    else:
        path = Path(config_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")

    required_sections = ("environment", "agent", "training", "evaluation")
    missing = [key for key in required_sections if key not in config]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")

    return config
