"""srtp_memory — 多智能体记忆共享调度系统核心包（headless，不依赖 UI）。"""

from pathlib import Path

# 加载项目根目录 .env（敏感密钥，已 gitignore，不进仓库）
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:  # pragma: no cover — 未装 dotenv 时仅依赖系统环境变量
    pass

__version__ = "0.1.0"
