"""配置加载：TOML 读取、路径解析、`.env` 读取与包内资源定位。

职责边界：

- `load_config()` 只负责「读 TOML → pydantic 校验 → 解析路径」，不读取 `.env`；
- `load_env()` 是显式的副作用，由 CLI / 库入口决定何时调用；
- 包内配置与提示词通过 `importlib.resources` 读取，兼容 zip 化安装。
"""

from __future__ import annotations

import importlib.resources
import os
import pathlib
import tomllib
from dataclasses import dataclass

from pydantic import ValidationError

from fastrag.config.models import Config
from fastrag.errors import FastragError

PACKAGE_RESOURCES = importlib.resources.files("fastrag.resources")

# editable 安装时 __file__ 位于 <repo>/src/fastrag/，据此识别源码树；
# 装成 wheel 后这里指向 site-packages，不再是项目根目录。
_SOURCE_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent


@dataclass(frozen=True, slots=True)
class ConfigRef:
    """配置文件来源：真实路径，或包内资源名。"""

    path: pathlib.Path | None = None
    resource: str | None = None


def home() -> pathlib.Path:
    """工作目录：数据、`.env`、自定义配置与提示词的根。

    优先级：`FASTRAG_HOME` > 源码树根目录（editable 运行）> 当前目录。
    """
    override = os.environ.get("FASTRAG_HOME")
    if override:
        return pathlib.Path(override).expanduser().resolve()
    if (_SOURCE_ROOT / "pyproject.toml").exists():
        return _SOURCE_ROOT
    return pathlib.Path.cwd()


def default_config_ref() -> ConfigRef:
    """默认配置：`FASTRAG_HOME/config/fastrag.toml`，不存在时用包内模板。"""
    user_config = home() / "config" / "fastrag.toml"
    if user_config.exists():
        return ConfigRef(path=user_config)
    return ConfigRef(resource="config/fastrag.toml")


def resolve_config(value: str) -> ConfigRef:
    """把 `--config` 的值解析成配置来源。

    优先当作路径；不存在时按包内配置名解析（`local` -> `config/fastrag.local.toml`）。
    `default` 表示走默认查找链。
    """
    candidate = pathlib.Path(value).expanduser()
    if candidate.exists():
        return ConfigRef(path=candidate.resolve())
    if value == "default":
        return default_config_ref()
    resource = f"config/fastrag.{value}.toml"
    if (PACKAGE_RESOURCES / resource).is_file():
        return ConfigRef(resource=resource)
    raise FastragError(f"配置文件不存在: {value}")


def endpoint_api_key(config: Config, endpoint: str) -> str:
    """取 endpoint 的 key；本地服务（api_key_env 为空）返回占位值。"""
    name = config.endpoints[endpoint].api_key_env
    if not name:
        return "local"
    value = os.environ.get(name)
    if not value:
        raise FastragError(f"缺少 {name}，请先 cp .env.example .env 并填写")
    return value


def display_path(path: pathlib.Path) -> str:
    """相对工作目录的展示路径；在目录外时退回绝对路径。"""
    try:
        return path.resolve().relative_to(home()).as_posix()
    except ValueError:
        return str(path.resolve())


def _env_value(raw: str) -> str:
    """解析 `.env` 的取值：去掉行尾注释，再去掉一层成对引号。"""
    quote = ""
    for index, char in enumerate(raw):
        if quote:
            if char == quote:
                quote = ""
        elif char in ("'", '"'):
            quote = char
        elif char == "#" and (index == 0 or raw[index - 1].isspace()):
            raw = raw[:index]
            break
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return value


def load_env(path: pathlib.Path | None = None) -> None:
    """把 `.env` 里的键值写进环境变量（不覆盖已存在的值），默认读工作目录。"""
    path = home() / ".env" if path is None else path
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), _env_value(value))


def _read_ref(ref: ConfigRef) -> str:
    """读取配置文本；包内资源不落临时文件，直接用 `Traversable.read_text`。"""
    if ref.path is not None:
        if not ref.path.exists():
            raise FastragError(f"配置文件不存在: {ref.path}")
        try:
            return ref.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise FastragError(f"读取配置失败: {ref.path}: {exc}") from None
    if ref.resource is not None:
        resource = PACKAGE_RESOURCES / ref.resource
        if not resource.is_file():
            raise FastragError(f"包内配置不存在: {ref.resource}")
        return resource.read_text(encoding="utf-8")
    raise FastragError("没有可用的配置文件")


def _resolve_paths(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise FastragError("配置缺少 [paths] 节")
    resolved: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(value, str):
            raise FastragError(f"paths.{key} 必须是字符串，当前为 {value!r}")
        resolved[key] = str((home() / value).expanduser().resolve())
    return resolved


def _format_error(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  - {location}: {error['msg']}")
    return "配置校验失败:\n" + "\n".join(lines)


def load_config(
    path: pathlib.Path | str | None = None,
    *,
    ref: ConfigRef | None = None,
) -> Config:
    """读取并校验统一配置。

    解析顺序：显式 `ref` > 显式 `path` > `FASTRAG_CONFIG` > 默认配置。
    """
    if ref is None:
        if path is not None:
            ref = ConfigRef(path=pathlib.Path(path).expanduser())
        else:
            override = os.environ.get("FASTRAG_CONFIG")
            if override:
                ref = ConfigRef(path=pathlib.Path(override).expanduser())
            else:
                ref = default_config_ref()

    text = _read_ref(ref)
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise FastragError(f"配置文件不是合法 TOML: {exc}") from None

    data = dict(raw)
    data["paths"] = _resolve_paths(data.get("paths"))
    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        raise FastragError(_format_error(exc)) from None
