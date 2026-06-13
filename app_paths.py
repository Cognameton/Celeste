from __future__ import annotations

import os
import sys


def runtime_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def resource_root() -> str:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return os.path.abspath(str(meipass))
    return runtime_root()


def resource_path(*parts: str) -> str:
    return os.path.join(resource_root(), *parts)


def llama_install_dir() -> str:
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "Celeste", "llama")


def installed_llama_server_path() -> str:
    return os.path.join(llama_install_dir(), "llama-server")


def default_config_path() -> str:
    if not getattr(sys, "frozen", False):
        return os.path.join(runtime_root(), "config.yaml")

    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "Celeste", "config.yaml")

    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "Celeste", "config.yaml")
