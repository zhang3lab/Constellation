from __future__ import annotations

def log_enabled(log_level: int, level: int) -> bool:
    return int(log_level) >= int(level)

def log(log_level: int, level: int, msg: str) -> None:
    if log_enabled(log_level, level):
        print(msg)

def log1(log_level: int, msg: str) -> None:
    if int(log_level) >= 1:
        print(msg)

def log2(log_level: int, msg: str) -> None:
    if int(log_level) >= 2:
        print(msg)
