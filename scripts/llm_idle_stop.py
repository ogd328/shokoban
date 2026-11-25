"""使っていない手元AI（llama-server）を止める。

1分おきなどに定期的に呼ぶ想定（macOS なら LaunchAgent、ほかは cron など。見本は
``deploy/macos-launchd/shokoban-llm-idle-stop.example.plist``）。出力は1行だけ。
起動台帳（llama-server.json）で、書庫番が起こしたと確認できたプロセスだけを止める
（確認できなければ止めない）。最後に使ってから ``runtime.json`` の ``idle_minutes`` を
過ぎていれば止める。

使い方::

    python scripts/llm_idle_stop.py --config <作業フォルダ>/runtime.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from shokoban.llm_runtime import (
    LLMRuntimeError,
    default_config_path,
    idle_seconds,
    is_running,
    load_runtime_config,
    stop,
)


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="使っていない手元AIを止めます（書庫番が起こしたものだけ）。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="runtime.json の場所（省略すると SHOKOBAN_WORK_DIR/runtime.json）",
    )
    args = parser.parse_args(argv)
    config_path = args.config if args.config is not None else default_config_path()

    try:
        cfg = load_runtime_config(config_path)
    except LLMRuntimeError as exc:
        print(f"設定を読めないので何もしません: {exc}")
        return 2

    try:
        if not is_running(cfg):
            print("手元AIは動いていません")
            return 0
        idle = idle_seconds(cfg)
    except LLMRuntimeError as exc:
        print(f"手元AIの状態を確認できないので何もしません: {exc}")
        return 2

    if idle is None:
        print("手元AIは動いています（最終使用の記録がないので止めません）")
        return 0
    if idle <= cfg.idle_minutes * 60:
        print(f"手元AIは動いています（アイドル{_minutes(idle)}分）")
        return 0

    try:
        stopped = stop(cfg)
    except LLMRuntimeError as exc:
        print(f"手元AIを止められませんでした: {exc}")
        return 1

    if stopped:
        print(f"手元AIを止めました（アイドル{_minutes(idle)}分）")
    else:
        print("手元AIはもう動いていません（記録を消しました）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
