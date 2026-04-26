from __future__ import annotations

import argparse

from src.footx_model.worldcup_agent import build_worldcup_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="CupOracle 免费公开数据版世界杯预测 Agent")
    parser.add_argument("--output-dir", default="outputs", help="输出目录")
    args = parser.parse_args()

    app = build_worldcup_graph()
    result = app.invoke({"output_dir": args.output_dir})
    print("CupOracle 世界杯预测 Agent 完成。")
    print("metrics:", result.get("metrics"))
    print("predictions:", f"{args.output_dir}/worldcup_predictions.csv")
    print("report:", result.get("report_path"))


if __name__ == "__main__":
    main()

