from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.footx_model.worldcup_agent import build_worldcup_graph

OUTPUT_DIR = Path("outputs")
PREDICTIONS_PATH = OUTPUT_DIR / "worldcup_predictions.csv"
REPORT_PATH = OUTPUT_DIR / "worldcup_report.json"

GROUP_LABELS = {
    "Group A": "A组",
    "Group B": "B组",
    "Group C": "C组",
    "Group D": "D组",
    "Group E": "E组",
    "Group F": "F组",
    "Group G": "G组",
    "Group H": "H组",
    "Group I": "I组",
    "Group J": "J组",
    "Group K": "K组",
    "Group L": "L组",
}
PICK_LABELS = {"H": "队伍1胜", "D": "平局", "A": "队伍2胜"}
COLUMN_LABELS = {
    "date": "日期",
    "time": "时间",
    "group": "小组",
    "team1": "队伍1",
    "team2": "队伍2",
    "ground": "比赛地",
    "prob_H": "队伍1胜率",
    "prob_D": "平局概率",
    "prob_A": "队伍2胜率",
    "pick": "模型方向",
    "odds_team1": "队伍1欧赔",
    "odds_draw": "平局欧赔",
    "odds_team2": "队伍2欧赔",
    "market_prob_H": "市场队伍1胜率",
    "market_prob_D": "市场平局概率",
    "market_prob_A": "市场队伍2胜率",
    "final_prob_H": "融合队伍1胜率",
    "final_prob_D": "融合平局概率",
    "final_prob_A": "融合队伍2胜率",
    "final_pick": "融合方向",
}


st.set_page_config(page_title="CupOracle 世界杯预测", layout="wide")
st.title("CupOracle 世界杯预测 Agent")
st.caption("查看 2026 世界杯小组赛胜平负概率，并可一键重新运行预测。")


def _load_report() -> dict:
    if not REPORT_PATH.exists():
        return {}
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(PREDICTIONS_PATH)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    return df


with st.sidebar:
    st.header("操作")
    odds_api_key = st.text_input(
        "The Odds API Key（可选）",
        type="password",
        help="填入后重新运行预测，会尝试拉取实时欧赔并生成融合概率。",
    )
    if st.button("重新运行世界杯预测", type="primary"):
        if odds_api_key.strip():
            os.environ["THE_ODDS_API_KEY"] = odds_api_key.strip()
        with st.spinner("正在抓取数据、训练模型并生成预测..."):
            result = build_worldcup_graph().invoke({"output_dir": str(OUTPUT_DIR)})
        st.success("预测已更新。")
        st.json(result.get("metrics", {}))

    st.divider()
    st.write("输出文件")
    st.code(str(PREDICTIONS_PATH))
    st.code(str(REPORT_PATH))


report = _load_report()
predictions = _load_predictions()

if report:
    st.subheader("模型评估")
    cols = st.columns(5)
    cols[0].metric("命中率", f"{report.get('accuracy', 0):.4f}")
    cols[1].metric("概率误差", f"{report.get('log_loss', 0):.4f}")
    cols[2].metric("训练样本", f"{int(report.get('train_rows', 0)):,}")
    cols[3].metric("测试样本", f"{int(report.get('test_rows', 0)):,}")
    cols[4].metric("预测比赛", f"{int(report.get('fixture_rows', 0)):,}")
    if "odds_rows" in report:
        odds_cols = st.columns(2)
        odds_cols[0].metric("实时赔率比赛", f"{int(report.get('odds_rows', 0)):,}")
        odds_cols[1].metric("成功匹配赔率", f"{int(report.get('odds_matched_rows', 0)):,}")
else:
    st.info("还没有训练报告。点击左侧“重新运行世界杯预测”生成结果。")

st.subheader("比赛预测")
if predictions.empty:
    st.warning("还没有预测文件。点击左侧按钮先运行一次预测。")
else:
    raw_groups = sorted(predictions["group"].dropna().unique().tolist())
    group_display = {"全部": "全部", **{GROUP_LABELS.get(g, g): g for g in raw_groups}}
    group_label = st.selectbox("按小组筛选", list(group_display.keys()))
    team_query = st.text_input("按球队搜索", "")

    view = predictions.copy()
    selected_group = group_display[group_label]
    if selected_group != "全部":
        view = view[view["group"] == selected_group]
    if team_query.strip():
        q = team_query.strip().lower()
        view = view[
            view["team1"].str.lower().str.contains(q, na=False)
            | view["team2"].str.lower().str.contains(q, na=False)
        ]

    display_cols = [
        "date",
        "time",
        "group",
        "team1",
        "team2",
        "ground",
        "prob_H",
        "prob_D",
        "prob_A",
        "pick",
        "odds_team1",
        "odds_draw",
        "odds_team2",
        "market_prob_H",
        "market_prob_D",
        "market_prob_A",
        "final_prob_H",
        "final_prob_D",
        "final_prob_A",
        "final_pick",
    ]
    existing_cols = [c for c in display_cols if c in view.columns]
    display_view = view[existing_cols].copy()
    if "group" in display_view.columns:
        display_view["group"] = display_view["group"].map(lambda v: GROUP_LABELS.get(v, v))
    for pick_col in ["pick", "final_pick"]:
        if pick_col in display_view.columns:
            display_view[pick_col] = display_view[pick_col].map(lambda v: PICK_LABELS.get(v, v))
    for prob_col in [
        "prob_H",
        "prob_D",
        "prob_A",
        "market_prob_H",
        "market_prob_D",
        "market_prob_A",
        "final_prob_H",
        "final_prob_D",
        "final_prob_A",
    ]:
        if prob_col in display_view.columns:
            display_view[prob_col] = pd.to_numeric(display_view[prob_col], errors="coerce")
            display_view[prob_col] = display_view[prob_col].map(
                lambda v: "" if pd.isna(v) else f"{v * 100:.2f}%"
            )
    display_view = display_view.rename(columns=COLUMN_LABELS)
    st.dataframe(display_view, use_container_width=True, hide_index=True)

    csv_bytes = predictions.to_csv(index=False).encode("utf-8")
    st.download_button(
        "下载完整预测 CSV",
        data=csv_bytes,
        file_name="worldcup_predictions.csv",
        mime="text/csv",
    )

    with st.expander("字段说明"):
        st.write(
            "队伍1胜率/平局概率/队伍2胜率：基础模型概率；"
            "市场概率：实时欧赔去水后的隐含概率；"
            "融合概率：基础模型与市场概率加权后的结果；"
            "融合方向：融合概率最高的结果。"
        )
