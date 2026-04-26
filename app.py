from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.footx_model.league_agent import build_league_graph
from src.footx_model.league_crawler import DEFAULT_SEASONS, LEAGUES
from src.footx_model.worldcup_agent import build_worldcup_graph

OUTPUT_DIR = Path("outputs")
PREDICTIONS_PATH = OUTPUT_DIR / "worldcup_predictions.csv"
REPORT_PATH = OUTPUT_DIR / "worldcup_report.json"
LEAGUE_PICK_LABELS = {"H": "主胜", "D": "平局", "A": "客胜"}
RECOMMENDATION_LABELS = {
    "H": "主胜",
    "D": "平局",
    "A": "客胜",
    "1X": "主队不败",
    "X2": "客队不败",
    "12": "分胜负",
    "双选含平": "双选含平",
    "观望": "观望",
}
TEAM_CN_NOTES = {
    "Arsenal": "阿森纳",
    "Aston Villa": "阿斯顿维拉",
    "Bournemouth": "伯恩茅斯",
    "Brentford": "布伦特福德",
    "Brighton": "布莱顿",
    "Burnley": "伯恩利",
    "Chelsea": "切尔西",
    "Crystal Palace": "水晶宫",
    "Everton": "埃弗顿",
    "Fulham": "富勒姆",
    "Ipswich": "伊普斯维奇",
    "Leeds": "利兹联",
    "Leicester": "莱斯特城",
    "Liverpool": "利物浦",
    "Luton": "卢顿",
    "Man City": "曼城",
    "Man United": "曼联",
    "Newcastle": "纽卡斯尔",
    "Nott'm Forest": "诺丁汉森林",
    "Sheffield United": "谢菲联",
    "Southampton": "南安普顿",
    "Tottenham": "热刺",
    "Watford": "沃特福德",
    "West Brom": "西布朗",
    "West Ham": "西汉姆联",
    "Wolves": "狼队",
    "Alaves": "阿拉维斯",
    "Ath Bilbao": "毕尔巴鄂竞技",
    "Ath Madrid": "马德里竞技",
    "Barcelona": "巴塞罗那",
    "Betis": "贝蒂斯",
    "Celta": "塞尔塔",
    "Eibar": "埃瓦尔",
    "Elche": "埃尔切",
    "Espanol": "西班牙人",
    "Getafe": "赫塔菲",
    "Girona": "赫罗纳",
    "Granada": "格拉纳达",
    "Las Palmas": "拉斯帕尔马斯",
    "Mallorca": "马洛卡",
    "Osasuna": "奥萨苏纳",
    "Real Madrid": "皇家马德里",
    "Sevilla": "塞维利亚",
    "Sociedad": "皇家社会",
    "Valencia": "瓦伦西亚",
    "Villarreal": "比利亚雷亚尔",
    "Bayern Munich": "拜仁慕尼黑",
    "Dortmund": "多特蒙德",
    "Ein Frankfurt": "法兰克福",
    "Freiburg": "弗赖堡",
    "Hoffenheim": "霍芬海姆",
    "Leverkusen": "勒沃库森",
    "M'gladbach": "门兴",
    "Mainz": "美因茨",
    "RB Leipzig": "莱比锡",
    "Stuttgart": "斯图加特",
    "Union Berlin": "柏林联合",
    "Werder Bremen": "云达不莱梅",
    "Wolfsburg": "沃尔夫斯堡",
    "Atalanta": "亚特兰大",
    "Bologna": "博洛尼亚",
    "Cagliari": "卡利亚里",
    "Empoli": "恩波利",
    "Fiorentina": "佛罗伦萨",
    "Genoa": "热那亚",
    "Inter": "国际米兰",
    "Juventus": "尤文图斯",
    "Lazio": "拉齐奥",
    "Milan": "AC米兰",
    "Napoli": "那不勒斯",
    "Roma": "罗马",
    "Sassuolo": "萨索洛",
    "Torino": "都灵",
    "Udinese": "乌迪内斯",
    "Verona": "维罗纳",
    "Lens": "朗斯",
    "Lille": "里尔",
    "Lyon": "里昂",
    "Marseille": "马赛",
    "Monaco": "摩纳哥",
    "Montpellier": "蒙彼利埃",
    "Nantes": "南特",
    "Nice": "尼斯",
    "Paris SG": "巴黎圣日耳曼",
    "Rennes": "雷恩",
    "Strasbourg": "斯特拉斯堡",
    "Toulouse": "图卢兹",
}

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


st.set_page_config(page_title="CupOracle", layout="wide")
st.title("CupOracle")
st.caption("世界杯预测 + 五大联赛近期回测。")


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


def _team_with_cn_note(name: object) -> object:
    if pd.isna(name):
        return name
    value = str(name)
    note = TEAM_CN_NOTES.get(value)
    return f"{value}（{note}）" if note else value


def _to_beijing_time(value: object) -> object:
    if pd.isna(value):
        return value
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return value
    return timestamp.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M 北京时间")


def _future_recommendation_reason(row: pd.Series) -> str:
    recommendation = str(row.get("recommendation", ""))
    upset_risk = row.get("upset_risk", "")
    value_pick = row.get("value_pick", "")
    market_agreement = row.get("market_agreement", "")
    final_pick = LEAGUE_PICK_LABELS.get(row.get("final_pick"), row.get("final_pick", ""))
    strength_pick = LEAGUE_PICK_LABELS.get(row.get("strength_pick"), row.get("strength_pick", ""))
    market_pick = LEAGUE_PICK_LABELS.get(row.get("market_pick"), row.get("market_pick", ""))

    reasons = []
    if upset_risk and not pd.isna(upset_risk):
        reasons.append(f"爆冷风险：{upset_risk}")
    if market_agreement is False or str(market_agreement).lower() == "false":
        reasons.append(f"实力方向{strength_pick}，市场方向{market_pick}，方向不一致")
    if value_pick and not pd.isna(value_pick) and value_pick not in {"无明显价值", "无赔率"}:
        reasons.append(f"价值方向：{LEAGUE_PICK_LABELS.get(value_pick, value_pick)}")

    if recommendation == "观望":
        base_reason = "；".join(reasons) if reasons else "概率优势不足，保守口径原本观望"
        return f"激进方向：{final_pick}；{base_reason}"
    return f"推荐方向：{RECOMMENDATION_LABELS.get(recommendation, recommendation)}，模型主方向：{final_pick}"


def _aggressive_recommendation_label(row: pd.Series) -> str:
    recommendation = row.get("recommendation", "")
    if recommendation == "观望":
        final_pick = row.get("final_pick", "")
        label = LEAGUE_PICK_LABELS.get(final_pick, final_pick)
        return f"{label}（激进）" if label else "观望"
    return RECOMMENDATION_LABELS.get(recommendation, recommendation)


with st.sidebar:
    st.header("通用操作")
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


worldcup_tab, league_tab = st.tabs(["世界杯预测", "五大联赛回测"])

with worldcup_tab:
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
        for team_col in ["team1", "team2"]:
            if team_col in display_view.columns:
                display_view[team_col] = display_view[team_col].map(_team_with_cn_note)
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

with league_tab:
    st.subheader("五大联赛近期回测")
    st.write("用于在当前正在进行的赛季上验证模型表现：训练较早比赛，回测最近 N 场已结束比赛。")
    c1, c2, c3, c4 = st.columns(4)
    league_label_to_code = {"五大联赛合并": "ALL", **{name: code for code, name in LEAGUES.items()}}
    league_name = c1.selectbox("选择联赛", list(league_label_to_code.keys()))
    holdout = c2.number_input("最近回测场数", min_value=20, max_value=200, value=80, step=10)
    mode_label = c3.selectbox("预测模式", ["稳健模式", "激进爆冷模式"])
    prediction_mode = "aggressive" if mode_label == "激进爆冷模式" else "steady"
    run_league = c4.button("运行联赛回测", type="primary")

    league_code = league_label_to_code[league_name]
    league_report_path = OUTPUT_DIR / f"league_{league_code}_report.json"
    league_pred_path = OUTPUT_DIR / f"league_{league_code}_backtest_predictions.csv"
    league_upcoming_path = OUTPUT_DIR / f"league_{league_code}_upcoming_predictions.csv"

    if run_league:
        if odds_api_key.strip():
            os.environ["THE_ODDS_API_KEY"] = odds_api_key.strip()
        with st.spinner(f"正在抓取{league_name}数据并回测..."):
            result = build_league_graph().invoke(
                {
                    "league_code": league_code,
                    "seasons": DEFAULT_SEASONS,
                    "holdout": int(holdout),
                    "prediction_mode": prediction_mode,
                    "output_dir": str(OUTPUT_DIR),
                }
            )
        st.success("联赛回测完成。")
        st.json(result.get("metrics", {}))

    if league_report_path.exists() and league_pred_path.exists():
        league_report = json.loads(league_report_path.read_text(encoding="utf-8"))
        league_predictions = pd.read_csv(league_pred_path)
        cols = st.columns(5)
        cols[0].metric("联赛", league_report.get("league_name", league_name))
        cols[1].metric("融合命中率", f"{league_report.get('accuracy', 0):.4f}")
        cols[2].metric("实力命中率", f"{league_report.get('strength_accuracy', 0):.4f}")
        cols[3].metric("训练场次", f"{int(league_report.get('train_rows', 0)):,}")
        cols[4].metric("回测场次", f"{int(league_report.get('test_rows', 0)):,}")
        extra_cols = st.columns(3)
        extra_cols[0].metric("推荐场次", f"{int(league_report.get('recommendation_rows', 0)):,}")
        extra_cols[1].metric("推荐命中率", f"{league_report.get('recommendation_accuracy', 0):.4f}")
        extra_cols[2].metric("未来比赛预测场次", f"{int(league_report.get('upcoming_rows', 0)):,}")

        st.markdown("### 最近已结束比赛回测")
        backtest_cols = [
            "date",
            "league_name",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "result",
            "market_pick",
            "strength_prob_H",
            "strength_prob_D",
            "strength_prob_A",
            "strength_pick",
            "final_prob_H",
            "final_prob_D",
            "final_prob_A",
            "final_pick",
            "upset_risk",
            "value_pick",
            "market_agreement",
            "process_delta_diff",
            "recommendation",
            "recommendation_type",
            "recommendation_confidence",
            "recommendation_hit",
            "final_hit",
        ]
        for col in backtest_cols:
            if col not in league_predictions.columns:
                league_predictions[col] = pd.NA
        show = league_predictions[backtest_cols].copy()
        for team_col in ["home_team", "away_team"]:
            show[team_col] = show[team_col].map(_team_with_cn_note)
        for pick_col in ["result", "market_pick", "strength_pick", "final_pick", "value_pick"]:
            show[pick_col] = show[pick_col].map(lambda v: LEAGUE_PICK_LABELS.get(v, v))
        show["market_agreement"] = show["market_agreement"].map({True: "一致", False: "不一致"})
        show["recommendation_hit"] = show["recommendation_hit"].map({True: "命中", False: "未中"})
        show["final_hit"] = show["final_hit"].map({True: "命中", False: "未中"})
        show["recommendation"] = show.apply(_aggressive_recommendation_label, axis=1)
        for prob_col in [
            "strength_prob_H",
            "strength_prob_D",
            "strength_prob_A",
            "final_prob_H",
            "final_prob_D",
            "final_prob_A",
        ]:
            show[prob_col] = pd.to_numeric(show[prob_col], errors="coerce").map(
                lambda v: "" if pd.isna(v) else f"{v * 100:.2f}%"
            )
        show = show.rename(
            columns={
                "date": "日期",
                "league_name": "联赛",
                "home_team": "主队",
                "away_team": "客队",
                "home_goals": "主队进球",
                "away_goals": "客队进球",
                "result": "赛果",
                "market_pick": "市场方向",
                "strength_prob_H": "实力主胜概率",
                "strength_prob_D": "实力平局概率",
                "strength_prob_A": "实力客胜概率",
                "strength_pick": "实力方向",
                "final_prob_H": "融合主胜概率",
                "final_prob_D": "融合平局概率",
                "final_prob_A": "融合客胜概率",
                "final_pick": "融合方向",
                "upset_risk": "爆冷风险",
                "value_pick": "价值方向",
                "market_agreement": "是否跟市场一致",
                "process_delta_diff": "过程修正差",
                "recommendation": "推荐",
                "recommendation_type": "推荐类型",
                "recommendation_confidence": "推荐信心",
                "recommendation_hit": "推荐是否命中",
                "final_hit": "是否命中",
            }
        )
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.download_button(
            "下载联赛回测 CSV",
            data=league_predictions.to_csv(index=False).encode("utf-8"),
            file_name=f"league_{league_code}_backtest_predictions.csv",
            mime="text/csv",
        )

        st.markdown("### 未来比赛预测")
        if league_upcoming_path.exists() and int(league_report.get("upcoming_rows", 0)) > 0:
            upcoming = pd.read_csv(league_upcoming_path)
            upcoming_cols = [
                "commence_time",
                "league_name",
                "home_team",
                "away_team",
                "odds_home",
                "odds_draw",
                "odds_away",
                "final_prob_H",
                "final_prob_D",
                "final_prob_A",
                "final_pick",
                "strength_pick",
                "market_pick",
                "upset_risk",
                "value_pick",
                "market_agreement",
                "recommendation",
            ]
            for col in upcoming_cols:
                if col not in upcoming.columns:
                    upcoming[col] = pd.NA
            future_show = upcoming[upcoming_cols].copy()
            if "commence_time" in future_show.columns:
                future_show["commence_time"] = future_show["commence_time"].map(_to_beijing_time)
            for team_col in ["home_team", "away_team"]:
                future_show[team_col] = future_show[team_col].map(_team_with_cn_note)
            future_show["risk_reason"] = future_show.apply(_future_recommendation_reason, axis=1)
            future_show["recommendation"] = future_show.apply(_aggressive_recommendation_label, axis=1)
            for prob_col in ["final_prob_H", "final_prob_D", "final_prob_A"]:
                future_show[prob_col] = pd.to_numeric(future_show[prob_col], errors="coerce").map(
                    lambda v: "" if pd.isna(v) else f"{v * 100:.2f}%"
                )
            future_show = future_show[
                [
                    "commence_time",
                    "league_name",
                    "home_team",
                    "away_team",
                    "odds_home",
                    "odds_draw",
                    "odds_away",
                    "final_prob_H",
                    "final_prob_D",
                    "final_prob_A",
                    "recommendation",
                    "risk_reason",
                ]
            ]
            future_show = future_show.rename(
                columns={
                    "commence_time": "北京时间",
                    "league_name": "联赛",
                    "home_team": "主队",
                    "away_team": "客队",
                    "odds_home": "主胜欧赔",
                    "odds_draw": "平局欧赔",
                    "odds_away": "客胜欧赔",
                    "final_prob_H": "主胜概率",
                    "final_prob_D": "平局概率",
                    "final_prob_A": "客胜概率",
                    "recommendation": "推荐",
                    "risk_reason": "风险理由",
                }
            )
            st.dataframe(future_show, use_container_width=True, hide_index=True)
            st.download_button(
                "下载未来比赛预测 CSV",
                data=upcoming.to_csv(index=False).encode("utf-8"),
                file_name=f"league_{league_code}_upcoming_predictions.csv",
                mime="text/csv",
            )
        else:
            st.info("暂未获取到未来比赛。请在左侧填写 The Odds API Key 后重新运行；如果仍为 0，说明该联赛当前 API 暂无未来赔率。")
    else:
        st.info("选择联赛后点击“运行联赛回测”。")
