from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import joblib
import numpy as np
import pandas as pd
from langgraph.graph import END, StateGraph
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

from .worldcup_crawler import fetch_international_results, fetch_worldcup_2026_fixtures
from .worldcup_odds import enrich_predictions_with_odds


class WorldCupState(TypedDict, total=False):
    output_dir: str
    history: pd.DataFrame
    fixtures: pd.DataFrame
    train_frame: pd.DataFrame
    predictions: pd.DataFrame
    metrics: dict
    model_path: str
    report_path: str


FEATURE_COLS = [
    "elo_diff",
    "form_points_diff",
    "form_goal_diff_diff",
    "recent_goals_for_diff",
    "recent_goals_against_diff",
    "is_neutral",
    "is_world_cup",
]


def _result_label(home_score: float, away_score: float) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _actual_score(home_score: float, away_score: float) -> tuple[float, float]:
    if home_score > away_score:
        return 1.0, 0.0
    if home_score < away_score:
        return 0.0, 1.0
    return 0.5, 0.5


def _team_snapshot(history_rows: list[dict], team: str, elo: dict[str, float]) -> dict[str, float]:
    recent = [r for r in reversed(history_rows) if r["team"] == team][:6]
    if not recent:
        return {
            "elo": elo.get(team, 1500.0),
            "form_points": 0.0,
            "form_goal_diff": 0.0,
            "recent_goals_for": 0.0,
            "recent_goals_against": 0.0,
        }
    points = sum(r["points"] for r in recent) / len(recent)
    gd = sum(r["goals_for"] - r["goals_against"] for r in recent) / len(recent)
    gf = sum(r["goals_for"] for r in recent) / len(recent)
    ga = sum(r["goals_against"] for r in recent) / len(recent)
    return {
        "elo": elo.get(team, 1500.0),
        "form_points": points,
        "form_goal_diff": gd,
        "recent_goals_for": gf,
        "recent_goals_against": ga,
    }


def build_training_frame(history: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], list[dict]]:
    elo: dict[str, float] = {}
    history_rows: list[dict] = []
    rows = []

    for _, match in history.sort_values("date").iterrows():
        home = str(match["home_team"])
        away = str(match["away_team"])
        home_snapshot = _team_snapshot(history_rows, home, elo)
        away_snapshot = _team_snapshot(history_rows, away, elo)
        label = _result_label(match["home_score"], match["away_score"])

        rows.append(
            {
                "date": match["date"],
                "home_team": home,
                "away_team": away,
                "result": label,
                "elo_diff": home_snapshot["elo"] - away_snapshot["elo"],
                "form_points_diff": home_snapshot["form_points"] - away_snapshot["form_points"],
                "form_goal_diff_diff": home_snapshot["form_goal_diff"] - away_snapshot["form_goal_diff"],
                "recent_goals_for_diff": home_snapshot["recent_goals_for"] - away_snapshot["recent_goals_for"],
                "recent_goals_against_diff": home_snapshot["recent_goals_against"] - away_snapshot["recent_goals_against"],
                "is_neutral": int(bool(match["neutral"])),
                "is_world_cup": int(match["tournament"] == "FIFA World Cup"),
            }
        )

        home_rating = elo.get(home, 1500.0)
        away_rating = elo.get(away, 1500.0)
        expected_home = _expected_score(home_rating, away_rating)
        actual_home, actual_away = _actual_score(match["home_score"], match["away_score"])
        k = 35.0 if match["tournament"] == "FIFA World Cup" else 25.0
        elo[home] = home_rating + k * (actual_home - expected_home)
        elo[away] = away_rating + k * (actual_away - (1.0 - expected_home))

        history_rows.append(
            {
                "team": home,
                "points": 3 if label == "H" else 1 if label == "D" else 0,
                "goals_for": float(match["home_score"]),
                "goals_against": float(match["away_score"]),
            }
        )
        history_rows.append(
            {
                "team": away,
                "points": 3 if label == "A" else 1 if label == "D" else 0,
                "goals_for": float(match["away_score"]),
                "goals_against": float(match["home_score"]),
            }
        )

    return pd.DataFrame(rows), elo, history_rows


def _fixture_features(fixtures: pd.DataFrame, elo: dict[str, float], history_rows: list[dict]) -> pd.DataFrame:
    rows = []
    for _, match in fixtures.iterrows():
        team1 = str(match["team1"])
        team2 = str(match["team2"])
        s1 = _team_snapshot(history_rows, team1, elo)
        s2 = _team_snapshot(history_rows, team2, elo)
        rows.append(
            {
                "match_no": match["match_no"],
                "date": match["date"],
                "time": match["time"],
                "round": match["round"],
                "group": match["group"],
                "team1": team1,
                "team2": team2,
                "ground": match["ground"],
                "elo_diff": s1["elo"] - s2["elo"],
                "form_points_diff": s1["form_points"] - s2["form_points"],
                "form_goal_diff_diff": s1["form_goal_diff"] - s2["form_goal_diff"],
                "recent_goals_for_diff": s1["recent_goals_for"] - s2["recent_goals_for"],
                "recent_goals_against_diff": s1["recent_goals_against"] - s2["recent_goals_against"],
                "is_neutral": 1,
                "is_world_cup": 1,
            }
        )
    return pd.DataFrame(rows)


def fetch_data(state: WorldCupState) -> WorldCupState:
    state["history"] = fetch_international_results()
    state["fixtures"] = fetch_worldcup_2026_fixtures()
    return state


def train_model(state: WorldCupState) -> WorldCupState:
    train_frame, elo, history_rows = build_training_frame(state["history"])
    train_frame = train_frame.dropna(subset=FEATURE_COLS + ["result"]).reset_index(drop=True)
    split_idx = int(len(train_frame) * 0.82)
    x_train = train_frame.loc[: split_idx - 1, FEATURE_COLS]
    y_train = train_frame.loc[: split_idx - 1, "result"]
    x_test = train_frame.loc[split_idx:, FEATURE_COLS]
    y_test = train_frame.loc[split_idx:, "result"]

    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=12, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    prob = model.predict_proba(x_test)

    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "worldcup_model.joblib"
    joblib.dump({"model": model, "elo": elo, "history_rows": history_rows}, model_path)

    state["train_frame"] = train_frame
    state["model_path"] = str(model_path)
    state["metrics"] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "log_loss": float(log_loss(y_test, prob, labels=model.classes_)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "history_rows": int(len(state["history"])),
        "fixture_rows": int(len(state["fixtures"])),
    }
    return state


def predict_worldcup(state: WorldCupState) -> WorldCupState:
    bundle = joblib.load(state["model_path"])
    model = bundle["model"]
    fixture_frame = _fixture_features(state["fixtures"], bundle["elo"], bundle["history_rows"])
    probs = model.predict_proba(fixture_frame[FEATURE_COLS])
    out = fixture_frame[["match_no", "date", "time", "round", "group", "team1", "team2", "ground"]].copy()
    for idx, cls in enumerate(model.classes_):
        out[f"prob_{cls}"] = np.round(probs[:, idx], 4)
    out["pick"] = model.predict(fixture_frame[FEATURE_COLS])
    state["predictions"] = out
    return state


def enrich_with_live_odds(state: WorldCupState) -> WorldCupState:
    enriched, odds_metrics = enrich_predictions_with_odds(state["predictions"], state["output_dir"])
    state["predictions"] = enriched
    state["metrics"] = {**state["metrics"], **odds_metrics}
    return state


def write_outputs(state: WorldCupState) -> WorldCupState:
    output_dir = Path(state["output_dir"])
    pred_path = output_dir / "worldcup_predictions.csv"
    report_path = output_dir / "worldcup_report.json"
    state["predictions"].to_csv(pred_path, index=False)
    pd.Series({**state["metrics"], "predictions_file": str(pred_path)}).to_json(
        report_path, force_ascii=False, indent=2
    )
    state["report_path"] = str(report_path)
    return state


def build_worldcup_graph():
    graph = StateGraph(WorldCupState)
    graph.add_node("fetch_data", fetch_data)
    graph.add_node("train_model", train_model)
    graph.add_node("predict_worldcup", predict_worldcup)
    graph.add_node("enrich_with_live_odds", enrich_with_live_odds)
    graph.add_node("write_outputs", write_outputs)
    graph.set_entry_point("fetch_data")
    graph.add_edge("fetch_data", "train_model")
    graph.add_edge("train_model", "predict_worldcup")
    graph.add_edge("predict_worldcup", "enrich_with_live_odds")
    graph.add_edge("enrich_with_live_odds", "write_outputs")
    graph.add_edge("write_outputs", END)
    return graph.compile()

