from __future__ import annotations

import json
import os
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

THE_ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
DEFAULT_SPORT_KEY = "soccer_fifa_world_cup"

TEAM_ALIASES = {
    "usa": "united states",
    "u.s.a.": "united states",
    "united states": "united states",
    "south korea": "korea republic",
    "korea republic": "korea republic",
    "ivory coast": "cote d ivoire",
    "côte d'ivoire": "cote d ivoire",
    "cote d'ivoire": "cote d ivoire",
    "dr congo": "congo dr",
    "democratic republic of congo": "congo dr",
    "curaçao": "curacao",
}


def _normalize_team(name: str) -> str:
    value = str(name).strip().lower()
    for old, new in {
        "&": " and ",
        ".": " ",
        "'": " ",
        "-": " ",
        "  ": " ",
    }.items():
        value = value.replace(old, new)
    value = " ".join(value.split())
    return TEAM_ALIASES.get(value, value)


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize_team(a), _normalize_team(b)).ratio()


def _fetch_json(url: str, timeout: int = 45) -> list[dict]:
    req = Request(url, headers={"User-Agent": "cuporacle-worldcup-agent/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_the_odds_api(
    api_key: str | None = None,
    sport_key: str | None = None,
    regions: str = "eu,uk,us",
    markets: str = "h2h",
    odds_format: str = "decimal",
) -> pd.DataFrame:
    api_key = api_key or os.getenv("THE_ODDS_API_KEY", "")
    if not api_key:
        return pd.DataFrame()

    sport_key = sport_key or os.getenv("THE_ODDS_API_SPORT_KEY", DEFAULT_SPORT_KEY)
    query = urlencode(
        {
            "apiKey": api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
        }
    )
    payload = _fetch_json(f"{THE_ODDS_API_URL.format(sport_key=sport_key)}?{query}")
    rows = []
    for event in payload:
        outcome_prices: dict[str, list[float]] = {}
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    name = str(outcome.get("name", "")).strip()
                    price = outcome.get("price")
                    if name and price:
                        outcome_prices.setdefault(name, []).append(float(price))
        avg_prices = {name: sum(values) / len(values) for name, values in outcome_prices.items() if values}
        if avg_prices:
            rows.append(
                {
                    "odds_event_id": event.get("id"),
                    "commence_time": event.get("commence_time"),
                    "home_team": event.get("home_team"),
                    "away_team": event.get("away_team"),
                    "outcome_prices": avg_prices,
                    "bookmaker_count": len(event.get("bookmakers", [])),
                }
            )
    return pd.DataFrame(rows)


def _find_outcome_price(prices: dict, target_team: str) -> float | None:
    best_name = None
    best_score = 0.0
    for name in prices:
        if _normalize_team(name) == "draw":
            continue
        score = _similarity(name, target_team)
        if score > best_score:
            best_name = name
            best_score = score
    if best_name and best_score >= 0.72:
        return float(prices[best_name])
    return None


def _find_draw_price(prices: dict) -> float | None:
    for name, price in prices.items():
        if _normalize_team(name) == "draw":
            return float(price)
    return None


def _match_event(odds_events: pd.DataFrame, team1: str, team2: str) -> pd.Series | None:
    if odds_events.empty:
        return None
    best_idx = None
    best_score = 0.0
    for idx, event in odds_events.iterrows():
        a = event.get("home_team", "")
        b = event.get("away_team", "")
        score_direct = (_similarity(team1, a) + _similarity(team2, b)) / 2
        score_reverse = (_similarity(team1, b) + _similarity(team2, a)) / 2
        score = max(score_direct, score_reverse)
        if score > best_score:
            best_idx = idx
            best_score = score
    if best_idx is not None and best_score >= 0.78:
        return odds_events.loc[best_idx]
    return None


def _market_probs(odds_team1: float, odds_draw: float, odds_team2: float) -> tuple[float, float, float]:
    raw = [1 / odds_team1, 1 / odds_draw, 1 / odds_team2]
    total = sum(raw)
    return raw[0] / total, raw[1] / total, raw[2] / total


def enrich_predictions_with_odds(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    blend_market_weight: float = 0.35,
) -> tuple[pd.DataFrame, dict]:
    odds_events = fetch_the_odds_api()
    output_dir = Path(output_dir)
    odds_path = output_dir / "worldcup_odds.csv"

    if odds_events.empty:
        return predictions, {"odds_rows": 0, "odds_matched_rows": 0, "odds_file": ""}

    odds_events.to_csv(odds_path, index=False)
    rows = []
    matched = 0
    for _, row in predictions.iterrows():
        out = row.to_dict()
        event = _match_event(odds_events, row["team1"], row["team2"])
        if event is not None:
            prices = event["outcome_prices"]
            odds_team1 = _find_outcome_price(prices, row["team1"])
            odds_team2 = _find_outcome_price(prices, row["team2"])
            odds_draw = _find_draw_price(prices)
            if odds_team1 and odds_draw and odds_team2:
                matched += 1
                m_h, m_d, m_a = _market_probs(odds_team1, odds_draw, odds_team2)
                out.update(
                    {
                        "odds_team1": round(odds_team1, 3),
                        "odds_draw": round(odds_draw, 3),
                        "odds_team2": round(odds_team2, 3),
                        "market_prob_H": round(m_h, 4),
                        "market_prob_D": round(m_d, 4),
                        "market_prob_A": round(m_a, 4),
                    }
                )
                model_probs = [row["prob_H"], row["prob_D"], row["prob_A"]]
                market_probs = [m_h, m_d, m_a]
                blended = [
                    (1 - blend_market_weight) * model_probs[i] + blend_market_weight * market_probs[i]
                    for i in range(3)
                ]
                labels = ["H", "D", "A"]
                out["final_prob_H"] = round(blended[0], 4)
                out["final_prob_D"] = round(blended[1], 4)
                out["final_prob_A"] = round(blended[2], 4)
                out["final_pick"] = labels[int(max(range(3), key=lambda i: blended[i]))]
        rows.append(out)

    enriched = pd.DataFrame(rows)
    return enriched, {
        "odds_rows": int(len(odds_events)),
        "odds_matched_rows": int(matched),
        "odds_file": str(odds_path),
    }

