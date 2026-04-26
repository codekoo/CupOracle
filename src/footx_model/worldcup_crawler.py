from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

HISTORY_RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
HISTORY_RESULTS_FALLBACK_URL = "https://cdn.jsdelivr.net/gh/martj42/international_results@master/results.csv"
WORLDCUP_2026_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
WORLDCUP_2026_FALLBACK_URL = "https://cdn.jsdelivr.net/gh/openfootball/worldcup.json@master/2026/worldcup.json"
CACHE_DIR = Path("outputs/cache")


def _fetch_text(url: str, timeout: int = 90) -> str:
    req = Request(url, headers={"User-Agent": "cuporacle-worldcup-agent/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _fetch_with_cache(cache_name: str, urls: list[str], timeout: int = 90) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_text(encoding="utf-8")

    errors = []
    for url in urls:
        try:
            text = _fetch_text(url, timeout=timeout)
            cache_path.write_text(text, encoding="utf-8")
            return text
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("所有公开数据源下载失败: " + " | ".join(errors))


def fetch_international_results(url: str = HISTORY_RESULTS_URL) -> pd.DataFrame:
    text = _fetch_with_cache(
        "international_results.csv",
        [url, HISTORY_RESULTS_FALLBACK_URL],
        timeout=180,
    )
    df = pd.read_csv(StringIO(text))
    expected = {"date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"历史结果缺少字段: {sorted(missing)}")

    df = df[list(expected)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"] = df["neutral"].astype(bool)
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    return df.sort_values("date").reset_index(drop=True)


def fetch_worldcup_2026_fixtures(url: str = WORLDCUP_2026_URL) -> pd.DataFrame:
    text = _fetch_with_cache(
        "worldcup_2026.json",
        [url, WORLDCUP_2026_FALLBACK_URL],
        timeout=60,
    )
    payload = json.loads(text)
    rows = []
    for idx, item in enumerate(payload.get("matches", []), start=1):
        team1 = str(item.get("team1", "")).strip()
        team2 = str(item.get("team2", "")).strip()
        # 只预测已知球队的小组赛，淘汰赛占位符等实际确定后再跑。
        if any(ch.isdigit() for ch in team1 + team2) or team1.startswith(("W", "L")) or team2.startswith(("W", "L")):
            continue
        rows.append(
            {
                "match_no": item.get("num", idx),
                "date": item.get("date"),
                "time": item.get("time"),
                "round": item.get("round"),
                "group": item.get("group", ""),
                "team1": team1,
                "team2": team2,
                "ground": item.get("ground", ""),
                "neutral": True,
            }
        )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date", "team1", "team2"]).sort_values("date").reset_index(drop=True)

