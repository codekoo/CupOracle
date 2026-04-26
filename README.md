# CupOracle

一个世界杯预测 Agent：基于历史国家队比赛、Elo 风格实力特征、近期状态，以及可选实时赔率，预测 2026 世界杯小组赛胜平负概率。

CupOracle 会自动联网抓取公开数据，训练模型，并输出预测结果。

> A World Cup prediction agent powered by historical international results, Elo-style form features, and optional live odds.

## 快速开始

1. 安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

2. 运行世界杯预测：

```bash
python3 run_worldcup_agent.py
```

如果你有 The Odds API Key，可以这样接入实时欧赔：

```bash
export THE_ODDS_API_KEY="你的API_KEY"
python3 run_worldcup_agent.py
```

3. 打开查看页面：

```bash
python3 -m streamlit run app.py
```

## 数据来源

- 历史国家队比赛结果：`martj42/international_results`
- 2026 世界杯公开赛程：`openfootball/worldcup.json`

首次运行会下载数据并缓存到 `outputs/cache`，后续运行会优先使用本地缓存。

## 输出文件

- `outputs/worldcup_model.joblib`：世界杯预测模型
- `outputs/worldcup_predictions.csv`：世界杯小组赛胜平负概率
- `outputs/worldcup_report.json`：训练评估报告
- `outputs/worldcup_odds.csv`：实时欧赔原始匹配数据（提供 API Key 后生成）

## 查看页面

页面提供：

- 模型评估指标
- 72 场小组赛胜平负概率表
- 按小组筛选
- 按球队搜索
- 一键重新运行预测
- 输入 The Odds API Key 后拉取实时欧赔
- 下载完整预测 CSV

## 实时赔率

当前支持 The Odds API：

- 环境变量：`THE_ODDS_API_KEY`
- 默认 sport key：`soccer_fifa_world_cup`
- 默认市场：`h2h`（胜平负）

接入后会在预测表中增加：

- 队伍1欧赔 / 平局欧赔 / 队伍2欧赔
- 市场队伍1胜率 / 市场平局概率 / 市场队伍2胜率
- 融合队伍1胜率 / 融合平局概率 / 融合队伍2胜率
- 融合方向

说明：如果 API 暂时没有世界杯赔率，项目会正常输出基础模型预测，赔率匹配数量会显示为 0。

## 当前特征

- 简化 Elo 差
- 近 6 场积分状态差
- 近 6 场净胜球差
- 近 6 场进球/失球差
- 是否中立场
- 是否世界杯比赛

