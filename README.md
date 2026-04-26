# CupOracle

一个足球预测 Agent：基于历史比赛、Elo 风格实力特征、近期状态，以及可选实时赔率，预测 2026 世界杯小组赛，并支持五大联赛近期回测。

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
- 五大联赛近期回测

## 五大联赛回测

世界杯开赛前，可以用正在进行的联赛测试模型稳定性，并在提供 The Odds API Key 后预测未来未开赛比赛。当前支持：

- 英超
- 西甲
- 德甲
- 意甲
- 法甲

命令行运行示例：

```bash
python3 run_league_agent.py --league E0 --holdout 80
```

把五大联赛合并成一个总池训练和回测：

```bash
python3 run_league_agent.py --league ALL --holdout 100
```

联赛代码：

- `ALL`：五大联赛合并
- `E0`：英超
- `SP1`：西甲
- `D1`：德甲
- `I1`：意甲
- `F1`：法甲

输出文件：

- `outputs/league_E0_report.json`
- `outputs/league_E0_backtest_predictions.csv`
- `outputs/league_ALL_report.json`
- `outputs/league_ALL_backtest_predictions.csv`

联赛回测会使用 football-data.co.uk 的公开赛果和 Bet365 赛前欧赔。逻辑是：用较早比赛训练，拿最近 N 场已结束比赛验证命中率。

为了避免模型只学会“低赔就赢”，联赛模块现在拆成两条路径：

- 实力路径：只使用 Elo、近 6 场积分、近 6 场净胜球、近期进失球，不使用欧赔训练
- 市场路径：只把欧赔换算成去水后的市场隐含概率

输出会展示：

- 实力方向
- 市场方向
- 融合方向
- 爆冷风险
- 价值方向
- 是否跟市场一致
- 推荐/观望
- 推荐类型（胜平负 / 不败 / 观望）
- 推荐命中率

过程表现特征包括：

- 近 6 场射门差
- 近 6 场射正差
- 近 6 场角球差
- 红黄牌压力差
- 犯规压力差
- 半场进球表现差
- 休息天数差
- 过程修正差：用射门、射正、角球估计的过程优势，和实际净胜球之间的偏差

这个修正用于处理“场面碾压但结果输了”的情况，避免模型只因为一场比分结果就过度看衰球队。

高置信推荐逻辑：

- 条件足够强时给出胜平负方向
- 如果胜平负不够强，但不败概率够高，则给出 `1X` 或 `X2`
- 风险较高或方向冲突时输出 `观望`
- 页面会单独统计推荐场次和推荐命中率

如果设置了 `THE_ODDS_API_KEY`，同一次运行还会通过 The Odds API 拉取未来比赛实时欧赔，并生成：

- `outputs/league_E0_upcoming_predictions.csv`

页面里的“五大联赛回测”标签页会同时展示：

- 最近已结束比赛回测
- 未来比赛预测

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

