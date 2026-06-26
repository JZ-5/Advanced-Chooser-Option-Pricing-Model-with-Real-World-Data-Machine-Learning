# Advanced-Chooser-Option-Pricing-Model-with-Real-World-Data-Machine-Learning



此表格为数据规范表格
本项目 Week 1 的初始原始数据集将包含以下多维核心字段，时间跨度为 2018-01-01 至 2024-12-31。

| 字段名 (Column) | 数据类型 (Type) | 含义说明 (Description) | 数据来源 (Source) |
| :--- | :--- | :--- | :--- |
| `Date` | `DateTime` (主键) | 交易日期，格式为 YYYY-MM-DD | 所有源 |
| `JPM_Close` | `Float` | 摩根大通（JPM）股票当天的收盘价 | Yahoo Finance |
| `VIX_Close` | `Float` | 市场恐慌指数（VIX）当天的收盘价 | Yahoo Finance |
| `Risk_Free_Rate` | `Float` | 3个月期美国国债收益率（无风险利率 ） | FRED API or Yahoo Finance|
| `Sentiment_Score`| `Float` | 针对 JPM 的每日新闻情绪得分（范围 0 到 1） | Alpha Vantage API |