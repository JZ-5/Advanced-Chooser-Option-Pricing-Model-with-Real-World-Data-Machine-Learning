# Advanced-Chooser-Option-Pricing-Model-with-Real-World-Data-Machine-Learning




本项目的初始原始数据集时间跨度为 2018-01-01 至 2024-12-31。

### 1. 原始数据明细表 

| 字段名 (Column) | 数据类型 (Type) | 含义说明 (Description) | 数据来源 (Source) |
| :--- | :--- | :--- | :--- |
| `Date` | `DateTime` (主键) | 交易日期，格式为 YYYY-MM-DD | 所有源 |
| `JPM_Close` | `Float` | 摩根大通（JPM）股票当天的收盘价 | Yahoo Finance |
| `VIX_Close` | `Float` | 市场恐慌指数（VIX）当天的收盘价 | Yahoo Finance |
| `Risk_Free_Rate` | `Float` | 3个月期美国国债收益率（无风险利率 ） | FRED API or Yahoo Finance|
| `Sentiment_Score`| `Float` | 针对 JPM 的每日新闻情绪得分（范围 0 到 1） | Alpha Vantage API |




### 2. 特征工程明细 
整个量化流水线（`src/pipeline_daily.py`）采用纯 API 凭证驱动架构，每日自动化抓取多源数据并衍生特征。最终生成的特征宽表（`features_ann_final.csv`）包含 1 个时间戳索引列（Date）和 11 个量化特征：

| 特征变量名 (Feature Name) | 分类 / 数据源 (Source) | 数学定义 / 公式 (Formula) | 金融学直觉与模型作用 (Financial Intuition) |
| :--- | :--- | :--- | :--- |
| **`JPM_Close`** | Alpha Vantage (Paid API) | 原始数据 (`5. adjusted close`) | 标的资产后复权收盘价。消除红利与拆股污染，决定期权当前的实值/虚值状态（Moneyness）。 |
| **`VIX_Decimal`** | 美联储 FRED API (`VIXCLS`) | $VIX\_Close / 100$ | 广义隐含波动率基准。捕捉市场整体对极端尾部风险的系统性前瞻定价。 |
| **`Risk_Free_Rate`** | 美联储 FRED API (`DTB3`) | 原始数据 (3M国债年化收益率) | 瞬时无风险利率 $r$ 的代理变量。作为贴现率与几何布朗运动（GBM）漂移项的基础基准。 |
| **`Daily_Return`** | 衍生特征 | $\ln(JPM\_Close_t / JPM\_Close_{t-1})$ | 消除价格绝对量纲。转化为平稳时间序列，完美对齐期权模型中的资产对数收益率随机过程。 |
| **`Rolling_Vol_20d`** | 衍生特征 | $\sqrt{252} \times \sigma(\{Daily\_Return\}_{t-19}^t)$ | 20日年化历史实现波动率（RV）。选择权期权多头天然做多波动率（Long Vega），该特征直接驱动期权时间价值。 |
| **`Dividend_Growth_Proxy`** | 衍生特征 | $\frac{1}{20} \sum_{k=0}^{19} \ln\left(\frac{JPM\_Close_{t-k}}{JPM\_Close_{t-k-252}}\right)$ | 过去252交易日长期对数收益率的20日滚动均值。用作连续红利发放率 $q$ 的代理变量，并为 ANN 提供长周期趋势动能信号。 |
| **`VIX_JPM_Corr_20d`** | 衍生特征 | $\rho\left(\{Daily\_Return\}_{t-19}^t, \{\Delta VIX\_Decimal\}_{t-19}^t\right)$ | 标的收益率与大盘恐慌变动的滚动相关系数。量化非对称“杠杆效应”，决定大跌时看跌期权溢价的增长敏感度。 |
| **`IR_Momentum_10d`** | 衍生特征 | $\frac{1}{10}\sum_{i=0}^9 Risk\_Free\_Rate_{t-i}$ | 10日滚动利率均值。过滤短期资金利率的微观噪声，提炼宏观货币政策变动的核心趋势。 |
| **`JPM_SMA20_Disparity`** | 衍生特征 | $(JPM\_Close_t / SMA20_t) - 1$ | 20日简单移动平均线（SMA）偏离度。捕捉标的资产短期超买/超卖技术面下的均值回归（Mean-reversion）压力。 |
| **`IV_RV_Spread`** | 衍生特征 | $VIX\_Decimal_t - Rolling\_Vol\_20d_t$ | 波动率风险溢价（VRP）。量化市场情绪保险费，引导神经网络精准修正 Black-Scholes 解析解对期权费的系统性偏差。 |
| **`Rate_Delta`** | 衍生特征 | $Risk\_Free\_Rate_t - Risk\_Free\_Rate_{t-1}$ | 无风险利率的一阶差分。为模型提供利率期限结构变动的边际信号。 |
