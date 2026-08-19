# Advanced-Chooser-Option-Pricing-Model-with-Real-World-Data-Machine-Learning

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Production-v1.0.0-brightgreen.svg)]()

> **Quant Research & Trading Department | 8-Week Capstone Project Release**  
> 基于 **Black-Scholes-Merton (BSM Rubinstein 1991) 物理模型底座** 与 **梯度提升树 (GBDT) 加性残差补偿结构** 的欧式选择权期权 (Chooser Option) 智能混合定价系统。

---

## 执行摘要

传统 BSM 模型在定价奇异衍生品（如 Chooser Option）时存在恒定波动率与无摩擦交易等假设偏误。本项目构建了基于 **JPMorgan Chase (JPM)** 标的的混合机器学习定价管线：
1. **物理与数据驱动融合**：利用 Rubinstein (1991) 解析解建立基础价格底座，采用 `HistGradientBoostingRegressor` 学习基于 12 维特征的美元残差补偿（Additive Residuals）。
2. **时序交叉验证 (TS-CV) 防过拟合**：采用 3 折 `TimeSeriesSplit` 滚动窗口调优，将测试集平均 MAE 压低至 $2.14 ~ $2.77 美元，比传统 BSM 模型（MAE 约 $9.13 美元）精度提升 67.7% ~ 80.5%。
3. **真实期权链与压力测试对齐**：通过 Alpha Vantage 真实期权链挂单组合对冲验证，并在极端波动率（10%~50%）与利率变动（1.81%~6.81%）下通过了压力测试。

---

## 系统架构图 

```text
 ┌─────────────────────────────────────────────────────────────┐
 │                外部真实行情数据源                             │
 │   Alpha Vantage API (JPM 股票/期权) + FRED API (VIX, DTB3)   │
 └──────────────────────────────┬──────────────────────────────┘
                                │ 动态重试机制 (Retry Loop)
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │       日频数据与特征工程管线 (src/pipelines/pipeline_app.py)  │
 │   - 多源序列对齐                                             │
 │   - 计算 12 维连续特征向量                                    │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                 混合定价                                     │
 │   1. BSM Rubinstein (1991) 解析解物理底座                    │
 │   2. Champion GBDT 序列化模型包 (gbdt_residual_champion.pkl) │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │             前端交互式 UI (app.py / Streamlit)               │
 │   - T₁ = 0.25, 0.50, 0.75 年多决策尺度双重对比报价            │
 │   - 做市误差边际 (Error Margins) 与做市参考区间显示            │
 │   - 量化仪表盘 (敏感性/存续期趋势/模型性能对比)                 │
 └─────────────────────────────────────────────────────────────┘
```

---

## 项目文件目录结构

```text
chooser-option-pricing/
├── app.py                         # Streamlit 交互式 UI 主程序
├── requirements.txt               # 生产环境依赖包列表
├── README.md                      # 项目部署与架构说明文档
├── .github/
│   ├── workflows/
│   │   └── data_pipeline.yml      # gitaction流水线文件
├── config/
│   └── parameters.json            # BSM 参数
├── models/
│   └── gbdt_residual_champion.pkl # 调优后的模型权重
├── src/
│   ├── models/
│   │   └── bsm_chooser.py         # Rubinstein Chooser Option 解析解向量化定价器
│   ├── pipeline_app.py            # 交互式 UI 主程序的日频真实数据抓取与 12 维特征工程自动化管线（不储存数据）
│   ├── pipeline_daily.py          # 真实数据抓取与 12 维特征工程自动化管线（储存数据）
│   └── pipeline_prediction.py     # 最新交易日推理流水线
├── notebooks/                     # 各周任务测试代码
│   ├── week3_bsm_replication.ipynb  
│   ├── week4_CME_fetch_and_performance_evaluation.ipynb  
│   ├── week5_feature_preparation_and_ML.ipynb  
│   ├── week6_optuna.ipynb  
│   └── week7_stress_testing.ipynb  
├── doc/                           
│   ├── performance_benchmark_documentation.md          # BSM 基准模型性能说明文档
│   └── week5_ml_architecture_design.md                 # Week 5 机器学习架构设计与评估文档
└── data/                          # 结构化数据存储目录
```

---

## 快速启动指南

1. 环境准备与依赖安装
建议使用 Python 3.10+ 虚拟环境：

```bash
# 克隆仓库
git clone [https://github.com/your-username/chooser-option-pricing.git](https://github.com/your-username/chooser-option-pricing.git)
cd chooser-option-pricing

# 创建并激活虚拟环境 (以 venv 为例)
python -m venv venv

# Windows 激活:
.\venv\Scripts\activate
# Linux/macOS 激活:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

2. 配置环境变量
如果需要连接 Alpha Vantage 实时数据源，请在根目录下创建 `.env` 文件：

```ini
ALPHA_VANTAGE_KEY=your_alpha_vantage_api_key
```

3. 一键启动 Streamlit 交互式定价应用

```bash
streamlit run app.py
```

---

## 风控与免责声明

* **应用场景限制**：本系统模型权重与特征参数针对 JPM 训练优化，迁移至低流动性小盘股或商品衍生品前需进行重新校准。
* **交易免责声明**：本项目的定价输出仅供参考，不构成任何具体的投资或交易建议。
* **数据周期说明**：本项目的初始原始数据集时间跨度为 2018-01-01 至 2024-12-31。

---

### 原始数据明细表

| 字段名 (Column) | 数据类型 (Type) | 含义说明 (Description) | 数据来源 (Source) |
| :--- | :--- | :--- | :--- |
| `Date` | `DateTime` (主键) | 交易日期，格式为 YYYY-MM-DD | 所有源 |
| `JPM_Close` | `Float` | 摩根大通（JPM）股票当天的收盘价 | Alpha Vantage API / Yahoo Finance |
| `VIX_Close` | `Float` | 市场恐慌指数（VIX）当天的收盘价 | FRED API |
| `Risk_Free_Rate` | `Float` | 3 个月期美国国债收益率（无风险利率） | FRED API |

---

### 特征工程明细

整个量化流水线（`src/pipeline_app.py`）采用纯 API 凭证驱动架构，每日自动化抓取多源数据并衍生特征。最终生成的特征宽表（`features_ann_final.csv`）包含 1 个时间戳索引列（`Date`）和 12 个量化特征：

| 特征变量名 (Feature Name) | 分类 / 数据源 (Source) | 数学定义 / 公式 (Formula) | 金融学与模型作用 (Financial Intuition) |
| :--- | :--- | :--- | :--- |
| **`JPM_Close`** | Alpha Vantage / Yahoo Finance | 原始数据 (`5. adjusted close`) | 标的资产后复权收盘价。消除红利与拆股污染，决定期权当前的实值/虚值状态。 |
| **`VIX_Decimal`** | 美联储 FRED API (`VIXCLS`) | $VIX\_Close / 100$ | 广义隐含波动率基准。捕捉市场整体对极端尾部风险的系统性前瞻定价。 |
| **`Risk_Free_Rate`** | 美联储 FRED API (`DTB3`) | 原始数据 (3M国债年化收益率) | 瞬时无风险利率 $r$ 的代理变量。作为贴现率与几何布朗运动（GBM）漂移项的基础基准。 |
| **`Daily_Return`** | 衍生特征 | $\ln(JPM\_Close_t / JPM\_Close_{t-1})$ | 消除价格绝对量纲。转化为平稳时间序列，对齐期权模型中的资产对数收益率随机过程。 |
| **`Rolling_Vol_20d`** | 衍生特征 | $\sqrt{252} \times \sigma(\{Daily\_Return\}_{t-19}^t)$ | 20 日年化历史实现波动率（RV）。选择权期权多头天然做多波动率（Long Vega），该特征直接驱动期权时间价值。 |
| **`Dividend_Growth_Proxy`** | 衍生特征 | $\frac{1}{20} \sum_{k=0}^{19} \ln\left(\frac{JPM\_Close_{t-k}}{JPM\_Close_{t-k-252}}\right)$ | 过去 252 交易日长期对数收益率的 20 日滚动均值。用作连续红利发放率 $q$ 的代理变量，并提供长周期趋势动能信号。 |
| **`Real_Dividend_Yield`** | Alpha Vantage (`7. dividend amount`) / 衍生特征 | $\frac{\sum_{k \in \text{past 12m}} Dividend_k}{JPM\_Close_t}$ | 365 日滚动真实历史派息率。精准度量标的连续红利发放率 $q$，直接决定 Rubinstein 模型中调整后行权价 $X'$ 的贴现折扣与持有成本（Cost-of-Carry）。 |
| **`VIX_JPM_Corr_20d`** | 衍生特征 | $\rho\left(\{Daily\_Return\}_{t-19}^t, \{\Delta VIX\_Decimal\}_{t-19}^t\right)$ | 标的收益率与大盘恐慌变动的滚动相关系数。量化非对称“杠杆效应”，决定大跌时看跌期权溢价的增长敏感度。 |
| **`IR_Momentum_10d`** | 衍生特征 | $\frac{1}{10}\sum_{i=0}^9 Risk\_Free\_Rate_{t-i}$ | 10 日滚动利率均值。过滤短期资金利率的微观噪声，提炼宏观货币政策变动的核心趋势。 |
| **`JPM_SMA20_Disparity`** | 衍生特征 | $(JPM\_Close_t / SMA20_t) - 1$ | 20 日简单移动平均线（SMA）偏离度。捕捉标的资产短期超买/超卖技术面下的均值回归压力。 |
| **`IV_RV_Spread`** | 衍生特征 | $VIX\_Decimal_t - Rolling\_Vol\_20d_t$ | 波动率风险溢价（VRP）。量化市场情绪保险费，引导机器学习模型精准修正 Black-Scholes 解析解对期权费的系统性偏差。 |
| **`Rate_Delta`** | 衍生特征 | $Risk\_Free\_Rate_t - Risk\_Free\_Rate_{t-1}$ | 无风险利率的一阶差分。为模型提供利率期限结构变动的边际信号。 |