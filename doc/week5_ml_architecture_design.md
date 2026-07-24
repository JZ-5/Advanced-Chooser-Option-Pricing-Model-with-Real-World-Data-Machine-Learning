#  JPM 选择权期权定价：Week 5 机器学习双路径架构设计与评估文档

* **测试集区间**：2023-11 至 2024-12（共 265 个交易日，严格时序封存）
* **生产架构决议**：**确定 Approach 2 (GBDT Residual Compensation) 为最终生产部署 Champion Model**。

---

## 1. 双路径架构设计概览

在 Week 5 中，我们并行设计并验证了两种机器学习定价架构：

### 1.1 Approach 1：物理约束驱动的间接波动率预测架构 (Indirect Volatility Model)
* **核心机制**：保留 Rubinstein (1991) 选择权封闭解结构，利用机器学习模型基于宏观与情绪特征预测前瞻隐含波动率 $\hat{\sigma}_{\text{ML}}$，二次代入 BSM 公式定价。
* **计算公式**：
  $$P_{\text{App1}} = \text{BsmChooserPricer}\left(S_0, X, T_1, T_2, r, q, \hat{\sigma}_{\text{ML}}(\mathbf{X})\right)$$

### 1.2 Approach 2：端到端数据驱动残差补偿架构 (Direct Residual Compensation)
* **核心机制**：以 BSM 理论估值为基底，使用 GBDT / 神经网络直接预测场内真实成交价与理论价之间的残差溢价 $\hat{R}_{\text{ML}}$。
* **计算公式**：
  $$P_{\text{App2}} = P_{\text{BSM\_Baseline}} + \hat{R}_{\text{ML}}(\mathbf{X}_{\text{Pure\_Regime}})$$

---

## 2. 测试集 (Test Set) 多尺度量化评估

在 70% Train / 15% Val / 15% Test 的严格时序切分下，各模型在未见测试集上的表现如下：

| 决策尺度 ($T_1$) | BSM Baseline MAE | Approach 1 (ML Vol) MAE | **Approach 2 (GBDT) MAE** | Approach 2 (ANN) MAE | **Champion 提升幅度** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.25 年** | $10.95 | $6.97 | **$2.24** | $11.11 | **+ 79.51%** |
| **0.50 年** | $7.84 | $5.12 | **$2.24** | $9.16 | **+ 71.46%** |
| **0.75 年** | $8.62 | $4.41 | **$2.94** | $7.61 | **+ 65.83%** |

---

## 3. 核心洞察与经验总结

1. **剔除绝对价格特征是关键**：当特征集中包含 BSM 绝对价格时，模型容易在 2024 年高股价区发生过拟合。剔除绝对价格、仅保留 `IV_RV_Spread` 和 `Rate_Delta` 等纯 Regime 指标后，GBDT 展现出了极强的泛化能力。

---

## 4. Week 6 部署与生产优化规划

1. **超参数精细化搜索 (Optuna)**：针对 Champion Model (GBDT) 开展贝叶斯超参数调优，进一步优化 `learning_rate` 与 `l2_regularization`。
2. **生产推理 Pipeline 封装**：将完整数据清洗、特征构建、BSM 基准计算与 GBDT 残差预测集成至 `pipeline_daily.py`，实现日频一键自动化出价。