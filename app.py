import os
import sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# 1. 自动定位项目根目录并挂载 sys.path
project_root = Path.cwd().parent if "notebooks" in os.getcwd() else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 尝试导入 BSM Chooser Pricer 与 实时数据管线
try:
    from src.models.bsm_chooser import BsmChooserPricer
except ImportError:
    from scipy.stats import norm
    class BsmChooserPricer:
        @staticmethod
        def price_standard_european(s0, x, t, r, q, sigma, option_type='call'):
            t = np.maximum(t, 1e-5)
            sigma = np.maximum(sigma, 1e-5)
            d1 = (np.log(s0 / x) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
            d2 = d1 - sigma * np.sqrt(t)
            if option_type.lower() == 'call':
                return s0 * np.exp(-q * t) * norm.cdf(d1) - x * np.exp(-r * t) * norm.cdf(d2)
            elif option_type.lower() == 'put':
                return x * np.exp(-r * t) * norm.cdf(-d2) - s0 * np.exp(-q * t) * norm.cdf(-d1)
            else:
                raise ValueError("仅支持 'call' 或 'put' 类型。")

        def price_chooser(self, s0, strike, t1, t2, r, q, sigma):
            strike_prime = strike * np.exp(-(r - q) * (t2 - t1))
            call_component = self.price_standard_european(s0, strike_prime, t1, r, q, sigma, option_type='call')
            put_component = self.price_standard_european(s0, strike, t2, r, q, sigma, option_type='put')
            return np.round(np.exp(-q * (t2 - t1)) * call_component + put_component, 4)

try:
    from src.pipeline_app import RealtimeOptionPipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

# 2. Streamlit 页面全局配置
st.set_page_config(
    page_title="Chooser Option Pricing System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. 缓存机制加载 Champion 模型包
@st.cache_resource
def load_champion_models():
    model_path = project_root / "models" / "gbdt_residual_champion.pkl"
    if not model_path.exists():
        st.error(f" 未找到 Champion 模型权重文件: {model_path.resolve()}")
        st.stop()
    return joblib.load(model_path)

models = load_champion_models()
pricer = BsmChooserPricer()

# 4. 侧边栏：市场参数控制台与日频数据对接
st.sidebar.title(" 市场参数控制台")

# 一键刷入最新真实日频数据
if PIPELINE_AVAILABLE:
    if st.sidebar.button(" 抓取 Alpha Vantage / FRED 最新日频数据"):
        with st.spinner("正在连接网络抓取 JPM、VIX 及 FRED 真实日频行情..."):
            try:
                pipeline = RealtimeOptionPipeline(symbol="JPM", max_retries=3)
                _, df_realtime = pipeline.predict_realtime_chooser()
                st.session_state['realtime_feat'] = df_realtime
                st.sidebar.success(" 真实日频行情抓取与 12 维特征计算成功！")
            except Exception as e:
                st.sidebar.error(f" 抓取失败 (已拒绝伪造数据): {e}")
else:
    st.sidebar.warning(" 未检测到 `src.pipelines.pipeline_realtime` 模块，已开启手动输入模式。")

# 提取管线刷入的实时特征（若未抓取，使用基准取值）
feat = st.session_state.get('realtime_feat', None)

default_s0 = float(feat['JPM_Close'].iloc[0]) if feat is not None else 356.20
default_vol = float(feat['Rolling_Vol_20d'].iloc[0]) if feat is not None else 0.2000
default_vix = float(feat['VIX_Decimal'].iloc[0]) if feat is not None else 0.1800
default_r = float(feat['Risk_Free_Rate'].iloc[0]) if feat is not None else 0.0381
default_q = float(feat['Real_Dividend_Yield'].iloc[0]) if feat is not None else 0.0168
default_ret = float(feat['Daily_Return'].iloc[0]) if feat is not None else 0.0010
default_sma_disp = float(feat['JPM_SMA20_Disparity'].iloc[0]) if feat is not None else 0.0150
default_corr = float(feat['VIX_JPM_Corr_20d'].iloc[0]) if feat is not None else -0.3500
default_r_delta = float(feat['Rate_Delta'].iloc[0]) if feat is not None else 0.0005
default_r_mom = float(feat['IR_Momentum_10d'].iloc[0]) if feat is not None else 0.0001

st.sidebar.subheader(" 标的与合约参数")
s0 = st.sidebar.number_input("标的现价 S₀ ($)", value=default_s0, step=1.0, format="%.2f")
strike = st.sidebar.number_input("行权价 K ($)", value=150.0, step=1.0, format="%.2f")
t2 = st.sidebar.slider("总存续期 T₂ (年)", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

st.sidebar.subheader(" 宏观与微观 Market Regimes")
vol_20d = st.sidebar.slider("20D 历史实现波动率 σ_RV", min_value=0.05, max_value=0.80, value=default_vol, step=0.01)
vix_decimal = st.sidebar.slider("CBOE VIX 隐含波动率", min_value=0.05, max_value=0.80, value=default_vix, step=0.01)
r_rate = st.sidebar.number_input("无风险利率 r", value=default_r, step=0.001, format="%.4f")
q_yield = st.sidebar.number_input("滚动股息率 q", value=default_q, step=0.001, format="%.4f")

st.sidebar.subheader(" 动能与技术衍生指标")
daily_return = st.sidebar.number_input("日对数收益率 Daily Return", value=default_ret, format="%.4f")
sma20_disp = st.sidebar.number_input("SMA20 偏离度", value=default_sma_disp, format="%.4f")
vix_jpm_corr = st.sidebar.slider("VIX-JPM 20D 相关性", min_value=-1.0, max_value=1.0, value=default_corr, step=0.05)
rate_delta = st.sidebar.number_input("利率变动 Rate Delta", value=default_r_delta, format="%.4f")
ir_momentum = st.sidebar.number_input("10D 利率动能", value=default_r_mom, format="%.4f")

# 5. 主界面渲染
st.title(" 选择权期权 (Chooser Option) 混合定价系统")
st.caption("物理模型 (BSM Rubinstein 1991) + 数据驱动加性残差补偿 (GBDT)")

st.markdown("---")

# 组装 12 维连续特征向量
features_dict = {
    'JPM_Close': s0,
    'VIX_Decimal': vix_decimal,
    'Risk_Free_Rate': r_rate,
    'Daily_Return': daily_return,
    'Rolling_Vol_20d': vol_20d,
    'Dividend_Growth_Proxy': q_yield,
    'Real_Dividend_Yield': q_yield,
    'VIX_JPM_Corr_20d': vix_jpm_corr,
    'IR_Momentum_10d': ir_momentum,
    'JPM_SMA20_Disparity': sma20_disp,
    'IV_RV_Spread': vix_decimal - vol_20d,
    'Rate_Delta': rate_delta
}
df_features = pd.DataFrame([features_dict])

# 6. 三大决策尺度实时估值卡片
st.subheader(" 多决策尺度  估值面板")

cols = st.columns(3)
t1_scales = [0.25, 0.50, 0.75]

for idx, t1 in enumerate(t1_scales):
    bsm_val = pricer.price_chooser(s0, strike, t1, t2, r_rate, q_yield, vol_20d)
    res_val = float(models[t1].predict(df_features)[0])
    final_val = bsm_val + res_val
    
    with cols[idx]:
        st.metric(
            label=f"决策尺度 T₁ = {t1} 年",
            value=f"${final_val:.2f}",
            delta=f"GBDT 补正 ${res_val:+.2f}",
            delta_color="normal"
        )
        st.caption(f"BSM 报价: **${bsm_val:.2f}**")

st.markdown("---")

# 7. 交互式敏感性分析图表 (标的价格 S₀ 动态扫描)
st.subheader(" 标的价格敏感性分析 (S₀ Price Curve)")

s0_range = np.linspace(s0 * 0.7, s0 * 1.3, 30)
curve_data = []

for s_val in s0_range:
    feat_temp = features_dict.copy()
    feat_temp['JPM_Close'] = s_val
    df_temp = pd.DataFrame([feat_temp])
    
    row = {"S0": s_val}
    for t1 in t1_scales:
        b_val = pricer.price_chooser(s_val, strike, t1, t2, r_rate, q_yield, vol_20d)
        r_val = float(models[t1].predict(df_temp)[0])
        row[f"T1={t1} 年 GBDT 混合报价"] = b_val + r_val
        row[f"T1={t1} 年 BSM 报价"] = b_val
    curve_data.append(row)

df_curve = pd.DataFrame(curve_data).set_index("S0")

st.line_chart(df_curve[[f"T1={t1} 年 GBDT 混合报价" for t1 in t1_scales]])

# 8. 展开查看完整的 12 维特征向量
with st.expander(" 查看当前驱动模型的 12 维 Market Regime 特征向量 (Feature Vector)"):
    st.dataframe(df_features.T.rename(columns={0: "特征当前取值"}))