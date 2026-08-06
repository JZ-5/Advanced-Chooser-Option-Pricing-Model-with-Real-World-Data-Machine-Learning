import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# 1. 路径自适应与工程模块导入
project_root = Path.cwd().parent if "notebooks" in os.getcwd() else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 尝试载入工程模块 BsmChooserPricer，若路径不匹配则使用内嵌降级类
try:
    from src.models.bsm_chooser import BsmChooserPricer
    print(" 成功从 src.models.bsm_chooser 导入 BsmChooserPricer")
except ImportError:
    from scipy.stats import norm
    
    class BsmChooserPricer:
        """基于 Rubinstein (1991) 封闭解的欧式选择权期权向量化定价器"""
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
            if np.any(t1 >= t2):
                raise ValueError("决策日 T1 必须严格早于最终到期日 T2。")
            strike_prime = strike * np.exp(-(r - q) * (t2 - t1))
            call_component = self.price_standard_european(s0, strike_prime, t1, r, q, sigma, option_type='call')
            put_component = self.price_standard_european(s0, strike, t2, r, q, sigma, option_type='put')
            chooser_value = np.exp(-q * (t2 - t1)) * call_component + put_component
            return float(np.round(chooser_value, 4))

# ==========================================
# 生产环境日频推理流水线类 (DailyPipeline)
# ==========================================
class ChooserOptionDailyPipeline:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = project_root / "models" / "gbdt_residual_champion.pkl"
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f" 找不到模型权重包: {model_path}")
            
        self.models = joblib.load(model_path)
        self.pricer = BsmChooserPricer()
        
        # Champion GBDT 模型要求的 12 维标准特征列表
        self.feature_cols = [
            'JPM_Close', 'VIX_Decimal', 'Risk_Free_Rate', 'Daily_Return',
            'Rolling_Vol_20d', 'Dividend_Growth_Proxy', 'Real_Dividend_Yield',
            'VIX_JPM_Corr_20d', 'IR_Momentum_10d', 'JPM_SMA20_Disparity',
            'IV_RV_Spread', 'Rate_Delta'
        ]
        self.t1_scales = [0.25, 0.5, 0.75]
        self.strike = 150.0  # Chooser 合约固定行权价
        self.t2 = 1.0        # 总存续期 1 年
        print(f" 生产推理流水线初始化成功！已挂载模型: {Path(model_path).name}")

    def predict_single_day(self, daily_row):
        if isinstance(daily_row, dict):
            daily_row = pd.Series(daily_row)
            
        daily_row = daily_row.copy()
        
        # 1. 提取原始无风险利率并进行 [百分比 -> 小数] 安全修复
        raw_r = float(daily_row['Risk_Free_Rate'])
        # 若 raw_r > 1.0 (例如 3.81)，说明是百分比格式，必须除以 100 变为 0.0381
        r = raw_r / 100.0 if raw_r > 1.0 else raw_r
        
        # 2. 更新修正后的 Rate 供 BSM 使用
        S = float(daily_row['JPM_Close'])
        q = float(daily_row['Real_Dividend_Yield'])
        sigma = float(daily_row['Rolling_Vol_20d'])
        date_str = str(daily_row.get('Date', '最新交易日'))[:10]
        
        # 3. 构造 12 维模型输入特征 X (注意：模型训练时若也用的是 0.0381，需保持一致)
        X_single = daily_row[self.feature_cols].to_frame().T
        # 确保传入模型特征中的 Risk_Free_Rate 也是小数形式
        X_single['Risk_Free_Rate'] = r
        
        results = {}
        print("\n" + "="*22 + f"  日频期权公允估值报告 [{date_str}] " + "="*22)
        print(f"标的价格 (S): ${S:.2f} | 20d历史波动率 (sigma): {sigma*100:.2f}% | 修正后无风险利率 (r): {r*100:.2f}% | 滚动股息率 (q): {q*100:.2f}%")
        print("-" * 75)
        
        for t1 in self.t1_scales:
            # 使用正确的小数利率 r (如 0.0381) 进行 BSM 计算
            bsm_price = self.pricer.price_chooser(s0=S, strike=self.strike, t1=t1, t2=self.t2, r=r, q=q, sigma=sigma)
            
            model = self.models[t1]
            pred_residual = float(model.predict(X_single)[0])
            final_price = bsm_price + pred_residual
            
            results[t1] = {
                'bsm_price': bsm_price,
                'pred_residual': pred_residual,
                'final_price': final_price
            }
            
            print(f" 决策尺度 T1 = {t1:<4} 年 -> BSM基准: ${bsm_price:>6.2f} | 残差补偿: ${pred_residual:>+6.2f} | 最终报价: ${final_price:>6.2f}")
            
        print("=" * 75)
        return results

# ==========================================
# 自动化执行入口：加载最新交易日特征并出价
# ==========================================
if __name__ == "__main__":
    file_path = project_root / "data" / "processed" / "features_ann_final.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f" 找不到特征数据文件: {file_path}")
        
    df = pd.read_csv(file_path, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    
    # 提取数据库中最新的单日样本
    latest_sample = df.iloc[-1]
    
    # 运行线上推理流水线
    pipeline = ChooserOptionDailyPipeline()
    valuation_output = pipeline.predict_single_day(latest_sample)