import os
import sys
import time
import joblib
import requests
import urllib3
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# 屏蔽 HTTPS 警告并加载环境变量
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

# 自动定位项目根目录
project_root = Path.cwd().parent.parent if "src" in str(Path.cwd()) else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.bsm_chooser import BsmChooserPricer

class RealtimeOptionPipeline:
    """
    工业级日频数据管线：
    1. 动态抓取 JPM 股票、CBOE VIX、FRED 无风险利率的完整日频历史数据；
    2. 严格按日频序列纯动态计算 12 维 Market Regime 特征，拒绝任何数字硬编码；
    3. 包含网络请求自动重试机制 (Retry Loop)，若抓取彻底失败直接抛出异常，绝不伪造随机数据。
    """
    def __init__(self, symbol="JPM", max_retries=3, retry_delay=2, use_proxy=False, proxy_port=7890):
        self.symbol = symbol
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.av_key = os.getenv("ALPHA_VANTAGE_KEY")
        self.model_path = project_root / "models" / "gbdt_residual_champion.pkl"
    
        if not self.model_path.exists():
            raise FileNotFoundError(f"❌ 未找到 Champion 模型权重文件: {self.model_path}")
        
        self.models = joblib.load(self.model_path)
        self.pricer = BsmChooserPricer()
    
    # 动态控制代理：默认不开启；仅当 use_proxy=True 时才走代理
        if use_proxy:
            self.proxies = {"http": f"http://127.0.0.1:{proxy_port}", "https": f"http://127.0.0.1:{proxy_port}"}
        else:
            self.proxies = None
    
    def _fetch_with_retry(self, url, description):
        """通用网络抓取重试函数：带浏览器伪装 Header"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"🔄 [{description}] 正在抓取数据 (第 {attempt}/{self.max_retries} 次尝试)...")
                response = requests.get(url, headers=headers, proxies=self.proxies, timeout=30, verify=False)
                
                # 校验是否误抓到了 HTML 报错网页
                if response.status_code == 200:
                    if "<html" in response.text.lower() or "<!doctype" in response.text.lower():
                        print(f"⚠️ [{description}] 接口返回了 HTML 页面而非数据，可能触发了反爬虫阻断。")
                    else:
                        return response
                else:
                    print(f"⚠️ [{description}] HTTP 状态码异常: {response.status_code}")
            except Exception as e:
                print(f"⚠️ [{description}] 请求网络异常: {e}")
            
            if attempt < self.max_retries:
                sleep_time = self.retry_delay * (2 ** (attempt - 1))
                print(f"⏳ {sleep_time} 秒后重新抓取...")
                time.sleep(sleep_time)

        raise RuntimeError(f"❌ 致命错误: [{description}] 在尝试 {self.max_retries} 次后抓取失败！已终止运行。")

    def fetch_jpm_daily(self):
        """抓取 JPM 日频调后收盘价与历史派息数据 (拉取 full 模式以确保覆盖 252 交易日)"""
        if not self.av_key:
            raise ValueError("❌ 未配置 ALPHA_VANTAGE_KEY 环境变量！")

        # 🔑 修正 1：将 outputsize 改为 full，确保能拿到至少 252 交易日以上的完整历史分红
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={self.symbol}&outputsize=full&apikey={self.av_key}"
        res = self._fetch_with_retry(url, "JPM 股票日频行情").json()
        
        ts_data = res.get("Time Series (Daily)", {})
        if not ts_data:
            raise ValueError("❌ Alpha Vantage API 未返回有效的 JPM 时间序列数据。")

        df = pd.DataFrame.from_dict(ts_data, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        df['close'] = df['5. adjusted close'].astype(float)
        df['dividend'] = df['7. dividend amount'].astype(float)
        
        # 保留最近 400 个交易日（足够计算 252 窗口）
        return df[['close', 'dividend']].tail(400)

    def fetch_vix_daily(self):
        """抓取 CBOE VIX 大盘波动率指数日频数据（优先使用 FRED API，回退至 CSV）"""
        fred_key = os.getenv("FRED_API_KEY")
        
        # 优先选择官方 API 接口，响应更快更稳定
        if fred_key:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id=VIXCLS&api_key={fred_key}&file_type=json"
            try:
                res = self._fetch_with_retry(url, "CBOE VIX 每日大盘指数 (API)").json()
                obs = res.get('observations', [])
                records = [{'DATE': pd.to_datetime(o['date']), 'VIXCLS': float(o['value'])} 
                           for o in obs if o['value'] != '.']
                df = pd.DataFrame(records).set_index('DATE').sort_index()
                return df['VIXCLS'] / 100.0
            except Exception as e:
                print(f"⚠️ API 提取 VIX 异常 ({e})，尝试回退至 CSV 导出通道...")

        # 保底方案：使用网页 CSV 导出 (将 timeout 设为 30s)
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
        res = self._fetch_with_retry(url, "CBOE VIX 每日大盘指数 (CSV)")
        from io import StringIO
        df = pd.read_csv(StringIO(res.text))
        df.columns = [str(c).upper().strip() for c in df.columns]
        df['DATE'] = pd.to_datetime(df['DATE'])
        vix_col = [c for c in df.columns if 'VIX' in c][0]
        df[vix_col] = pd.to_numeric(df[vix_col], errors='coerce')
        df = df.dropna().set_index('DATE').sort_index()
        return df[vix_col] / 100.0

    def fetch_risk_free_rate_daily(self):
        """抓取美国 3 个月国债基准利率 (DTB3) 日频数据（优先使用 FRED API，回退至 CSV）"""
        fred_key = os.getenv("FRED_API_KEY")
        
        # 优先选择官方 API 接口
        if fred_key:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id=DTB3&api_key={fred_key}&file_type=json"
            try:
                res = self._fetch_with_retry(url, "3个月国债无风险利率 DTB3 (API)").json()
                obs = res.get('observations', [])
                records = [{'DATE': pd.to_datetime(o['date']), 'DTB3': float(o['value'])} 
                           for o in obs if o['value'] != '.']
                df = pd.DataFrame(records).set_index('DATE').sort_index()
                return df['DTB3'] / 100.0
            except Exception as e:
                print(f"⚠️ API 提取 DTB3 异常 ({e})，尝试回退至 CSV 导出通道...")

        # 保底方案：使用网页 CSV 导出
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
        res = self._fetch_with_retry(url, "3个月国债无风险利率 DTB3 (CSV)")
        from io import StringIO
        df = pd.read_csv(StringIO(res.text))
        df.columns = [str(c).upper().strip() for c in df.columns]
        df['DATE'] = pd.to_datetime(df['DATE'])
        rate_col = [c for c in df.columns if 'DTB3' in c][0]
        df[rate_col] = pd.to_numeric(df[rate_col], errors='coerce')
        df = df.dropna().set_index('DATE').sort_index()
        return df[rate_col] / 100.0

    def build_dynamic_12d_features(self):
        """
        按时间轴合并三个日频数据源，纯动态计算所有 12 维特征（与训练集 pipeline_daily.py 100% 对齐）
        """
        print("\n📥 [Pipeline] 开始对齐多源日频序列并计算 12 维 Market Regime 动态特征...")
        
        # 1. 分别抓取三个源的完整日频历史
        df_jpm = self.fetch_jpm_daily()
        s_vix = self.fetch_vix_daily()
        s_rf = self.fetch_risk_free_rate_daily()

        # 2. 按日期求交集对齐 (Inner Join)
        df = df_jpm.join(s_vix, how='inner').join(s_rf, how='inner').sort_index()
        
        if len(df) < 260:
            raise ValueError(f"❌ 对齐后的有效日频序列过短 ({len(df)} 天)，无法支撑 252D 滚动窗口特征计算！")

        # 3. 纯动态计算 12 维连续特征列（严格对齐离线训练代码）
        df['JPM_Close'] = df['close']
        df['VIX_Decimal'] = df['VIXCLS']
        df['Risk_Free_Rate'] = df['DTB3']
        
        # 3.1 对数日收益率
        df['Daily_Return'] = np.log(df['JPM_Close'] / df['JPM_Close'].shift(1))
        
        # 3.2 20D 实现波动率 (年化)
        df['Rolling_Vol_20d'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
        
        # 3.3 股息率与一年期股价动能代理 ( 修正 2：修复公式对齐)
        df['Real_Dividend_Yield'] = df['dividend'].rolling(window=252, min_periods=1).sum() / df['JPM_Close']
        df['Dividend_Growth_Proxy'] = np.log(df['JPM_Close'] / df['JPM_Close'].shift(252)).rolling(window=20).mean()
        
        # 3.4 20D VIX 变动量与 JPM 收益率的相关性 ( 修正 3：使用 .diff() 与离线一致)
        df['VIX_JPM_Corr_20d'] = df['Daily_Return'].rolling(window=20).corr(df['VIX_Decimal'].diff())
        
        # 3.5 10D 利率动能与 Rate_Delta ( 修正 4：利率本身的 10D 均线)
        df['IR_Momentum_10d'] = df['Risk_Free_Rate'].rolling(window=10).mean()
        df['Rate_Delta'] = df['Risk_Free_Rate'].diff()
        
        # 3.6 SMA20 偏离度
        sma20 = df['JPM_Close'].rolling(window=20).mean()
        df['JPM_SMA20_Disparity'] = (df['JPM_Close'] / sma20) - 1.0
        
        # 3.7 波动率风险溢价 (IV - RV Spread)
        df['IV_RV_Spread'] = df['VIX_Decimal'] - df['Rolling_Vol_20d']

        # 4. 提取最新一个交易日的 12 维特征向量
        feature_cols = [
            'JPM_Close', 'VIX_Decimal', 'Risk_Free_Rate', 'Daily_Return',
            'Rolling_Vol_20d', 'Dividend_Growth_Proxy', 'Real_Dividend_Yield',
            'VIX_JPM_Corr_20d', 'IR_Momentum_10d', 'JPM_SMA20_Disparity',
            'IV_RV_Spread', 'Rate_Delta'
        ]
        
        latest_features = df[feature_cols].dropna().tail(1)
        if latest_features.empty:
            raise ValueError("❌ 滚动特征计算后包含 NaN，无法提取最新的 12 维特征向量！")
            
        return latest_features

    def predict_realtime_chooser(self, strike=150.0, t2=1.0):
        """一键日频推理主入口"""
        df_latest_feat = self.build_dynamic_12d_features()
        
        latest_date = df_latest_feat.index[0].strftime('%Y-%m-%d')
        s0 = float(df_latest_feat['JPM_Close'].iloc[0])
        r = float(df_latest_feat['Risk_Free_Rate'].iloc[0])
        q = float(df_latest_feat['Real_Dividend_Yield'].iloc[0])
        vol = float(df_latest_feat['Rolling_Vol_20d'].iloc[0])
        vix = float(df_latest_feat['VIX_Decimal'].iloc[0])
        corr = float(df_latest_feat['VIX_JPM_Corr_20d'].iloc[0])
        
        print("\n" + "="*80)
        print(f" 【日频真实行情推理结果】 {self.symbol} Chooser Option (最新交易日: {latest_date})")
        print("="*80)
        print(f" 标的股价 S₀ : ${s0:.2f} | 20D 实现波动率 σ: {vol*100:.2f}% | 滚动股息率 q: {q*100:.2f}%")
        print(f" CBOE VIX 指数 : {vix*100:.2f}% | 3M 国债利率 r: {r*100:.2f}% | 20D VIX-JPM 相关性: {corr:.4f}")
        print("-" * 80)
        
        results = []
        for t1 in [0.25, 0.50, 0.75]:
            bsm_price = self.pricer.price_chooser(s0, strike, t1, t2, r, q, vol)
            res_pred = float(self.models[t1].predict(df_latest_feat)[0])
            final_price = bsm_price + res_pred
            
            print(f" 决策尺度 T₁ = {t1:<4} 年:")
            print(f"   ├─ BSM 理论底座报价   : ${bsm_price:.2f}")
            print(f"   ├─ GBDT 加性残差补偿 : ${res_pred:+.2f}")
            print(f"   └─ 最终公允市场报价   : ${final_price:.2f}\n")
            
            results.append({
                "T1": t1,
                "BSM_Price": bsm_price,
                "Residual": res_pred,
                "Final_Price": final_price
            })
            
        print("="*80)
        return pd.DataFrame(results), df_latest_feat

if __name__ == "__main__":
    try:
        pipeline = RealtimeOptionPipeline(symbol="JPM", max_retries=3, retry_delay=2)
        results_df, features_df = pipeline.predict_realtime_chooser()
    except Exception as e:
        print(f"\n 流水线运行失败: {e}")