import os
import requests
import urllib3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

# 关闭 HTTPS 证书警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# 1. 路径自适应对齐
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
    
PROCESSED_FILE_PATH = f"{project_root}/data/processed/features_ann_final.csv"

if os.getenv('GITHUB_ACTIONS'):
    PROXIES = None
    print("检测到 GitHub Actions 环境，已自动关闭代理。")
else:
    PROXY_PORT = "7890"
    PROXIES = {
        "http": f"http://127.0.0.1:{PROXY_PORT}",
        "https": f"http://127.0.0.1:{PROXY_PORT}"
    }
    print(f"本地开发环境，已启用代理端口: {PROXY_PORT}")

def get_pipeline_time_window():
    default_start = "2018-01-01"
    
    if not os.path.exists(PROCESSED_FILE_PATH):
        print(f"未检测到历史特征文件，将从初始起点 {default_start} 开始全量构建。")
        return default_start, datetime.now().strftime("%Y-%m-%d"), None
        
    df_history = pd.read_csv(PROCESSED_FILE_PATH, parse_dates=['Date']).set_index('Date')
    if df_history.empty:
        return default_start, datetime.now().strftime("%Y-%m-%d"), None
        
    last_recorded_date = df_history.index.max()
    print(f"当前数据集最新记录日期为: {last_recorded_date.strftime('%Y-%m-%d')}")
    
    # 向前回溯 380 个自然日，确保包含 252 个完整的交易工作日以供特征滚动计算
    fetch_start_dt = last_recorded_date - timedelta(days=380)
    
    fetch_start_str = fetch_start_dt.strftime("%Y-%m-%d")
    fetch_end_str = datetime.now().strftime("%Y-%m-%d")
    
    return fetch_start_str, fetch_end_str, last_recorded_date

def fetch_raw_data_block(start_str, end_str):
    print(f"正在进行增量交易日抓取: 从 {start_str} 到 {end_str}")
    
    session = requests.Session()
    if PROXIES:
        session.proxies.update(PROXIES)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    # yfinance 默认只下载周一至周五的交易日数据，排除了周末
    jpm = yf.download("JPM", start=start_str, end=end_str, session=session, progress=False)[['Close']].rename(columns={'Close': 'JPM_Close'})
    vix = yf.download("^VIX", start=start_str, end=end_str, session=session, progress=False)[['Close']].rename(columns={'Close': 'VIX_Close'})
    
    if isinstance(jpm.columns, pd.MultiIndex):
        jpm.columns = jpm.columns.droplevel(1)
        jpm.columns = ['JPM_Close']
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.droplevel(1)
        vix.columns = ['VIX_Close']
        
    df_market = pd.merge(jpm, vix, left_index=True, right_index=True, how='outer')
    
    # FRED 利率抓取（工作日发布）
    fred_key = os.getenv("FRED_API_KEY")
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=DTB3&api_key={fred_key}&file_type=json"
    
    try:
        response = requests.get(url, proxies=PROXIES, timeout=120, verify=False).json()
        records = []
        for obs in response.get('observations', []):
            obs_date = obs['date']
            val = obs['value']
            if start_str <= obs_date <= end_str and val != '.':
                records.append({'Date': pd.to_datetime(obs_date), 'Risk_Free_Rate': float(val) / 100.0})
        df_rates = pd.DataFrame(records).set_index('Date')
    except Exception as e:
        print(f"FRED 获取失败，启动 Yahoo Finance 备用交易日利率抓取: {e}")
        irx = yf.download("^IRX", start=start_str, end=end_str, session=session, progress=False)[['Close']]
        if isinstance(irx.columns, pd.MultiIndex):
            irx.columns = irx.columns.droplevel(1)
        irx['Risk_Free_Rate'] = irx['Close'] / 100.0
        df_rates = irx[['Risk_Free_Rate']]
        
    # --- 数据清洗步骤一：时间轴对齐 (Time Alignment) ---
    df_market.index = pd.to_datetime(df_market.index).normalize()
    df_rates.index = pd.to_datetime(df_rates.index).normalize()
    df_block = pd.merge(df_market, df_rates, left_index=True, right_index=True, how='outer')
    
    # --- 数据清洗步骤二：缺失值插值 (Interpolation) ---
    # 利用线性插值与前后填充，平滑洗净因为股债市场假期错配导致的极少数 NaN 数据坏点
    df_block = df_block.interpolate(method='linear').ffill().bfill()
    
    return df_block

def compute_pipeline_features(df_block):
    print("正在计算连续量化特征矩阵...")
    
    # --- 数据清洗步骤三：异常值清洗 (IQR Clamping) ---
    # 同步历史稳定边界，对新注入的无风险利率进行 IQR 盖帽限制，防止外部 API 偶发性脏数据冲击
    q1 = df_block['Risk_Free_Rate'].quantile(0.25)
    q3 = df_block['Risk_Free_Rate'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df_block['Risk_Free_Rate'] = np.clip(df_block['Risk_Free_Rate'], lower_bound, upper_bound)
    
    # 特征计算
    df_block['Daily_Return'] = np.log(df_block['JPM_Close'] / df_block['JPM_Close'].shift(1))
    df_block['Rolling_Vol_20d'] = df_block['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    df_block['Dividend_Growth_Proxy'] = np.log(df_block['JPM_Close'] / df_block['JPM_Close'].shift(252)).rolling(window=20).mean()
    
    df_block['VIX_Decimal'] = df_block['VIX_Close'] / 100.0
    vix_delta = df_block['VIX_Decimal'].diff()
    df_block['VIX_JPM_Corr_20d'] = df_block['Daily_Return'].rolling(window=20).corr(vix_delta)
    df_block['IR_Momentum_10d'] = df_block['Risk_Free_Rate'].rolling(window=10).mean()
    
    df_block['JPM_SMA20_Disparity'] = (df_block['JPM_Close'] / df_block['JPM_Close'].rolling(window=20).mean()) - 1
    df_block['IV_RV_Spread'] = df_block['VIX_Decimal'] - df_block['Rolling_Vol_20d']
    df_block['Rate_Delta'] = df_block['Risk_Free_Rate'].diff()
    
    # 剔除滚动初始阶段必然产生的冷启动 NaN 行
    df_block_clean = df_block.dropna().copy()
    
    feature_columns = [
        'JPM_Close', 'VIX_Decimal', 'Risk_Free_Rate',
        'Daily_Return', 'Rolling_Vol_20d', 'Dividend_Growth_Proxy',
        'VIX_JPM_Corr_20d', 'IR_Momentum_10d',
        'JPM_SMA20_Disparity', 'IV_RV_Spread', 'Rate_Delta'
    ]
    return df_block_clean[feature_columns].round(4)

def append_new_rows(df_new_features, last_recorded_date):
    if df_new_features.empty:
        print("未生成有效特征。")
        return
        
    if last_recorded_date is None:
        print("首次全量初始化保存。")
        df_new_features.to_csv(PROCESSED_FILE_PATH)
        return
        
    # 过滤出真正大于历史最后记录日期的全新增量交易日
    df_incremental = df_new_features[df_new_features.index > last_recorded_date]
    
    if df_incremental.empty:
        print("大表已是最新，本日无须追加新交易日。")
        return
        
    print(f"成功捕获到 {len(df_incremental)} 个新交易日的特征数据。准备追加...")
    
    df_history = pd.read_csv(PROCESSED_FILE_PATH, parse_dates=['Date']).set_index('Date')
    
    # 拼接增量数据并按时间重排
    df_combined = pd.concat([df_history, df_incremental]).sort_index()
    df_combined.to_csv(PROCESSED_FILE_PATH)
    print(f"追加合并成功。特征矩阵最新终点已延伸至: {df_combined.index.max().strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    try:
        start_str, end_str, last_date = get_pipeline_time_window()
        df_raw = fetch_raw_data_block(start_str, end_str)
        df_features = compute_pipeline_features(df_raw)
        append_new_rows(df_features, last_date)
        print("自动化增量清洗流水线执行完毕。")
    except Exception as pipeline_error:
        print(f"流水线本日运行异常: {pipeline_error}")