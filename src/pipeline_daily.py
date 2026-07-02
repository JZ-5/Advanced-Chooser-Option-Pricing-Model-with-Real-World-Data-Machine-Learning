import os
import time
import random
import requests
import urllib3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# 1. 路径自适应对齐
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
    
PROCESSED_FILE_PATH = f"{project_root}/data/processed/features_ann_final.csv"

def get_pipeline_time_window():
    default_start = "2018-01-01"
    if not os.path.exists(PROCESSED_FILE_PATH):
        return default_start, datetime.now().strftime("%Y-%m-%d"), None
        
    df_history = pd.read_csv(PROCESSED_FILE_PATH, parse_dates=['Date']).set_index('Date')
    if df_history.empty:
        return default_start, datetime.now().strftime("%Y-%m-%d"), None
        
    last_recorded_date = df_history.index.max()
    print(f"当前数据集最新记录日期为: {last_recorded_date.strftime('%Y-%m-%d')}")
    
    fetch_start_dt = last_recorded_date - timedelta(days=380)
    return fetch_start_dt.strftime("%Y-%m-%d"), datetime.now().strftime("%Y-%m-%d"), last_recorded_date

def fetch_jpm_from_alpha_vantage_paid(start_str, end_str):
    """商业级数据通道：提取 100% 准确的 JPM 后复权收盘价"""
    av_key = os.getenv("ALPHA_VANTAGE_KEY")
    if not av_key:
        raise ValueError("未在环境变量中检测到 ALPHA_VANTAGE_KEY，请检查配置！")
        
    print("正在通过 Alpha Vantage 付费级 TIME_SERIES_DAILY_ADJUSTED 接口增量提取 JPM...")
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=JPM&apikey={av_key}&outputsize=full"
    
    proxies = None if os.getenv('GITHUB_ACTIONS') else {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
    response = requests.get(url, proxies=proxies, timeout=20, verify=False).json()
    
    time_series = response.get("Time Series (Daily)", {})
    if not time_series:
        raise RuntimeError(f"Alpha Vantage 未返回 JPM 有效时序。API 响应: {list(response.keys())}")
        
    records = []
    for date_str, data in time_series.items():
        if start_str <= date_str <= end_str:
            records.append({'Date': pd.to_datetime(date_str), 'JPM_Close': float(data['5. adjusted close'])})
            
    if not records:
        raise RuntimeError("Alpha Vantage 未返回指定区间内的任何有效 JPM 数据。")
        
    return pd.DataFrame(records).set_index('Date').sort_index()

def fetch_from_fred(series_id, start_str, end_str, column_name):
    """通用 FRED 官方 API 提取器：100% 免疫任何机房风控与封锁"""
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        raise ValueError("未在环境变量中检测到 FRED_API_KEY！")
        
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={fred_key}&file_type=json"
    proxies = None if os.getenv('GITHUB_ACTIONS') else {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
    
    response = requests.get(url, proxies=proxies, timeout=15, verify=False).json()
    if 'error_message' in response:
        raise ValueError(f"FRED API 拒绝请求 ({series_id})，原因: {response['error_message']}")
        
    observations = response.get('observations', [])
    records = []
    for obs in observations:
        obs_date = obs['date']
        val = obs['value']
        if start_str <= obs_date <= end_str and val != '.':
            records.append({'Date': pd.to_datetime(obs_date), column_name: float(val)})
            
    if not records:
        raise ValueError(f"FRED 未返回 {series_id} 在指定时间区间内的有效观测。")
        
    return pd.DataFrame(records).set_index('Date').sort_index()

def fetch_raw_data_block(start_str, end_str):
    print(f"正在进行增量交易日抓取: 从 {start_str} 到 {end_str}")
    
    # 1. 商业通道提取 JPM
    jpm = fetch_jpm_from_alpha_vantage_paid(start_str, end_str)
    
    # 2. 官方通道提取大盘恐慌指数 VIX (Series ID: VIXCLS)
    print("开始从 FRED 官方接口提取大盘恐慌指数(VIX)行情...")
    vix = fetch_from_fred("VIXCLS", start_str, end_str, "VIX_Close")
    
    # 3. 官方通道提取国债无风险利率 (Series ID: DTB3)
    print("开始从 FRED 官方接口提取 3-Month Treasury Bill 基准利率...")
    rates = fetch_from_fred("DTB3", start_str, end_str, "Risk_Free_Rate")
    
    # 强行剥离潜在的时区标签，确保时间轴完美对齐
    jpm.index = pd.to_datetime(jpm.index).tz_localize(None).normalize()
    vix.index = pd.to_datetime(vix.index).tz_localize(None).normalize()
    rates.index = pd.to_datetime(rates.index).tz_localize(None).normalize()
    
    # 横向干净合并
    df_market = pd.merge(jpm, vix, left_index=True, right_index=True, how='outer')
    df_block = pd.merge(df_market, rates, left_index=True, right_index=True, how='outer')
    
    return df_block.interpolate(method='linear').ffill().bfill()

def compute_pipeline_features(df_block):
    print("正在计算连续量化特征矩阵...")
    q1 = df_block['Risk_Free_Rate'].quantile(0.25)
    q3 = df_block['Risk_Free_Rate'].quantile(0.75)
    iqr = q3 - q1
    df_block['Risk_Free_Rate'] = np.clip(df_block['Risk_Free_Rate'], q1 - 1.5*iqr, q3 + 1.5*iqr)
    
    df_block['Daily_Return'] = np.log(df_block['JPM_Close'] / df_block['JPM_Close'].shift(1))
    df_block['Rolling_Vol_20d'] = df_block['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    df_block['Dividend_Growth_Proxy'] = np.log(df_block['JPM_Close'] / df_block['JPM_Close'].shift(252)).rolling(window=20).mean()
    
    df_block['VIX_Decimal'] = df_block['VIX_Close'] / 100.0
    df_block['VIX_JPM_Corr_20d'] = df_block['Daily_Return'].rolling(window=20).corr(df_block['VIX_Decimal'].diff())
    df_block['IR_Momentum_10d'] = df_block['Risk_Free_Rate'].rolling(window=10).mean()
    
    df_block['JPM_SMA20_Disparity'] = (df_block['JPM_Close'] / df_block['JPM_Close'].rolling(window=20).mean()) - 1
    df_block['IV_RV_Spread'] = df_block['VIX_Decimal'] - df_block['Rolling_Vol_20d']
    df_block['Rate_Delta'] = df_block['Risk_Free_Rate'].diff()
    
    return df_block.dropna()[['JPM_Close', 'VIX_Decimal', 'Risk_Free_Rate', 'Daily_Return', 'Rolling_Vol_20d', 'Dividend_Growth_Proxy', 'VIX_JPM_Corr_20d', 'IR_Momentum_10d', 'JPM_SMA20_Disparity', 'IV_RV_Spread', 'Rate_Delta']].round(4)

def append_new_rows(df_new_features, last_recorded_date):
    if df_new_features.empty or last_recorded_date is None: return
    df_incremental = df_new_features[df_new_features.index > last_recorded_date]
    if df_incremental.empty:
        print("大表已是最新，本日无须追加新交易日。")
        return
    print(f"成功捕获到 {len(df_incremental)} 个新交易日的特征数据。准备追加...")
    df_history = pd.read_csv(PROCESSED_FILE_PATH, parse_dates=['Date']).set_index('Date')
    
    df_combined = pd.concat([df_history, df_incremental]).sort_index()
    df_combined.to_csv(PROCESSED_FILE_PATH)
    print(f"追加合并成功。特征矩阵最新终点已成功延伸至: {df_incremental.index.max().strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    try:
        start_str, end_str, last_date = get_pipeline_time_window()
        df_raw = fetch_raw_data_block(start_str, end_str)
        df_features = compute_pipeline_features(df_raw)
        append_new_rows(df_features, last_date)
        print("自动化增量清洗流水线执行完毕。")
    except Exception as pipeline_error:
        print(f"流水线本日运行异常: {pipeline_error}")