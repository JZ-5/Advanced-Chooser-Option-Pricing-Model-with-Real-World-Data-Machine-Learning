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

# 动态反爬虫浏览器指纹池
USER_AGENTS = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2.1 Safari/605.1.15'
]

def get_clean_session():
    """建立完全独立的、无历史粘滞痕迹的全新网络会话，洗净被污染的Cookie"""
    session = requests.Session()
    if not os.getenv('GITHUB_ACTIONS'):
        PROXY_PORT = "7890"
        session.proxies.update({
            "http": f"http://127.0.0.1:{PROXY_PORT}",
            "https": f"http://127.0.0.1:{PROXY_PORT}"
        })
    
    session.headers.update({
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    })
    return session

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

def fetch_ticker_with_hard_retry(ticker_symbol, start_str, end_str, max_retries=4):
    """稳健数据抓取核心：抗击429频控拦截"""
    for attempt in range(max_retries):
        current_session = get_clean_session()
        try:
            ticker = yf.Ticker(ticker_symbol, session=current_session)
            hist = ticker.history(start=start_str, end=end_str)
            if not hist.empty:
                return hist
            raise RuntimeError("数据返回为空表，触发隐式防火墙拦截。")
        except Exception as e:
            wait_time = (attempt ** 2) * 10 if attempt > 0 else 5
            if attempt < max_retries - 1:
                print(f"提取 {ticker_symbol} 遭遇防火墙拦截。将销毁旧缓存，并在 {wait_time} 秒后切换新指纹执行第 {attempt + 2} 次重试... 原因: {e}")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"在连续重试 {max_retries} 次后仍无法穿透雅虎财经频控拦截。")

def fetch_raw_data_block(start_str, end_str):
    print(f"正在进行增量交易日抓取: 从 {start_str} 到 {end_str}")
    
    print("开始提取摩根大通(JPM)日线行情...")
    jpm_hist = fetch_ticker_with_hard_retry("JPM", start_str, end_str)
    jpm = jpm_hist[['Close']].rename(columns={'Close': 'JPM_Close'})
    
    print("开始提取大盘恐慌指数(^VIX)行情...")
    vix_hist = fetch_ticker_with_hard_retry("^VIX", start_str, end_str)
    vix = vix_hist[['Close']].rename(columns={'Close': 'VIX_Close'})
    
    if isinstance(jpm.columns, pd.MultiIndex): jpm.columns = ['JPM_Close']
    if isinstance(vix.columns, pd.MultiIndex): vix.columns = ['VIX_Close']
        
    df_market = pd.merge(jpm, vix, left_index=True, right_index=True, how='outer')
    
    # FRED 官方利率抓取
    fred_key = os.getenv("FRED_API_KEY")
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=DTB3&api_key={fred_key}&file_type=json"
    
    try:
        proxies = None if os.getenv('GITHUB_ACTIONS') else {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
        response = requests.get(url, proxies=proxies, timeout=15, verify=False).json()
        if 'error_message' in response: raise ValueError(response['error_message'])
            
        records = []
        for obs in response.get('observations', []):
            obs_date = obs['date']
            val = obs['value']
            if start_str <= obs_date <= end_str and val != '.':
                records.append({'Date': pd.to_datetime(obs_date), 'Risk_Free_Rate': float(val) / 100.0})
        df_rates = pd.DataFrame(records).set_index('Date')
    except Exception as e:
        print(f"FRED 利率获取失败，自动切换至 Yahoo Finance 备用利率防封锁更新链: {e}")
        irx_hist = fetch_ticker_with_hard_retry("^IRX", start_str, end_str)
        if isinstance(irx_hist.columns, pd.MultiIndex): irx_hist.columns = irx_hist.columns.droplevel(1)
        irx_hist['Risk_Free_Rate'] = irx_hist['Close'] / 100.0
        df_rates = irx_hist[['Risk_Free_Rate']]
        
    # --- 核心修正：双向强制时区剥离 (Timezone Stripping) 与时间归一化 ---
    df_market.index = pd.to_datetime(df_market.index).tz_localize(None).normalize()
    df_rates.index = pd.to_datetime(df_rates.index).tz_localize(None).normalize()
    
    # 此时两个 DataFrame 的索引均为纯净的 tz-naive 类型，横向外连接完美通行
    df_block = pd.merge(df_market, df_rates, left_index=True, right_index=True, how='outer')
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
    df_history = pd.read_csv(PROCESSED_FILE_PATH, parse_dates=['Date']).set_index('Date')
    pd.concat([df_history, df_incremental]).sort_index().to_csv(PROCESSED_FILE_PATH)
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