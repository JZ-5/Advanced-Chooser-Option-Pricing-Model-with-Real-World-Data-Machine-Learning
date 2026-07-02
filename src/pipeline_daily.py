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

USER_AGENTS = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
]

def get_authenticated_yahoo_session():
    """高级伪装网关：先模拟人类访问雅虎主页获取合规的 Cookie 凭证，突破云端 429 限制"""
    session = requests.Session()
    if not os.getenv('GITHUB_ACTIONS'):
        PROXY_PORT = "7890"
        session.proxies.update({
            "http": f"http://127.0.0.1:{PROXY_PORT}",
            "https": f"http://127.0.0.1:{PROXY_PORT}"
        })
    
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    session.headers.update(headers)
    
    try:
        # 悄悄访问一次主页，借用雅虎服务器颁发的合法 Cookie 挂件
        session.get("https://finance.yahoo.com", timeout=10, verify=False)
    except Exception:
        pass # 静默失败，允许继续尝试下载
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

def fetch_ticker_from_yahoo_safe(ticker_symbol, start_str, end_str, max_retries=4):
    """带指数退避和动态 Cookie 洗净机制的雅虎 K 线数据安全提取器"""
    for attempt in range(max_retries):
        # 每次重试都重新生成带有新鲜 Cookie 的 Session，彻底洗净 429 历史粘滞
        current_session = get_authenticated_yahoo_session()
        try:
            ticker = yf.Ticker(ticker_symbol, session=current_session)
            # 采用默认的 history 将自动保持历史一致的后复权价格体系（Adjusted Prices）
            hist = ticker.history(start=start_str, end=end_str)
            if not hist.empty:
                return hist
            raise RuntimeError("数据流返回空表。")
        except Exception as e:
            # 云端环境下采取深度冷却策略 (45秒 -> 90秒 -> 180秒)
            wait_time = (attempt ** 2) * 45 if attempt > 0 else 20
            if attempt < max_retries - 1:
                print(f"提取 {ticker_symbol} 被云端拒接。已重置 Session，将在 {wait_time} 秒后重试... 原因: {e}")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"已耗尽所有云端网络穿透策略，仍无法下载 {ticker_symbol}。")

def fetch_raw_data_block(start_str, end_str):
    print(f"正在进行增量交易日抓取: 从 {start_str} 到 {end_str}")
    
    print("开始提取摩根大通(JPM)后复权日线行情...")
    jpm_hist = fetch_ticker_from_yahoo_safe("JPM", start_str, end_str)
    jpm = jpm_hist[['Close']].rename(columns={'Close': 'JPM_Close'})
    
    print("开始提取大盘恐慌指数(^VIX)行情...")
    vix_hist = fetch_ticker_from_yahoo_safe("^VIX", start_str, end_str)
    vix = vix_hist[['Close']].rename(columns={'Close': 'VIX_Close'})
    
    if isinstance(jpm.columns, pd.MultiIndex): jpm.columns = ['JPM_Close']
    if isinstance(vix.columns, pd.MultiIndex): vix.columns = ['VIX_Close']
    
    # --- 核心修正一：强制去时区标签，彻底干掉 Join 冲突 ---
    jpm.index = pd.to_datetime(jpm.index).tz_localize(None).normalize()
    vix.index = pd.to_datetime(vix.index).tz_localize(None).normalize()
    df_market = pd.merge(jpm, vix, left_index=True, right_index=True, how='outer')
    
    # FRED 利率抓取
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
        print(f"FRED 利率获取失败，自动切入 Yahoo Finance 备用利率防封锁更新链: {e}")
        irx_hist = fetch_ticker_from_yahoo_safe("^IRX", start_str, end_str)
        if isinstance(irx_hist.columns, pd.MultiIndex): irx_hist.columns = irx_hist.columns.droplevel(1)
        irx_hist['Risk_Free_Rate'] = irx_hist['Close'] / 100.0
        df_rates = irx_hist[['Risk_Free_Rate']]
        
    df_rates.index = pd.to_datetime(df_rates.index).tz_localize(None).normalize()
    
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