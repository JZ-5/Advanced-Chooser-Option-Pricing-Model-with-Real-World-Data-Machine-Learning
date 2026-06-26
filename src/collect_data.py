import os
import time
import pandas as pd
import yfinance as yf
import requests
import urllib3
from datetime import datetime
from dotenv import load_dotenv

# 关闭本地代理抓取时产生的 HTTPS 证书警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 初始化并加载环境变量
load_dotenv()
FRED_KEY = os.getenv("FRED_API_KEY")
AV_KEY = os.getenv("ALPHA_VANTAGE_KEY")

START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

# 配置代理信息
PROXY_PORT = "7890"
PROXIES = {
    "http": f"http://127.0.0.1:{PROXY_PORT}",
    "https": f"http://127.0.0.1:{PROXY_PORT}"
}

def get_browser_session():
    """创建一个带有浏览器伪装的 Session，防止被 Yahoo 限流"""
    session = requests.Session()
    session.proxies.update(PROXIES)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    return session

def fetch_market_data():
    print("正在从 Yahoo Finance 下载 JPM 和 VIX 数据...")
    session = get_browser_session()
    
    jpm = yf.download("JPM", start=START_DATE, end=END_DATE, session=session, progress=False)[['Close']].rename(columns={'Close': 'JPM_Close'})
    time.sleep(2)
    vix = yf.download("^VIX", start=START_DATE, end=END_DATE, session=session, progress=False)[['Close']].rename(columns={'Close': 'VIX_Close'})
    
    if jpm.empty or vix.empty:
        raise ValueError("Yahoo Finance 返回了空数据，请尝试在代理软件中切换到其他节点。")
        
    market_df = pd.merge(jpm, vix, left_index=True, right_index=True, how='outer')
    return market_df

def fetch_macro_rates():
    print("正在从 FRED API 下载美国国债利率数据...")
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=DTB3&api_key={FRED_KEY}&file_type=json"
    
    try:
        response = requests.get(url, proxies=PROXIES, timeout=15, verify=False).json()
        
        if "error_message" in response:
            print(f"FRED API 报错: {response['error_message']}")
            print("使用备用方案：转向 Yahoo Finance 下载国债利率...")
            return fetch_macro_rates_backup()
            
        records = []
        for obs in response.get('observations', []):
            date = obs['date']
            val = obs['value']
            if START_DATE <= date <= END_DATE and val != '.':
                records.append({'Date': pd.to_datetime(date), 'Risk_Free_Rate': float(val) / 100.0})
                
        if not records:
            print("FRED 未返回指定范围内的数据，使用备用方案...")
            return fetch_macro_rates_backup()
            
        df_rates = pd.DataFrame(records, columns=['Date', 'Risk_Free_Rate']).set_index('Date')
        return df_rates
        
    except Exception as e:
        print(f"FRED 请求失败 ({e})，使用备用方案...")
        return fetch_macro_rates_backup()

def fetch_macro_rates_backup():
    print("正在从 Yahoo Finance 抓取 ^IRX (3个月国债利率代用值)...")
    session = get_browser_session()
    irx = yf.download("^IRX", start=START_DATE, end=END_DATE, session=session, progress=False)[['Close']]
    if irx.empty:
        raise ValueError("备用数据源 Yahoo Finance 也无法获取利率数据。")
    irx['Risk_Free_Rate'] = irx['Close'] / 100.0
    return irx[['Risk_Free_Rate']]

# 由于存在 Alpha Vantage 新闻情绪分数数据获取问题，此部分代码需停用。
# def fetch_sentiment_data():
#     print("正在从 Alpha Vantage 获取新闻情绪...")
#     all_records = []
    
#     # 调整策略：从当前最新时间开始往前无缝追溯
#     current_to = datetime.now().strftime("%Y%m%dT%H%M")
#     start_target = "20180101T0000"
    
#     request_count = 0
      
    
#     while current_to > start_target :
#         print(f"  进度: 正在抓取截止到 {current_to} 的前 1000 条新闻... (已请求 {request_count} 次)")
#         url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=JPM&time_to={current_to}&limit=1000&apikey={AV_KEY}"
        
#         try:
#             response = requests.get(url, proxies=PROXIES, timeout=15, verify=False).json()
            
#             if "Note" in response:
#                 print(f"  Alpha Vantage 频控提示: {response['Note']}")
#                 print("  触发限制。 ")
#                 time.sleep(60)
#                 continue
                
#             feed = response.get('feed', [])
#             if not feed:
#                 print("  未找到更多新闻，已到达该账户权限支持的历史数据最深边界。")
#                 break
                
#             oldest_time_in_batch = None
#             for item in feed:
#                 time_published = item['time_published']
                
#                 if oldest_time_in_batch is None or time_published < oldest_time_in_batch:
#                     oldest_time_in_batch = time_published
                
#                 # 时间过滤：只保留 2018 到 2024 年之间的数据
#                 if time_published < start_target or time_published > "20241231T2359":
#                     continue
                
#                 date_str = time_published[:8]
#                 date = pd.to_datetime(date_str, format='%Y%m%d')
                
#                 for ticker_sentiment in item.get('ticker_sentiment', []):
#                     if ticker_sentiment['ticker'] == 'JPM':
#                         score = float(ticker_sentiment['ticker_sentiment_score'])
#                         all_records.append({'Date': date, 'Sentiment_Score': score})
            
#             request_count += 1
            
#             # 确保时间戳确实在递减，防止死循环
#             if oldest_time_in_batch and oldest_time_in_batch < current_to:
#                 current_to = oldest_time_in_batch
#             else:
#                 break
                
#             time.sleep(1)
            
#         except Exception as api_err:
#             print(f"  当前批次请求失败: {api_err}")
#             break

#     df_sent = pd.DataFrame(all_records)
#     if not df_sent.empty:
#         df_sent = df_sent.groupby('Date').mean()
#     return df_sent

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    
    try:
        df_market = fetch_market_data()
        df_market.to_csv("D:/Git/Advanced-Chooser-Option-Pricing-Model-with-Real-World-Data-Machine-Learning/data/raw/jpm_vix_raw.csv")
        print("股票与 VIX 数据保存成功。")
    except Exception as e:
        print(f"股票与 VIX 数据抓取失败: {e}")
        
    try:
        df_rates = fetch_macro_rates()
        df_rates.to_csv("D:/Git/Advanced-Chooser-Option-Pricing-Model-with-Real-World-Data-Machine-Learning/data/raw/macro_rates_raw.csv")
        print("国债利率数据保存成功。")
    except Exception as e:
        print(f"国债利率数据抓取失败: {e}")
        
    # try:
    #     df_sent = fetch_sentiment_data()
    #     if not df_sent.empty:
    #         df_sent.to_csv("D:/Git/Advanced-Chooser-Option-Pricing-Model-with-Real-World-Data-Machine-Learning/data/raw/sentiment_raw.csv")
    #         print("新闻情绪历史数据保存成功。")
    #     else:
    #         print("未能过滤出 2018-2024 范围内的有效情绪数据。")
    # except Exception as e:
    #     print(f"新闻情绪历史数据抓取失败: {e}")
        
    print("/n数据抓取流程执行完毕。")
