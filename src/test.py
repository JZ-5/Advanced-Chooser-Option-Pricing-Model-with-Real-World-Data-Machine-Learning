import os
import requests
import urllib3
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# 屏蔽 HTTPS 警告并加载环境变量
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

# 1. 冠军 GBDT 模型预测输出 (2026-07-27)
MODEL_PREDICTIONS = {
    0.25: 215.08,
    0.50: 211.31,
    0.75: 217.84
}

# 基础物理参数
T2 = 1.0          # 总到期期限 1 年
STRIKE = 150.0    # 标的行权价 K
R = 0.0381        # 无风险利率 3.81%
Q = 0.0168        # 滚动股息率 1.68%

def fetch_av_options_data(symbol="JPM"):
    """
    调用 HISTORICAL_OPTIONS 接口（兼容普通与高级 API Key，自动取最新交易日），
    并进行强壮的列名兼容与脏数据清洗。
    """
    av_key = os.getenv("ALPHA_VANTAGE_KEY")
    if not av_key:
        raise ValueError(" 未在 .env 环境变量中检测到 ALPHA_VANTAGE_KEY，请检查配置！")
        
    print(f" 正在通过 Alpha Vantage API 抓取 {symbol} 最新期权链数据...")
    
    # 使用 HISTORICAL_OPTIONS 接口，不加 date 参数会自动返回最近一个交易日的完整期权链
    url = f"https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol={symbol}&apikey={av_key}"
    proxies = None if os.getenv('GITHUB_ACTIONS') else {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
    
    try:
        response = requests.get(url, proxies=proxies, timeout=25, verify=False).json()
    except Exception as e:
        raise RuntimeError(f" 请求 Alpha Vantage API 失败: {e}")
        
    # 打印诊断提示（如有）
    if "Information" in response:
        print(f" Alpha Vantage 提示: {response['Information']}")
    if "Error Message" in response:
        raise RuntimeError(f" Alpha Vantage 返回错误: {response['Error Message']}")
        
    data = response.get("data", [])
    if not data:
        raise RuntimeError(f" API 未返回 'data' 字段。响应结果为: {response}")
        
    df_options = pd.DataFrame(data)
    
    # 1. 自动适配到期日列名 (兼容 expiration / expiration_date)
    exp_col = None
    for candidate in ['expiration', 'expiration_date', 'expirationDate', 'expiry']:
        if candidate in df_options.columns:
            exp_col = candidate
            break
            
    if exp_col is None:
        raise KeyError(f" 无法在返回的数据集中找到到期日列，现有列名为: {list(df_options.columns)}")
        
    # 2. 安全解析日期 (format='mixed' 避免 UserWarning)
    df_options['expiration_clean'] = pd.to_datetime(df_options[exp_col], format='mixed', errors='coerce')
    df_options = df_options.dropna(subset=['expiration_clean']).copy()
    
    # 过滤掉占位年份 (如 2099 年)
    df_options = df_options[df_options['expiration_clean'].dt.year <= 2030].copy()
    
    if df_options.empty:
        raise ValueError("❌ 清洗无效日期后剩余记录数为 0，请检查 API 返回内容。")
        
    # 3. 标准化列名与数值
    df_options['expiration'] = df_options['expiration_clean']
    df_options['strike'] = df_options['strike'].astype(float)
    
    # 4. 清洗挂单价格
    for col in ['bid', 'ask', 'mark', 'last']:
        if col in df_options.columns:
            df_options[col] = pd.to_numeric(df_options[col], errors='coerce').fillna(0.0)
        else:
            df_options[col] = 0.0
            
    # 计算中间价 (Bid-Ask Midpoint)
    df_options['price'] = np.where(
        (df_options['bid'] > 0) & (df_options['ask'] > 0),
        (df_options['bid'] + df_options['ask']) / 2.0,
        np.where(df_options['mark'] > 0, df_options['mark'], df_options['last'])
    )
    
    print(f" 成功清洗并提取到 {len(df_options)} 条 Alpha Vantage 有效期权挂单数据！")
    return df_options

def calculate_market_chooser_price():
    # 1. 抓取与清洗期权数据
    df_opts = fetch_av_options_data("JPM")
    
    today = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
    expirations = sorted(df_opts['expiration'].unique())
    
    print("\n" + "="*75)
    print(" Alpha Vantage 真实期权对冲复制 vs. Champion GBDT 估值对比表")
    print("="*75)
    
    # 识别期权类型列 (type / option_type)
    type_col = 'type' if 'type' in df_opts.columns else 'option_type'
    
    for t1 in [0.25, 0.50, 0.75]:
        # A. 计算调整行权价 K'
        strike_prime = STRIKE * np.exp(-(R - Q) * (T2 - t1))
        
        # B. 匹配最接近 T1 和 T2 目标时间的到期日
        target_t1_dt = today + pd.Timedelta(days=int(t1 * 365))
        target_t2_dt = today + pd.Timedelta(days=int(T2 * 365))
        
        best_exp_t1 = min(expirations, key=lambda x: abs(x - target_t1_dt))
        best_exp_t2 = min(expirations, key=lambda x: abs(x - target_t2_dt))
        
        # C. 筛选 Call (T1) 合约并匹配 strike_prime
        calls = df_opts[(df_opts[type_col].str.lower() == 'call') & (df_opts['expiration'] == best_exp_t1)]
        if calls.empty:
            continue
        best_call_idx = (calls['strike'] - strike_prime).abs().idxmin()
        best_call = calls.loc[best_call_idx]
        call_price = best_call['price']
        
        # D. 筛选 Put (T2) 合约并匹配 STRIKE (150)
        puts = df_opts[(df_opts[type_col].str.lower() == 'put') & (df_opts['expiration'] == best_exp_t2)]
        if puts.empty:
            continue
        best_put_idx = (puts['strike'] - STRIKE).abs().idxmin()
        best_put = puts.loc[best_put_idx]
        put_price = best_put['price']
        
        # E. 根据 Rubinstein 对冲复制公式计算真实 Chooser 组合价格
        real_market_chooser = call_price * np.exp(-Q * (T2 - t1)) + put_price
        
        # F. 误差校验
        model_pred = MODEL_PREDICTIONS[t1]
        abs_err = abs(model_pred - real_market_chooser)
        pct_err = (abs_err / real_market_chooser) * 100
        
        exp_t1_str = best_exp_t1.strftime('%Y-%m-%d')
        exp_t2_str = best_exp_t2.strftime('%Y-%m-%d')
        
        print(f" 决策尺度 T1 = {t1:<4} 年 (匹配到期日: Call@{exp_t1_str}, Put@{exp_t2_str}):")
        print(f"   ├─ Call 合约(行权价=${best_call['strike']:.1f}) 真实报价 : ${call_price:.2f}")
        print(f"   ├─ Put  合约(行权价=${best_put['strike']:.1f}) 真实报价   : ${put_price:.2f}")
        print(f"   ├─ Alpha Vantage 组合复制真实价 (Actual) : ${real_market_chooser:.2f}")
        print(f"   └─ Champion GBDT 模型预测报价 (Pred)    : ${model_pred:.2f} | 绝对误差: ${abs_err:.2f} (相对误差: {pct_err:.2f}%) \n")
        
    print("="*75)

if __name__ == "__main__":
    try:
        calculate_market_chooser_price()
    except Exception as err:
        print(f"\n 执行失败: {err}")