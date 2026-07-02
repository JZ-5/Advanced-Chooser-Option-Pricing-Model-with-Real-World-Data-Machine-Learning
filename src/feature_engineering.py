import pandas as pd
import numpy as np
from pathlib import Path

def run_advanced_data_pipeline():
    print("开始执行数据清洗与特征工程一体化管道...")
    
    # 1. 路径自适应对齐
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    
    market_raw_path = project_root / "data" / "raw" / "jpm_vix_raw.csv"
    rates_raw_path = project_root / "data" / "raw" / "macro_rates_raw(fred).csv"
    output_final_path = project_root / "data" / "processed" / "features_ann_final.csv"
    
    if not market_raw_path.exists() or not rates_raw_path.exists():
        raise FileNotFoundError("未能在指定路径找到原始数据 CSV 文件，请检查 data/raw 目录。")
        
    # --- 2. 多源时间对齐与表头清洗 ---
    df_market = pd.read_csv(market_raw_path)
    df_market = df_market.iloc[2:].copy()
    df_market = df_market.rename(columns={'Price': 'Date'})
    
    df_market['Date'] = pd.to_datetime(df_market['Date']).dt.normalize()
    df_market['JPM_Close'] = pd.to_numeric(df_market['JPM_Close'])
    df_market['VIX_Close'] = pd.to_numeric(df_market['VIX_Close'])
    df_market = df_market.drop_duplicates(subset=['Date']).set_index('Date')
    
    df_rates = pd.read_csv(rates_raw_path)
    df_rates['Date'] = pd.to_datetime(df_rates['Date']).dt.normalize()
    df_rates['Risk_Free_Rate'] = pd.to_numeric(df_rates['Risk_Free_Rate'])
    df_rates = df_rates.drop_duplicates(subset=['Date']).set_index('Date')
    
    # 纵向外连接合并，保留两端所有日期
    df = pd.merge(df_market, df_rates, left_index=True, right_index=True, how='outer')
    
    # 先进行价格利率的基础插值
    df = df.interpolate(method='linear').ffill().bfill()
    
    # --- 3. 异常值处理 (IQR 截断法优化) ---
    # 核心修正：JPM_Close 具有趋势性，VIX_Close 具有极端肥尾危机特征，绝不能应用 IQR 截断。
    # 遵循量化规范，我们仅对可能存在异常噪点的宏观基准 Risk_Free_Rate 运行 IQR 异常清洗。
    for col in ['Risk_Free_Rate']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[col] = np.clip(df[col], lower_bound, upper_bound)
        
    print("时序基础净化已完成。开始计算真实的传统与高级量化特征...")
    
    # --- 4. 核心特征工程衍生 ---
    # 传统金融特征 (Traditional Features)
    df['Daily_Return'] = np.log(df['JPM_Close'] / df['JPM_Close'].shift(1))
    df['Rolling_Vol_20d'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    df['Dividend_Growth_Proxy'] = np.log(df['JPM_Close'] / df['JPM_Close'].shift(252)).rolling(window=20).mean()
    
    # 高级金融特征 (Advanced Features)
    df['VIX_Decimal'] = df['VIX_Close'] / 100.0
    vix_delta = df['VIX_Decimal'].diff()
    df['VIX_JPM_Corr_20d'] = df['Daily_Return'].rolling(window=20).corr(vix_delta)
    df['IR_Momentum_10d'] = df['Risk_Free_Rate'].rolling(window=10).mean()
    
    # 基础与附加衍生特征
    df['JPM_SMA20_Disparity'] = (df['JPM_Close'] / df['JPM_Close'].rolling(window=20).mean()) - 1
    df['IV_RV_Spread'] = df['VIX_Decimal'] - df['Rolling_Vol_20d']
    df['Rate_Delta'] = df['Risk_Free_Rate'].diff()
    
    # --- 5. 采用智能填充向后修复冷启动缺失 ---
    df_filled = df.bfill().ffill()
    
    # --- 6. 严格截取目标区间 ---
    start_boundary = pd.to_datetime("2018-01-01")
    end_boundary = pd.to_datetime("2024-12-31")
    df_output = df_filled[(df_filled.index >= start_boundary) & (df_filled.index <= end_boundary)]
    
    # 锁定最终导出的 11 个核心特征大表
    feature_columns = [
        'JPM_Close', 'VIX_Decimal', 'Risk_Free_Rate',
        'Daily_Return', 'Rolling_Vol_20d', 'Dividend_Growth_Proxy',
        'VIX_JPM_Corr_20d', 'IR_Momentum_10d',
        'JPM_SMA20_Disparity', 'IV_RV_Spread', 'Rate_Delta'
    ]
    df_final = df_output[feature_columns].round(4)
    
    # 保存至 processed 生产环境目录
    output_final_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_final_path)
    
    print("特征工程管道重新执行完毕。")
    print(f"成功恢复并交付样本交易日天数: {len(df_final)} 天")
    print(f"实际交付时间跨度: 从 {df_final.index.min().strftime('%Y-%m-%d')} 到 {df_final.index.max().strftime('%Y-%m-%d')}")
    print("\n特征数据前 3 行验证预览:")
    print(df_final.head(3))

if __name__ == "__main__":
    run_advanced_data_pipeline()