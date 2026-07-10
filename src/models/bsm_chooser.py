import numpy as np
from scipy.stats import norm

class BsmChooserPricer:
    """基于 Rubinstein (1991) 封闭解的欧式选择权期权向量化定价器"""
    
    def __init__(self):
        pass

    @staticmethod
    def price_standard_european(s0, x, t, r, q, sigma, option_type='call'):
        """Black-Scholes-Merton 欧式期权标准解析解组件"""
        # 边界安全性防崩溃保护
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
        """
        利用复合拆解法计算选择权期权价格
        支持传入标量或同样长度的 Numpy Array
        """
        if np.any(t1 >= t2):
            raise ValueError("决策日 T1 必须严格早于最终到期日 T2。")
            
        # 1. 计算调整后的欧式看涨期权行权价 X'
        strike_prime = strike * np.exp(-(r - q) * (t2 - t1))
        
        # 2. 计算对应的 Call 组件 (期限为 T1)
        call_component = self.price_standard_european(s0, strike_prime, t1, r, q, sigma, option_type='call')
        
        # 3. 计算对应的 Put 组件 (期限为 T2)
        put_component = self.price_standard_european(s0, strike, t2, r, q, sigma, option_type='put')
        
        # 4. 组装溢价
        chooser_value = np.exp(-q * (t2 - t1)) * call_component + put_component
        return np.round(chooser_value, 4)