"""評価指標"""
import numpy as np

def calculate_sharpe_ratio(returns, annualization_factor=252):
    """シャープレシオの計算"""
    return np.mean(returns) / np.std(returns) * np.sqrt(annualization_factor)

def calculate_sortino_ratio(returns, risk_free_rate=0, target_return=0, annualization_factor=252):
    """ソルティノレシオの計算"""
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < target_return]
    downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
    
    if downside_deviation == 0:
        return np.nan
    
    return (np.mean(excess_returns) / downside_deviation) * np.sqrt(annualization_factor)

def calculate_max_drawdown(returns):
    """最大ドローダウンの計算"""
    cumulative_returns = np.cumprod(1 + returns) - 1
    # 累積最大値を計算
    cumulative_max = np.maximum.accumulate(cumulative_returns)
    # ドローダウンを計算
    drawdowns = (cumulative_returns - cumulative_max) / (1 + cumulative_max)
    # 最大ドローダウンを返す
    return np.min(drawdowns)

def calculate_win_rate(returns):
    """勝率の計算"""
    return np.mean(returns > 0)

def calculate_profit_loss_ratio(returns):
    """損益率の計算"""
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    
    if len(negative_returns) == 0 or np.mean(np.abs(negative_returns)) == 0:
        return np.inf
    
    return np.mean(positive_returns) / np.mean(np.abs(negative_returns))