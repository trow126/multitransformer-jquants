"""モデル評価モジュール"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_volatility_model(y_true, y_pred):
    """ボラティリティ予測モデルの評価"""
    mse = np.mean((y_true - y_pred.flatten())**2)
    mae = np.mean(np.abs(y_true - y_pred.flatten()))
    
    results = {
        'mse': mse,
        'mae': mae
    }
    
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    
    return results

def evaluate_return_model(y_true, y_pred, percentile_high=80, percentile_low=20):
    """収益率予測モデルの評価"""
    # 予測値を基にしたトレーディングシグナル生成
    percentile_80 = np.percentile(y_pred, percentile_high)
    percentile_20 = np.percentile(y_pred, percentile_low)
    
    signals = np.zeros_like(y_pred)
    signals[y_pred > percentile_80] = 1  # ロング
    signals[y_pred < percentile_20] = -1  # ショート
    
    daily_returns = signals.flatten() * y_true  # シグナル×実際のリターン
    
    # 勝率の計算
    win_rate = np.mean((daily_returns > 0).astype(float))
    
    # シャープレシオの計算（年率換算）
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    
    # 累積リターンの計算
    cumulative_return = np.cumprod(1 + daily_returns) - 1
    
    results = {
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'cumulative_return': cumulative_return,
        'daily_returns': daily_returns
    }
    
    print(f"勝率: {win_rate:.4f}")
    print(f"シャープレシオ: {sharpe_ratio:.4f}")
    print(f"最終累積リターン: {cumulative_return[-1]:.4f}")
    
    return results