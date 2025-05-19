"""可視化関数"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_volatility_prediction(y_true, y_pred, n_samples=100):
    """ボラティリティ予測結果の可視化"""
    plt.figure(figsize=(12, 6))
    
    # ランダムに100サンプル選択して散布図を作成
    if len(y_true) > n_samples:
        indices = np.random.choice(len(y_true), n_samples, replace=False)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred.flatten()[indices]
    else:
        y_true_sample = y_true
        y_pred_sample = y_pred.flatten()
    
    # 散布図
    plt.scatter(y_true_sample, y_pred_sample, alpha=0.5)
    
    # 完全予測の直線
    max_val = max(np.max(y_true_sample), np.max(y_pred_sample))
    min_val = min(np.min(y_true_sample), np.min(y_pred_sample))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title('実現ボラティリティ vs 予測ボラティリティ')
    plt.xlabel('実現ボラティリティ')
    plt.ylabel('予測ボラティリティ')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def plot_cumulative_returns(cumulative_returns):
    """累積リターンの可視化"""
    plt.figure(figsize=(12, 6))
    
    # 累積リターンの推移
    plt.plot(cumulative_returns)
    plt.title('戦略の累積リターン')
    plt.xlabel('取引日数')
    plt.ylabel('累積リターン')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def plot_feature_importance(model, feature_names):
    """特徴量重要度の可視化"""
    # Transformerモデルの場合はshapを使用する必要があるため
    # 実装は省略します（単純な特徴量重要度の取得が難しいため）
    pass