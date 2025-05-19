"""特徴量生成モジュール"""
import numpy as np
import pandas as pd

def calc_features_and_targets(df_ohlcv, prediction_type='volatility'):
    """論文セクション2.1に準拠した特徴量と目的変数の計算"""
    df_feats = df_ohlcv[['Date','Code','Open','Close','Volume']].copy()
    
    # 対数収益率（論文式(1)）
    df_feats['log_return'] = np.log(df_ohlcv['Close'] / df_ohlcv['Close'].shift(1))
    
    # 特徴量：過去の対数収益率とそのラグ（論文式(2)）
    for i in range(1, 11):  # ラグ1から10まで
        df_feats[f'log_return_lag{i}'] = df_feats['log_return'].shift(i)
    
    # 特徴量：過去5日間の対数収益率の標準偏差とそのラグ（論文式(3)）
    df_feats['volatility_5d'] = df_feats['log_return'].rolling(5).std()
    for i in range(1, 11):  # ラグ1から10まで
        df_feats[f'volatility_5d_lag{i}'] = df_feats['volatility_5d'].shift(i)
    
    # 目的変数の計算
    if prediction_type == 'volatility':
        # 将来5日間の実現ボラティリティ（論文式(4)）
        df_feats['target_vol'] = df_feats['log_return'].rolling(5).std().shift(-5)
    else:  # 'intraday_return'
        # 翌日の日中収益率（寄り引け戦略用）
        df_feats['target_return'] = (df_ohlcv['Close'] / df_ohlcv['Open'] - 1.0).shift(-1)
    
    return df_feats