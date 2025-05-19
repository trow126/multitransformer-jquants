"""GARCHモデル定義"""
import pandas as pd
import numpy as np
from arch import arch_model
from tqdm import tqdm

def fit_garch_models_by_code(df_train):
    """各銘柄ごとのGARCH(1,1)モデルを作成し予測値を返す"""
    garch_results = {}
    
    for code in tqdm(df_train['Code'].unique()):
        # 銘柄ごとのデータ抽出
        train_data = df_train[df_train['Code'] == code]['log_return'].dropna()
        
        if len(train_data) > 30:  # 十分なデータがある場合のみモデル作成
            try:
                # GARCH(1,1)モデルの作成と学習
                garch_model = arch_model(train_data, vol='Garch', p=1, q=1)
                garch_result = garch_model.fit(disp='off')
                
                # データと同じインデックスでボラティリティ予測値を返す
                forecast = garch_result.conditional_volatility
                
                # 銘柄コードとインデックスのマッピングを保存
                garch_results[code] = pd.Series(
                    forecast, 
                    index=train_data.index
                )
            except Exception as e:
                print(f"Error fitting GARCH model for code {code}: {str(e)}")
    
    return garch_results

def prepare_garch_features(df, garch_results):
    """GARCHモデルの予測値を特徴量として追加"""
    df_with_garch = df.copy()
    df_with_garch['garch_vol'] = np.nan
    
    for code, garch_series in garch_results.items():
        # この銘柄のインデックスを特定
        code_idx = df_with_garch['Code'] == code
        
        # インデックスが一致する行にGARCH予測値を代入
        for idx, value in garch_series.items():
            date_idx = df_with_garch['Date'] == idx
            df_with_garch.loc[code_idx & date_idx, 'garch_vol'] = value
    
    # 欠損値を前方補完
    df_with_garch['garch_vol'] = df_with_garch.groupby('Code')['garch_vol'].fillna(method='ffill')
    
    # それでも残る欠損値を0で埋める
    df_with_garch['garch_vol'] = df_with_garch['garch_vol'].fillna(0)
    
    return df_with_garch