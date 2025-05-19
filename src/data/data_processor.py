"""データ前処理モジュール"""
import pandas as pd
import numpy as np

def standardize_code(df):
    """銘柄コードの標準化"""
    df["Code"] = df["Code"].astype(str)
    # 普通株 (5桁で末尾が0) の銘柄コードを4桁に変換
    df.loc[(df["Code"].str.len() == 5) & (df["Code"].str[-1] == "0"), "Code"] = \
        df.loc[(df["Code"].str.len() == 5) & (df["Code"].str[-1] == "0"), "Code"].str[:-1]
    return df

def filter_topix500(df, tickers):
    """TOPIX500銘柄のみを抽出"""
    return df[df['Code'].isin(tickers)]

def train_test_split_by_date(df, split_date):
    """日付でデータを訓練用とテスト用に分割"""
    df_train = df[df['Date'] < split_date]
    df_test = df[df['Date'] >= split_date]
    return df_train, df_test