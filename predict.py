"""予測実行スクリプト"""
import os
import argparse
import yaml
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.jquants_client import JQuantsClient
from src.data.data_processor import standardize_code, filter_topix500
from src.features.feature_generator import calc_features_and_targets
from src.features.sequence_generator import prepare_sequences
from src.models.garch import fit_garch_models_by_code, prepare_garch_features
from src.utils.config import load_config

def parse_args():
    parser = argparse.ArgumentParser(description="MultiTransformerによる予測スクリプト")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="設定ファイルのパス")
    parser.add_argument("--prediction_type", type=str, choices=["volatility", "intraday_return"], 
                        default="volatility", help="予測タイプ")
    parser.add_argument("--include_garch", action="store_true", help="GARCHモデルを含める")
    parser.add_argument("--model_path", type=str, help="モデルファイルのパス")
    parser.add_argument("--output_path", type=str, default="predictions.csv", help="予測結果の出力パス")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 設定ファイルの読み込み
    config = load_config(args.config)
    
    # モデルの存在確認
    model_name = args.model_path or f"models/mt_{'garch_' if args.include_garch else ''}{args.prediction_type}.h5"
    if not os.path.exists(model_name):
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_name}")
    
    print("J-Quants APIからデータを取得中...")
    
    # J-Quants APIからデータ取得
    client = JQuantsClient(
        mail_address=config.get('jquants', {}).get('mail_address'),
        password=config.get('jquants', {}).get('password'),
        refresh_token=config.get('jquants', {}).get('refresh_token')
    )
    
    # データ期間の設定（直近のデータを取得）
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=365)  # 1年分のデータを取得（特徴量計算に必要）
    
    # TOPIX500銘柄の取得
    topix500_path = os.path.join(config['paths']['data_dir'], 'topix500.txt')
    if os.path.exists(topix500_path):
        # すでにファイルが存在する場合は読み込み
        with open(topix500_path, 'r') as f:
            topix500 = f.read().splitlines()
    else:
        # 新規取得
        topix500 = client.get_topix500_tickers(save_path=topix500_path)
    
    # 株価データの取得
    print("直近の株価データを取得中...")
    stock_price = client.get_stock_data(start_dt, end_dt)
    
    # データ前処理
    stock_price = standardize_code(stock_price)
    df_ohlcv = filter_topix500(stock_price, topix500)
    
    print("特徴量の計算中...")
    # 特徴量の計算
    df_feats = calc_features_and_targets(df_ohlcv, prediction_type=args.prediction_type)
    
    # 特徴量の定義
    if args.prediction_type == 'volatility':
        feature_cols = [col for col in df_feats.columns if 'log_return_lag' in col or 'volatility_5d' in col]
        target_col = 'target_vol'
    else:  # 'intraday_return'
        feature_cols = [col for col in df_feats.columns if 'log_return_lag' in col or 'volatility_5d' in col]
        target_col = 'target_return'
    
    # GARCHモデルの適用（オプション）
    if args.include_garch:
        print("GARCHモデルの学習中...")
        garch_results = fit_garch_models_by_code(df_feats)
        
        # GARCH予測値を特徴量として追加
        print("GARCH予測値を特徴量として追加中...")
        df_feats = prepare_garch_features(df_feats, garch_results)
        garch_col = 'garch_vol'
    else:
        garch_col = None
    
    print("シーケンスデータの準備中...")
    # シーケンスデータの準備（最新日のみ）
    seq_length = config['model']['seq_length']
    
    # 最新の日付でフィルタリング
    latest_date = df_feats['Date'].max()
    df_latest = df_feats[df_feats['Date'] == latest_date]
    
    # 各銘柄ごとに特徴量を抽出
    predictions = []
    
    for code, group in df_latest.groupby('Code'):
        df_code = df_feats[df_feats['Code'] == code].sort_values('Date')
        
        # シーケンスデータの準備に十分なデータがあるか確認
        if len(df_code) < seq_length + 1:
            continue
        
        # 最新のシーケンスを抽出
        feature_data = df_code[feature_cols].values
        
        # 最新のシーケンス（最後のseq_length分のデータ）
        seq = feature_data[-seq_length:]
        
        # 欠損値の有無を確認
        if np.isnan(seq).any():
            continue
        
        # シーケンスのバッチ次元を追加
        seq = np.expand_dims(seq, axis=0)
        
        # GARCHデータの準備（必要な場合）
        if args.include_garch:
            garch_value = df_code[garch_col].values[-1]
            if np.isnan(garch_value):
                continue
            
            garch_value = np.array([[garch_value]])
            X_pred = [seq, garch_value]
        else:
            X_pred = seq
        
        # モデルの読み込みと予測
        model = tf.keras.models.load_model(model_name, custom_objects={
            'PositionalEncodingLayer': PositionalEncodingLayer,
            'MultiTransformerLayer': MultiTransformerLayer
        })
        
        # 予測の実行
        pred = model.predict(X_pred)[0][0]
        
        # 結果を保存
        predictions.append({
            'Date': latest_date,
            'Code': code,
            'Prediction': pred
        })
    
    # 予測結果をDataFrameに変換
    df_predictions = pd.DataFrame(predictions)
    
    # 予測結果の保存
    df_predictions.to_csv(args.output_path, index=False)
    
    print(f"予測完了: {len(df_predictions)}銘柄の予測結果を{args.output_path}に保存しました")
    
    # トップとボトムの表示
    if args.prediction_type == 'volatility':
        print("\n予測ボラティリティ上位10銘柄:")
        print(df_predictions.sort_values('Prediction', ascending=False).head(10))
    else:  # 'intraday_return'
        print("\n予測収益率上位10銘柄 (ロング推奨):")
        print(df_predictions.sort_values('Prediction', ascending=False).head(10))
        
        print("\n予測収益率下位10銘柄 (ショート推奨):")
        print(df_predictions.sort_values('Prediction').head(10))

if __name__ == "__main__":
    # 必要なモジュールのインポート
    from src.models.layers import PositionalEncodingLayer, MultiTransformerLayer
    main()