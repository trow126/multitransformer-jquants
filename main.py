"""MultiTransformerモデル学習実行スクリプト"""
import os
import argparse
import yaml
import tensorflow as tf
from datetime import datetime, timedelta

from src.data.jquants_client import JQuantsClient
from src.data.data_processor import standardize_code, filter_topix500, train_test_split_by_date
from src.features.feature_generator import calc_features_and_targets
from src.features.sequence_generator import prepare_sequences
from src.models.transformer import build_mt_garch_model
from src.models.garch import fit_garch_models_by_code, prepare_garch_features
from src.training.trainer import train_model
from src.training.evaluation import evaluate_volatility_model, evaluate_return_model
from src.utils.visualization import plot_volatility_prediction, plot_cumulative_returns
from src.utils.config import load_config

def parse_args():
    parser = argparse.ArgumentParser(description="MultiTransformer学習スクリプト")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="設定ファイルのパス")
    parser.add_argument("--prediction_type", type=str, choices=["volatility", "intraday_return"], 
                        default="volatility", help="予測タイプ")
    parser.add_argument("--include_garch", action="store_true", help="GARCHモデルを含める")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 設定ファイルの読み込み
    config = load_config(args.config)
    
    # データディレクトリの作成
    os.makedirs(config['paths']['data_dir'], exist_ok=True)
    os.makedirs(config['paths']['models_dir'], exist_ok=True)
    
    print("J-Quants APIからデータを取得中...")
    
    # J-Quants APIからデータ取得
    client = JQuantsClient(
        mail_address=config.get('jquants', {}).get('mail_address'),
        password=config.get('jquants', {}).get('password'),
        refresh_token=config.get('jquants', {}).get('refresh_token')
    )
    
    # データ期間の設定
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=365 * config['data']['historical_years'])
    
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
    stock_price_path = os.path.join(config['paths']['data_dir'], 'stock_price.csv.gz')
    if os.path.exists(stock_price_path):
        # すでにファイルが存在する場合は読み込み
        stock_price = pd.read_csv(stock_price_path, compression='gzip')
    else:
        # 新規取得
        stock_price = client.get_stock_data(start_dt, end_dt, save_path=stock_price_path)
    
    # データ前処理
    stock_price = standardize_code(stock_price)
    df_ohlcv = filter_topix500(stock_price, topix500)
    
    print("特徴量と目的変数の計算中...")
    # 特徴量と目的変数の計算
    df_feats = calc_features_and_targets(df_ohlcv, prediction_type=args.prediction_type)
    
    # データの分割
    split_date = end_dt - timedelta(days=365 * config['data']['test_years'])
    df_train, df_test = train_test_split_by_date(df_feats, split_date)
    
    # 特徴量とターゲットの定義
    if args.prediction_type == 'volatility':
        feature_cols = [col for col in df_feats.columns if 'log_return_lag' in col or 'volatility_5d' in col]
        target_col = 'target_vol'
    else:  # 'intraday_return'
        feature_cols = [col for col in df_feats.columns if 'log_return_lag' in col or 'volatility_5d' in col]
        target_col = 'target_return'
    
    # GARCHモデルの適用（オプション）
    if args.include_garch:
        print("GARCHモデルの学習中...")
        garch_results = fit_garch_models_by_code(df_train)
        
        # GARCH予測値を特徴量として追加
        print("GARCH予測値を特徴量として追加中...")
        df_train = prepare_garch_features(df_train, garch_results)
        df_test = prepare_garch_features(df_test, garch_results)
        garch_col = 'garch_vol'
    else:
        garch_col = None
    
    print("シーケンスデータの準備中...")
    # シーケンスデータの準備
    seq_length = config['model']['seq_length']
    
    if args.include_garch:
        X_train_seq, y_train, garch_train = prepare_sequences(
            df_train, feature_cols, target_col, seq_length, garch_col=garch_col
        )
        X_test_seq, y_test, garch_test = prepare_sequences(
            df_test, feature_cols, target_col, seq_length, garch_col=garch_col
        )
        
        X_train = [X_train_seq, garch_train]
        X_test = [X_test_seq, garch_test]
    else:
        X_train_seq, y_train = prepare_sequences(
            df_train, feature_cols, target_col, seq_length
        )
        X_test_seq, y_test = prepare_sequences(
            df_test, feature_cols, target_col, seq_length
        )
        
        X_train = X_train_seq
        X_test = X_test_seq
    
    print("モデル構築中...")
    # モデル構築
    model = build_mt_garch_model(
        seq_length=seq_length,
        num_features=X_train_seq.shape[2],
        prediction_type=args.prediction_type,
        include_garch=args.include_garch
    )
    
    # モデル保存パス
    model_name = f"mt_{'garch_' if args.include_garch else ''}{args.prediction_type}"
    model_path = os.path.join(config['paths']['models_dir'], f"{model_name}.h5")
    
    print("モデル学習中...")
    # モデル学習
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        patience=config['training']['patience'],
        model_path=model_path
    )
    
    print("テストデータでの予測中...")
    # テストデータでの予測
    y_pred = model.predict(X_test)
    
    # 評価
    print("評価結果:")
    if args.prediction_type == 'volatility':
        results = evaluate_volatility_model(y_test, y_pred)
        plot_volatility_prediction(y_test, y_pred)
    else:  # 'intraday_return'
        results = evaluate_return_model(y_test, y_pred)
        plot_cumulative_returns(results['cumulative_return'])
    
    print("学習完了")

if __name__ == "__main__":
    # pandasのインポート
    import pandas as pd
    main()