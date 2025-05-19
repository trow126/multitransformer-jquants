# MultiTransformer for J-Quants

日本株データに対して MultiTransformer モデルを適用し、ボラティリティ予測と日中収益率予測を行うプロジェクトです。

## 概要

このプロジェクトでは、J-Quants API から取得した日本株データに対して、MultiTransformer モデルを適用し、以下の 2 つの予測タスクを実行します：

1. **ボラティリティ予測**: 将来の株価変動の大きさ（ボラティリティ）を予測します。
2. **日中収益率予測**: 翌日の始値から終値までの収益率を予測します。

MultiTransformer は、複数の Transformer エンコーダーを平行に配置し、その出力を集約することで、アンサンブル効果を得るモデルです。従来の GARCH モデルと組み合わせることで、より優れたボラティリティ予測性能を発揮します。

## 特徴

- 複数の Transformer モデルを並列に使用するアンサンブルアプローチ
- GARCH モデルとの統合による予測性能の向上
- J-Quants API を使用した日本株データの取得と処理
- 時系列特徴量の自動生成
- モデルの訓練と評価のための包括的なパイプライン

## インストール

以下のコマンドでリポジトリをクローンし、必要なパッケージをインストールしてください：

```bash
git clone https://github.com/yourusername/multitransformer-jquants.git
cd multitransformer-jquants
pip install -r requirements.txt
```

## 必要なパッケージ

以下の主要なパッケージが必要です：

- tensorflow
- numpy
- pandas
- matplotlib
- arch (GARCH モデル用)
- jquants-api-client
- pyyaml
- tqdm
- scikit-learn
- jupyterlab (ノートブック用)

## 使用方法

### データの取得

J-Quants API を使用するには、API アクセストークンの設定が必要です。以下のいずれかの方法で設定できます：

1. 環境変数に設定
```bash
export JQUANTS_API_MAIL_ADDRESS=your_email
export JQUANTS_API_PASSWORD=your_password
export JQUANTS_API_REFRESH_TOKEN=your_token
```

2. `.env` ファイルを作成して設定
```
JQUANTS_API_MAIL_ADDRESS=your_email
JQUANTS_API_PASSWORD=your_password
JQUANTS_API_REFRESH_TOKEN=your_token
```

### モデルの学習

以下のコマンドでモデルの学習を実行できます：

```bash
# ボラティリティ予測（GARCH モデル統合あり）
python main.py --prediction_type volatility --include_garch

# 日中収益率予測（GARCH モデル統合なし）
python main.py --prediction_type intraday_return
```

### 予測の実行

学習済みモデルを使用して予測を実行するには：

```bash
# ボラティリティ予測
python predict.py --prediction_type volatility --include_garch

# 日中収益率予測
python predict.py --prediction_type intraday_return
```

### Jupyter Notebook での分析

`notebooks` ディレクトリ内のノートブックを使用して、データ探索や分析を行うことができます：

```bash
jupyter lab notebooks/01_data_exploration.ipynb
```

## プロジェクト構成

```
multitransformer-jquants/
│
├── data/                          # データ関連のディレクトリ
│   ├── raw/                       # J-Quantsから取得した生データの保存先
│   └── processed/                 # 前処理済みデータの保存先
│
├── models/                        # 学習済みモデルの保存先
│   ├── transformer/               # Transformerモデル
│   └── garch/                     # GARCHモデル
│
├── notebooks/                     # Jupyter notebookディレクトリ
│   ├── 01_data_exploration.ipynb  # データ探索用ノートブック
│   ├── 02_feature_analysis.ipynb  # 特徴量分析用ノートブック
│   └── 03_model_evaluation.ipynb  # モデル評価用ノートブック
│
├── src/                           # ソースコードディレクトリ
│   ├── data/                      # データ処理モジュール
│   ├── features/                  # 特徴量エンジニアリングモジュール
│   ├── models/                    # モデル定義モジュール
│   ├── training/                  # 学習関連モジュール
│   └── utils/                     # ユーティリティモジュール
│
├── configs/                       # 設定ファイルディレクトリ
│   ├── model_config.yaml          # モデル設定
│   ├── data_config.yaml           # データ設定
│   └── train_config.yaml          # 学習設定
│
├── tests/                         # テストコード
│
├── requirements.txt               # 依存パッケージリスト
├── setup.py                       # パッケージインストール設定
├── README.md                      # プロジェクト説明
├── main.py                        # メインスクリプト（学習実行）
└── predict.py                     # 予測実行スクリプト
```

## モデルアーキテクチャ

MultiTransformer モデルは以下の特徴を持ちます：

1. 複数の Transformer エンコーダーを並列に配置
2. 各 Transformer はランダムにサンプリングされたデータで学習（バギング手法）
3. 各 Transformer の出力を平均化して最終出力を生成
4. GARCH モデルからの予測値を追加特徴量として統合（オプション）

## ライセンス

MIT

## 謝辞

- J-Quants プロジェクト（日本取引所グループと東京証券取引所）
- Transformer と GARCH モデルの研究コミュニティ