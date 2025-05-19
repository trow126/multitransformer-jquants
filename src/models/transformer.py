"""MultiTransformerモデル定義"""
import tensorflow as tf
from src.models.layers import PositionalEncodingLayer, MultiTransformerLayer

def build_mt_garch_model(seq_length, num_features, prediction_type='volatility', include_garch=True):
    """論文Figure 5, 6に準拠したMultiTransformer-GARCHモデル"""
    # 通常の特徴量入力
    inputs = tf.keras.layers.Input(shape=(seq_length, num_features))
    
    # Positional Encoding（論文Figure 3, 5に必須）
    x = PositionalEncodingLayer(seq_length, num_features)(inputs)
    
    # MultiTransformerレイヤー（論文Figure 5に準拠）
    x = MultiTransformerLayer(
        d_model=num_features, 
        num_heads=4, 
        ff_dim=num_features*4,
        dropout_rate=0.1,
        num_transformers=3
    )(x, training=True)
    
    # シーケンス次元の集約
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # GARCHモデルからの特徴量入力（論文Figure 6のハイブリッドモデル）
    if include_garch:
        garch_input = tf.keras.layers.Input(shape=(1,))
        
        # Transformerの出力とGARCH予測を結合
        x = tf.keras.layers.Concatenate()([x, garch_input])
        
        inputs_list = [inputs, garch_input]
    else:
        inputs_list = inputs
    
    # 最終的な予測層（論文Figure 5, 6準拠）
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    
    # 出力層（予測タイプに応じて活性化関数を調整）
    if prediction_type == 'volatility':
        outputs = tf.keras.layers.Dense(1, activation='softplus')(x)  # 非負制約
    else:  # 'intraday_return'
        outputs = tf.keras.layers.Dense(1)(x)  # 制約なし
    
    model = tf.keras.models.Model(inputs=inputs_list, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model