"""カスタムレイヤー定義"""
import tensorflow as tf
import numpy as np

class PositionalEncodingLayer(tf.keras.layers.Layer):
    """Transformerに必須のPositional Encoding"""
    def __init__(self, sequence_length, d_model):
        super(PositionalEncodingLayer, self).__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model
        
    def build(self, input_shape):
        # 正弦波と余弦波を使用したPositional Encoding
        position = np.arange(self.sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pe = np.zeros((self.sequence_length, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.convert_to_tensor(pe[np.newaxis, ...], dtype=tf.float32)
        
    def call(self, inputs):
        # 入力にPositional Encodingを加算
        return inputs + self.pe


class MultiTransformerLayer(tf.keras.layers.Layer):
    """論文Figure 5に忠実なMultiTransformerレイヤー"""
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1, num_transformers=3):
        super(MultiTransformerLayer, self).__init__()
        self.num_transformers = num_transformers
        self.d_model = d_model
        
        # 複数のMulti-Head Attentionを作成
        self.attention_layers = [
            tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=d_model//num_heads, dropout=dropout_rate)
            for _ in range(num_transformers)
        ]
        
        # 共通の層（論文Figure 5に準拠）
        self.attention_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.attention_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # 共通FFN部分（Attentionの出力平均後に適用）
        self.ffn1 = tf.keras.layers.Dense(ff_dim, activation='relu')
        self.ffn2 = tf.keras.layers.Dense(d_model)
        self.ffn_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.ffn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, inputs, training=False, mask=None):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # 各Multi-Head Attentionの出力を格納するリスト
        attention_outputs = []
        
        for i in range(self.num_transformers):
            # 論文セクション2.5に基づくバギングアプローチ
            # 観測データ（行/サンプル）の90%をランダムに選択
            if training:
                # 各時系列の90%のタイムステップをランダムに選択（論文準拠）
                sample_rate = 0.9
                random_indices = tf.random.shuffle(tf.range(seq_len))
                num_samples = tf.cast(tf.math.ceil(sample_rate * tf.cast(seq_len, tf.float32)), tf.int32)
                selected_indices = random_indices[:num_samples]
                
                # 選択したタイムステップのみのデータを抽出
                input_sample = tf.gather(inputs, selected_indices, axis=1)
                
                # Multi-Head Attention
                attn_output = self.attention_layers[i](input_sample, input_sample)
                
                # 元のシーケンス長に戻す（padded_attnの作成）
                padded_attn = tf.zeros_like(inputs)
                
                for b in range(batch_size):
                    # 各バッチに対して選択したインデックスに出力を配置
                    idx_tensor = tf.stack([
                        tf.ones(num_samples, dtype=tf.int32) * b,
                        selected_indices
                    ], axis=1)
                    padded_attn = tf.tensor_scatter_nd_update(
                        padded_attn, idx_tensor, attn_output[b]
                    )
                
                attention_outputs.append(padded_attn)
            else:
                # 推論時は全データを使用
                attn_output = self.attention_layers[i](inputs, inputs)
                attention_outputs.append(attn_output)
        
        # T個のアテンション出力の平均（論文Equation 20: AMHに準拠）
        avg_attention = tf.reduce_mean(attention_outputs, axis=0)
        
        # 残りは通常のTransformerブロックの処理
        avg_attention = self.attention_dropout(avg_attention, training=training)
        out1 = self.attention_norm(inputs + avg_attention)  # 残差接続 + 正規化
        
        # 共通のFFN（平均化されたアテンション出力に対して適用）
        ffn_output = self.ffn1(out1)
        ffn_output = self.ffn2(ffn_output)
        ffn_output = self.ffn_dropout(ffn_output, training=training)
        out2 = self.ffn_norm(out1 + ffn_output)  # 残差接続 + 正規化
        
        return out2