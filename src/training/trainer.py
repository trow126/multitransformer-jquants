"""モデル学習モジュール"""
import tensorflow as tf
import numpy as np
import os
import json

def train_model(model, X_train, y_train, X_val=None, y_val=None, 
                epochs=100, batch_size=32, patience=10, model_path=None):
    """モデル学習用関数"""
    callbacks = []
    
    # 早期停止
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss' if X_val is not None else 'loss',
        patience=patience,
        restore_best_weights=True
    )
    callbacks.append(early_stopping)
    
    # モデル保存
    if model_path:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            save_best_only=True,
            monitor='val_loss' if X_val is not None else 'loss'
        )
        callbacks.append(checkpoint)
    
    # 学習実行
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val) if X_val is not None else None,
        callbacks=callbacks
    )
    
    # 学習履歴の保存
    if model_path:
        history_path = os.path.join(os.path.dirname(model_path), 'history.json')
        with open(history_path, 'w') as f:
            json.dump(history.history, f)
    
    return history