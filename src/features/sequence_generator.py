"""シーケンスデータ生成モジュール"""
import numpy as np
from tqdm import tqdm

def prepare_sequences(df, features, target, seq_length, garch_col=None):
    """時系列シーケンスデータの作成"""
    sequences = []
    targets = []
    garch_values = []
    
    for code, group in tqdm(df.groupby('Code')):
        group = group.sort_values('Date')
        
        feature_data = group[features].values
        target_data = group[target].values
        
        if garch_col is not None:
            garch_data = group[garch_col].values
        
        for i in range(len(group) - seq_length):
            if not np.isnan(target_data[i+seq_length]):
                seq = feature_data[i:i+seq_length]
                tar = target_data[i+seq_length]
                
                if not np.isnan(seq).any():
                    sequences.append(seq)
                    targets.append(tar)
                    
                    if garch_col is not None:
                        garch_values.append(garch_data[i+seq_length])
    
    if garch_col is not None:
        return np.array(sequences), np.array(targets), np.array(garch_values).reshape(-1, 1)
    else:
        return np.array(sequences), np.array(targets)