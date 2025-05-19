"""設定管理モジュール"""
import os
import yaml
from dotenv import load_dotenv

def load_config(config_path):
    """設定ファイルの読み込み"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 環境変数の読み込み
    load_dotenv()
    
    # J-Quants API認証情報を環境変数から取得
    if 'jquants' in config:
        config['jquants']['mail_address'] = os.getenv('JQUANTS_API_MAIL_ADDRESS', 
                                                    config['jquants'].get('mail_address'))
        config['jquants']['password'] = os.getenv('JQUANTS_API_PASSWORD', 
                                                config['jquants'].get('password'))
        config['jquants']['refresh_token'] = os.getenv('JQUANTS_API_REFRESH_TOKEN', 
                                                    config['jquants'].get('refresh_token'))
    
    return config