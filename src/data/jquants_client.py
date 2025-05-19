"""J-Quants APIクライアント"""
import jquantsapi
from datetime import datetime, timedelta

class JQuantsClient:
    """J-Quants APIのラッパークラス"""
    def __init__(self, mail_address=None, password=None, refresh_token=None):
        self.client = jquantsapi.Client(
            mail_address=mail_address,
            password=password,
            refresh_token=refresh_token
        )
    
    def get_stock_data(self, start_dt, end_dt, save_path=None):
        """株価データの取得"""
        stock_price = self.client.get_price_range(start_dt=start_dt, end_dt=end_dt)
        
        if save_path:
            stock_price.to_csv(save_path, compression='gzip', index=False)
        
        return stock_price
        
    def get_topix500_tickers(self, save_path=None):
        """TOPIX500銘柄の取得"""
        stock_list = self.client.get_listed_info()
        
        # TOPIX500銘柄の抽出
        categories = ['TOPIX Mid400', 'TOPIX Large70', 'TOPIX Core30']
        topix500 = stock_list[stock_list['ScaleCategory'].isin(categories)]['Code'].unique()
        topix500 = topix500.astype(str)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write('\n'.join(topix500))
        
        return topix500