import numpy as np
import pandas as pd
from database.preprocessing.preprocessing import DatasetPreprocessing


class OHLCVPreprocessing(DatasetPreprocessing):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _preprocess_ohlcv_columns(ohlcv_df: pd.DataFrame):
        # Rename columns to match expected format
        rename_map = {
            'open_time_dt': 'date',
            'trade_count': 'trades'
        }
        ohlcv_df.rename(columns=rename_map, inplace=True)
        
        # Ensure date format is correct (string "YYYY-MM-DD HH:MM:SS")
        if 'date' in ohlcv_df.columns:
             # Check if it's already string or datetime
            if pd.api.types.is_datetime64_any_dtype(ohlcv_df['date']):
                ohlcv_df['date'] = ohlcv_df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                 # Assume it might be string, just ensure format if needed or leave as is if it looks right
                 # The old code did: date.split('.')[0].replace('T', ' ')
                 # The new downloader outputs standard ISO format or similar.
                 pass

        # Extract hour
        if 'date' in ohlcv_df.columns:
             ohlcv_df['hour'] = ohlcv_df['date'].apply(lambda date: int(str(date).split(' ')[1].split(':')[0]))
             
        return ohlcv_df

    @staticmethod
    def _append_ohlcv_log_returns_to_df(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        for col in ['open', 'high', 'low', 'close', 'volume', 'trades']:
            if col in ohlcv_df.columns:
                ohlcv_df[f'{col}_log_returns'] = np.log(ohlcv_df[col]).diff()
        return ohlcv_df

    def preprocess(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        ohlcv_df = self._preprocess_ohlcv_columns(ohlcv_df=ohlcv_df)
        ohlcv_df = self._append_ohlcv_log_returns_to_df(ohlcv_df=ohlcv_df)
        return ohlcv_df
