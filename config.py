from database.entities.crypto import Crypto
# from database.network.coinapi.ohlcv import OHLCVDownloader

# --- Database ---
# supported_cryptos = { ... } # Removed as it was only used by the old downloader

ohlcv_dataset_period_id = '1h'
ohlcv_history_filepath = 'database/storage/downloads/ohlcv/{}.csv'
# gtrends_history_filepath = 'database/storage/downloads/gtrends/{}.csv' # Removed
dataset_save_filepath = 'database/storage/datasets/{}.csv'
# all_features = [...] # Removed as it is unused

regression_features = [
    'open_log_returns', 'high_log_returns', 'low_log_returns',
    'close_log_returns', 'volume_log_returns', 'hour',
    'macd_signal_diffs', 'stoch', 'aroon_up', 'aroon_down', 'rsi', 'adx', 'cci',
    'close_dema', 'close_vwap', 'bband_up_close', 'close_bband_down', 'adl_diffs', 'obv_diffs'
]

# --- Model ---
checkpoint_dir = 'database/storage/checkpoints/'


