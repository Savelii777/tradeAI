import pandas as pd

df = pd.read_csv('data/candles/NEAR_USDT_USDT_5m.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

start_time = pd.to_datetime('2025-12-26 07:15:00')
end_time = pd.to_datetime('2025-12-26 10:05:00')

mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
subset = df.loc[mask]

print(subset[['timestamp', 'low', 'high', 'close']].to_string())

min_low = subset['low'].min()
print(f"\nLowest Price in window: {min_low}")
