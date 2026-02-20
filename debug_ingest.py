
import os
import pandas as pd
import subprocess

def debug_ingest():
    zip_path = "/tmp/smart_dl/DOTUSDT-1m-2023-03.zip"
    dl_dir = "/tmp/smart_dl"
    base_sym = "DOTUSDT"
    y = 2023
    m = 3
    
    print(f"Checking {zip_path}...")
    if not os.path.exists(zip_path):
        print("Zip not found")
        return

    print("Unzipping...")
    subprocess.run(["unzip", "-o", zip_path, "-d", dl_dir], check=True)
    
    csv_name = f"{base_sym}-1m-{y}-{m:02d}.csv"
    csv_path = os.path.join(dl_dir, csv_name)
    print(f"Looking for {csv_path}...")
    
    if os.path.exists(csv_path):
        print("CSV found. Reading...")
        try:
            df = pd.read_csv(csv_path, header=None, 
                             names=['open_time','open','high','low','close','volume','close_time','qav','num_trades','taker_base','taker_quote','ignore'])
            
            print(f"Raw shape: {df.shape}")
            print(df.head())
            
            df = df[['open_time','open','high','low','close','volume']]
            df.rename(columns={'open_time': 'timestamp'}, inplace=True)
            df['timestamp'] = df['timestamp'] / 1000
            df['timestamp'] = df['timestamp'].astype('int64')
            df.set_index(pd.to_datetime(df['timestamp'], unit='s'), inplace=True)
            df.index.name = 'time'
            
            print("Processed:")
            print(df.head())
            print(df.index.dtype)
            
            # Simulated Upsert
            print("Simulating Arrow write...")
            import pyarrow.feather as feather
            out_path = "debug_arrow.feather"
            df.reset_index().to_feather(out_path, compression='uncompressed')
            print(f"Written to {out_path}")
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("CSV not found after unzip")

if __name__ == "__main__":
    debug_ingest()
