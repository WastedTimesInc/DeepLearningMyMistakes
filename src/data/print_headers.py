import os
import pandas as pd

for filename in os.listdir("./processed/labels/"):
        if filename.endswith(".parquet"):
            file_path = os.path.join("./processed/labels/", filename)
            try:
                df = pd.read_parquet(file_path, engine='auto')
                print(f"\n📄 File: {filename}")
                print("🧩 Columns:", list(df.columns))
            except Exception as e:
                print(f"❌ Could not read {filename}: {e}")

for filename in os.listdir("./processed/dataset/"):
        if filename.endswith(".parquet"):
            file_path = os.path.join("./processed/dataset/", filename)
            try:
                df = pd.read_parquet(file_path, engine='auto')
                print(f"\n📄 File: {filename}")
                print("🧩 Columns:", list(df.columns))
            except Exception as e:
                print(f"❌ Could not read {filename}: {e}")
