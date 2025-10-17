import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    print(f"dataset shape: {df.shape}")
    return df
