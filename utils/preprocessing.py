import pandas as pd

def load_and_split_data(file_path, start_date="2013-01-01"):
    df = pd.read_csv(file_path)
    df.index = pd.date_range(start=start_date, periods=len(df), freq="MS")
    df["R"] = df["I_allsky"] / df["I_clearsky"]
    train = df.iloc[:int(len(df)*0.8)].copy()
    test = df.iloc[int(len(df)*0.8):].copy()
    return train, test, df
