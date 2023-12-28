from pathlib import Path
import pandas as pd


data_path = Path("data")
data = []

for file in data_path.iterdir():
    if file.is_file() and file.suffix == '.csv':
        data.append(pd.read_csv(file))

out_path = data_path / "merged.csv"
pd.concat(data).to_csv(out_path)
