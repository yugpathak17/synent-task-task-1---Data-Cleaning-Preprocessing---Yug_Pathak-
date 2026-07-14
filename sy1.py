from pathlib import Path

import pandas as pd

script_dir = Path(__file__).resolve().parent
dataset_path = script_dir / "codealpha projects" / "tested.csv"
output_path = script_dir / "titanic_cleaned.csv"

df = pd.read_csv(dataset_path)

df = df.drop_duplicates()

for col in df.columns:
    if df[col].isnull().sum() == 0:
        continue

    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        mode = df[col].mode(dropna=True)
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])
        else:
            df[col] = df[col].fillna("")

numeric_cols = ["Age", "Fare", "Survived", "Pclass", "SibSp", "Parch"]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

for col in ["Survived", "Pclass", "SibSp", "Parch"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isnull().sum() == 0:
            df[col] = df[col].astype(int)

df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

df.to_csv(output_path, index=False)

print(df.head())
print(df.info())