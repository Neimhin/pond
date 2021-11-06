import pandas as pd
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv("ie.csv")
    print(df.head())
    df["coords"] = df["lat"].astype(str) + " " + df["lng"].astype(str)
    for index, row in df.iterrows():
        print(row["city"],row["coords"])
    exit()
    coords = df["coords"].values
    f = open("coords_ie.txt","w",encoding="utf8")
    for coord in coords:
        f.write(f"{coord}\n")
    f.close()