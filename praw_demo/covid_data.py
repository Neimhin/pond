import pandas as pd
import numpy as np
import math
if __name__ == "__main__":
    df = pd.read_csv("owid-covid-data(1).csv",usecols=["iso_code","date","total_cases","new_cases","reproduction_rate",])
    df = df.fillna(0)
    df_world = df[df["iso_code"] == "OWID_WRL"]
    df_world["new_cases"] = np.log(df_world["new_cases"]+1)
    df_world.to_csv("world_covid_data.csv")