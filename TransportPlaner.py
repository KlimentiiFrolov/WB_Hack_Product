
import pandas as pd
import numpy as np
time_car = 120
max_capacity = 10
def calculate_vehicles(raw:pd.DataFrame,predict:pd.DataFrame)->pd.DataFrame:
    pred_map = predict.set_index("id")["y_pred"]
    raw["cars_predict"] = np.ceil(raw["id"].map(pred_map).astype(int) / max_capacity)
    raw["timestamp"]+=2
    return raw

#calculate_vehicles(pd.read_csv("submission_team.csv"))

print(pd.read_parquet("train_team_track.parquet"))