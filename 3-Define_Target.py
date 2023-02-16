import json
import gzip
import joblib
import tqdm
import numpy as np
import pandas as pd
from Variable import *

"""
大架構：
1. 輸入 contents.csv 與 correlations.csv
2. 確認所有 contents.csv 有哪些
3. 建立 one-hot encoding 格式的關聯性
"""

def load_content_and_correlation_data():
    
    raw_content_data = pd.read_csv(raw_content_data_path)
    raw_correlation_data = pd.read_csv(raw_correlation_data_path)
    return raw_content_data, raw_correlation_data

def find_contents(oneData):
    return {
        oneData["topic_id"]: tuple([unique_contents_id.index(i) for i in oneData["content_ids"]])
    }

if __name__ == "__main__":
    raw_content_data, raw_correlation_data = load_content_and_correlation_data()

    unique_contents_id = raw_content_data["id"].unique().tolist()

    raw_correlation_data["content_ids"] = raw_correlation_data.copy()["content_ids"].apply(lambda x: x.split(" "))
    
    delayed_func = tqdm.tqdm([joblib.delayed(find_contents)(one_data) for _, one_data in list(raw_correlation_data.iterrows())])
    parallel = joblib.Parallel(n_jobs = -1)
    result = {oneKey: oneValue for i in parallel(delayed_func) for oneKey, oneValue in i.items()}

    # 儲存關係結果為 JSON 檔案
    with gzip.open(os.path.join(main_path, "raw_data", "Correlations_with_oneHotEncoding_and_SinglePairs.json.gz"), "wt") as f:
        json.dump(result, f)