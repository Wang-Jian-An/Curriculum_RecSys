import json
import zipfile
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

def one_hot_encoding_format_for_correlation(target_list):

    return_list = init_list.copy()
    for i in target_list:
        return_list[unique_contents_id.index(i)] = 1
    return return_list

if __name__ == "__main__":
    raw_content_data, raw_correlation_data = load_content_and_correlation_data()

    unique_contents_id = raw_content_data["id"].unique().tolist()

    raw_correlation_data["content_ids"] = raw_correlation_data.copy()["content_ids"].apply(lambda x: x.split(" "))

    init_list = [0] * unique_contents_id.__len__()
    
    result = {
        one_data["topic_id"]: one_hot_encoding_format_for_correlation(one_data["content_ids"]) for _, one_data in raw_correlation_data.iterrows()
    }

    # 儲存關係結果為 JSON 檔案
    with open("Correlations_with_oneHotEncoding.json", "w") as f:
        json.dump(dict(result), f)

    with zipfile.ZipFile(os.path.join(main_path, "raw_data", "Correlations_with_oneHotEncoding.zip"), "w") as f:
        f.write("Correlations_with_oneHotEncoding.json")
        os.remove("Correlations_with_oneHotEncoding.json")