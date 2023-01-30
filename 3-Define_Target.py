import compress_json
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

if __name__ == "__main__":
    raw_content_data, raw_correlation_data = load_content_and_correlation_data()

    unique_contents_id = raw_content_data["id"].unique().tolist()

    raw_correlation_data["content_ids"] = raw_correlation_data.copy()["content_ids"].apply(lambda x: x.split(" "))
    
    # result = {
    #     one_data["topic_id"]: tuple([1 if i in one_data["content_ids"] else 0 for i in unique_contents_id]) for _, one_data in list(raw_correlation_data.iterrows())[:1000]
    # }
    result = {
        one_data["topic_id"]: {
            i: 1 if i in one_data["content_ids"] else 0 for i in unique_contents_id
        } for _, one_data in list(raw_correlation_data.iterrows())[:100]
    }

    # 儲存關係結果為 JSON 檔案
    compress_json.dump(result, f"{main_path}/raw_data/Correlations_with_oneHotEncoding_and_SinglePairs_for_development.json.gz")