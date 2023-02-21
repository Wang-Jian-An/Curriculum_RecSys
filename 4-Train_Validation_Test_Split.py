"""
Hints: 由於是用已知 topic 預測未知 topic，所以是抽 topic
"""

import os
import tqdm
import compress_json
import gzip
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from Variable import *

jsonFilePath = os.path.join(main_path, "raw_data", "Correlations_with_oneHotEncoding_and_SinglePairs.json.gz")

if __name__ == "__main__":

    # rawCorrelation = compress_json.load(jsonFilePath)
    # 儲存關係結果為 JSON 檔案
    with gzip.open(os.path.join(main_path, "raw_data", jsonFilePath), "rt") as f:
        rawCorrelation  = json.load(f)

    unique_topics = list(rawCorrelation.keys())

    trainTopics, testTopics = train_test_split(unique_topics, test_size = 0.2, shuffle = True)
    trainTopics, valiTopics = train_test_split(trainTopics, test_size = 0.25, shuffle = True)
    totalResult = [
        {
            "Data-ID": 1,
            "Split": splitName,
            "Topic": oneTopic,
            "Correlation": rawCorrelation[oneTopic]
        } for splitName, splitObj in zip(["train", "vali", "test"], [trainTopics, valiTopics, testTopics]) for oneTopic in tqdm.tqdm(splitObj) 
    ]
    compress_json.dump(pd.DataFrame(totalResult).to_dict("records"), os.path.join(main_path, "raw_data", "SplitData_for_Correlation_for_developments.json.gz"))