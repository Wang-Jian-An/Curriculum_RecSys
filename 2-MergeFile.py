import os
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # 設定主要路徑
    main_path = "./"

    # 輸入資料
    correlations_data = pd.read_csv(os.path.join(main_path, "raw_data", "correlations.csv"))
    content_data = pd.read_csv(os.path.join(main_path, "raw_data", "content.csv"))
    topics_data = pd.read_csv(os.path.join(main_path, "raw_data", "topics.csv"))

    # 分別把 Content Data 與 Topic Data 的欄位名稱全都加上資料表名稱
    content_data = content_data.rename(columns = {i: f"content_{i}" for i in content_data.columns.tolist()})
    topics_data = topics_data.rename(columns = {i: f"topics_{i}" for i in topics_data.columns.tolist()})

    # 合併 Correlations Data 與 Content Data
    mergeData = pd.merge(left = correlations_data, 
                         right = content_data,
                         how = "left",
                         left_on = "content_ids",
                         right_on = "content_id")

    # 合併 Correlations Data 與 Topics Data
    mergeData = pd.merge(left = mergeData,
                         right = topics_data,
                         how = "left",
                         left_on = "topic_id",
                         right_on = "topics_id" )

    # Store mergeData
    mergeData.to_pickle(os.path.join(main_path, "raw_data", "mergeData.gzip"), "gzip")