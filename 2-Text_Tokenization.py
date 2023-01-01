"""
Reference: https://blog.csdn.net/u013250861/article/details/124535020
"""

import os
import numpy as np
import pandas as pd

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

if __name__ == "__main__":
    main_path = "./"
        
    # 輸入資料
    content_data = pd.read_csv(os.path.join(main_path, "raw_data", "content.csv"))
    topics_data = pd.read_csv(os.path.join(main_path, "raw_data", "topics.csv"))

    # 將 Topics Data 中的 Title、Description 進行 Tokenize，取得 token_id, token_type_id 與 attention_mask
    topics_title_token_result = tokenizer(topics_data["title"].tolist()[:2], padding = True)
    print(topics_title_token_result)