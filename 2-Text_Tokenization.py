"""
Reference: https://blog.csdn.net/u013250861/article/details/124535020 
Reference: https://stackoverflow.com/questions/52455774/googletrans-stopped-working-with-error-nonetype-object-has-no-attribute-group
"""

import os
import json
import zipfile
import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from googletrans import Translator

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
translator = Translator()

if __name__ == "__main__":
    main_path = r"D:\OneDrive - tmu.edu.tw\AI_Competition\Learning Equality - Curriculum Recommendations"
        
    # 輸入資料
    content_data = pd.read_csv(os.path.join(main_path, "raw_data", "content.csv"))
    topics_data = pd.read_csv(os.path.join(main_path, "raw_data", "topics.csv"))

    # 把各國語言翻譯成英文
    translator = Translator()
    topics_data["en_title"] = topics_data.copy().apply(lambda x: translator.translate(x["title"], dest = "en").text if x["language"] != "en" else x["title"], axis = 1)    
    
    translator = Translator()
    content_data["en_title"] = content_data.copy().apply(lambda x: translator.translate(x["title"], dest = "en").text if x["language"] != "en" else x["title"], axis = 1)

    # 將 Topics Data 中的 Title、Description 進行 Tokenize，取得 token_id, token_type_id 與 attention_mask
    title_token_result = {
        data_name : tokenizer(data["en_title"].tolist(), padding = True) for data_name, data in zip(["content_title", "topic_title"], [content_data, topics_data])
    }
    
    # 儲存文字切割結果為 JSON 檔案
    for one_file_name, one_data in title_token_result.items():
        with open(f"{one_file_name}.json", "w") as f:
            json.dump(dict(one_data), f)

    with zipfile.ZipFile(os.path.join(main_path, "raw_data", "token_data.zip"), "w") as f:
        for one_file_name in title_token_result.keys():
            f.write(f"{one_file_name}.json")
            os.remove(f"{one_file_name}.json")

    