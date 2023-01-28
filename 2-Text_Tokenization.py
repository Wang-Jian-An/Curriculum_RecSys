"""
Reference: https://blog.csdn.net/u013250861/article/details/124535020 
Reference: https://stackoverflow.com/questions/52455774/googletrans-stopped-working-with-error-nonetype-object-has-no-attribute-group
"""

import os
import gc
import json
import time
import zipfile
import numpy as np
import pandas as pd
import tqdm
tqdm.tqdm.pandas()
import compress_json
from transformers import AutoTokenizer
from googletrans import Translator
from Variable import *

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
translator = Translator()

if __name__ == "__main__":
        
    # 輸入資料
    content_data = pd.read_csv(os.path.join(main_path, "raw_data", "content.csv"))
    topics_data = pd.read_csv(os.path.join(main_path, "raw_data", "topics.csv"))

    # 將遺失值填補為 None
    content_data["title"] = content_data.copy()["title"].fillna("None")
    topics_data["title"] = topics_data.copy()["title"].fillna("None")

    # 把各國語言翻譯成英文
    translator = Translator(user_agent = r"Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Mobile Safari/537.36")
    topics_data["en_title"] = topics_data.copy().progress_apply(lambda x: translator.translate(x["title"], dest = "en").text if x["language"] != "en" else x["title"], axis = 1)    
    del translator
    gc.collect()
    time.sleep(10)

    topicsToken = {
        oneKey: [tuple(i) for i in oneValue] for oneKey, oneValue in tokenizer(topics_data["en_title"].tolist(), padding = True).items()
    } 
    topics_data = pd.concat([
        topics_data["id"],
        pd.DataFrame(topicsToken)
    ], axis = 1)
    topics_data.to_excel(os.path.join(main_path, "raw_data", "tokenizered_topics.xlsx"), index = None)

    translator = Translator(user_agent = r"Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Mobile Safari/537.36")
    content_data_en_title_part_one = content_data.iloc[:(content_data.shape[0]//2), :].apply(lambda x: translator.translate(x["title"], dest = "en").text if x["language"] != "en" else x["title"], axis = 1).tolist()
    del translator
    gc.collect()
    time.sleep(10)

    translator = Translator(user_agent = "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Mobile Safari/537.36")
    content_data_en_title_part_two = content_data.iloc[(content_data.shape[0]//2):, :].apply(lambda x: translator.translate(x["title"], dest = "en").text if x["language"] != "en" else x["title"], axis = 1).tolist()
    content_data = content_data.copy()
    content_data["en_title"] = content_data_en_title_part_one+content_data_en_title_part_two

    # 將 Topics Data 中的 Title、Description 進行 Tokenize，取得 token_id, token_type_id 與 attention_mask 
    contentToken = {
        oneKey: [tuple(i) for i in oneValue] for oneKey, oneValue in tokenizer(content_data["en_title"].tolist(), padding = True).items()
    } 
    content_data = pd.concat([
        content_data["id"],
        pd.DataFrame(contentToken)
    ], axis = 1)
    
    # 儲存文字切割結果為 EXCEL 檔案
    content_data.to_excel(os.path.join(main_path, "raw_data", "tokenizered_content.xlsx"), index = None)