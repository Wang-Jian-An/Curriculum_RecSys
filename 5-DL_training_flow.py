import gc
import numpy as np
import compress_json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from Variable import *
from Transformer_RecSys_Model import Transformer_RecSys_Model

rawDataPath = f"{main_path}/raw_data/SplitData_for_Correlation_for_development.json.gz"
topicDataPath = os.path.join(main_path, "raw_data", "tokenizered_topics.xlsx") 
contentDataPath = os.path.join(main_path, "raw_data", "tokenizered_content.xlsx")
device = "cuda" if torch.cuda.is_available else "cpu"
epochs = 10

def combine_input_ids_and_attention_mask_in_one_list(oneInputIds, oneAttentionMask):
    return [
        [i, j] for i, j in zip(oneInputIds, oneAttentionMask)
    ]

def generate_three_obj_for_train_vali_test_data(data):
    data = pd.DataFrame(data)
    return [
        data.query("Data-ID == 1 and Split == @oneSplit") for oneSplit in ["train", "vali", "test"]
    ]

def generate_TrainDataLoader(oneData: pd.DataFrame, batch_size = 128):

    topicInputIdsAttentionMask = [rawTopics[oneTopic] for oneTopic in oneData["Topic"]]
    contentInputIdsAttentionMask = [rawContent[oneContent] for oneContent in oneData["Content"]]
    correlation = oneData["Correlation"].tolist()
    tensorDataset = TensorDataset(
        torch.LongTensor(topicInputIdsAttentionMask),
        torch.LongTensor(contentInputIdsAttentionMask),
        torch.FloatTensor(correlation)
    )
    return DataLoader(tensorDataset, batch_size = batch_size, shuffle = True)

def generate_PredictionDataLoader(oneData: pd.DataFrame, batch_size):

    topicInputIdsAttentionMask = [rawTopics[oneTopic] for oneTopic in oneData["Topic"]]
    contentInputIdsAttentionMask = [rawContent[oneContent] for oneContent in oneData["Content"]]
    tensorDataset = TensorDataset(
        torch.LongTensor(topicInputIdsAttentionMask),
        torch.LongTensor(contentInputIdsAttentionMask)
    )
    return DataLoader(tensorDataset, batch_size = batch_size, shuffle = False)

def DL_model_training():
    trainLossDict = list()
    for oneEpoch in range(epochs):
        LossMean = list()
        for oneTopic, oneContent, oneCorrelation in trainDataLoader:
            optimizer.zero_grad()
            yhat = model(topicData = oneTopic.to(device), contentData = oneContent.to(device))
            loss = loss_func(yhat, oneCorrelation.to(device))
            loss.backward()
            optimizer.step()
            LossMean.append(loss.cpu().item())
        trainLossDict.append({
            "Epoch": oneEpoch,
            "Loss": np.mean(LossMean)
        })
    return trainLossDict

def DL_model_prediction(onePredictionDataLoader):

    with torch.no_grad():
        yhat = [i for oneTopic, oneContent in onePredictionDataLoader\
            for i in model(topicData = oneTopic.to(device), contentData = oneContent.to(device)).cpu().flatten().tolist()]
    return yhat



if __name__ == "__main__":

    # Load Data
    rawTopics = pd.read_excel(topicDataPath)
    rawContent = pd.read_excel(contentDataPath)
    rawData = pd.DataFrame(compress_json.load(rawDataPath))

    # Combine input_ids and attention mask to one list
    rawTopics["input_ids_attention_mask"] = rawTopics.copy().apply(lambda x: combine_input_ids_and_attention_mask_in_one_list(oneInputIds = x["input_ids"], oneAttentionMask = x["attention_mask"]), axis = 1)
    rawContent["input_ids_attention_mask"] = rawContent.copy().apply(lambda x: combine_input_ids_and_attention_mask_in_one_list(oneInputIds = x["input_ids"], oneAttentionMask = x["attention_mask"]), axis = 1)

    # convert the type of rawTopics and rawContent to json
    rawTopics = rawTopics.set_index("id")["input_ids_attention_mask"].to_dict()
    rawContent = rawContent.set_index("id")["input_ids_attention_mask"].to_dict()

    # Generate train, validation and test data
    trainData, valiData, testData = generate_three_obj_for_train_vali_test_data(data = rawData)
    del rawData
    gc.collect()

    # Generate DataLoader
    trainDataLoader = generate_TrainDataLoader(oneData = trainData, batch_size = 128)
    valiDataLoader, testDataLoader = [
        generate_PredictionDataLoader(oneData = oneData, batch_size = rawContent.shape[0]) for oneData in [valiData, testData]
    ]

    # Model Training
    model = Transformer_RecSys_Model(transformer_model_name = "bert-base-cased").to(device)
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3) 

    # Model Evaluation（先預測後計算評估指標）
    testYhat = DL_model_prediction(onePredictionDataLoader=testDataLoader)
