import gc
import math
import random
import numpy as np
import compress_json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from Variable import *
from Transformer_RecSys_Model import Transformer_RecSys_Model

rawDataPath = os.path.join(main_path, "raw_data", "SplitData_for_Correlation.json.gz")
topicDataPath = os.path.join(main_path, "raw_data", "tokenizered_topics.xlsx") 
contentDataPath = os.path.join(main_path, "raw_data", "tokenizered_content.xlsx")
# device = "cuda" if torch.cuda.is_available else "cpu"
device = "cpu"
epochs = 10

def combine_input_ids_and_attention_mask_in_one_list(oneInputIds, oneAttentionMask):
    return [
        [i, j] for i, j in zip(oneInputIds, oneAttentionMask)
    ]

def generate_three_obj_for_train_vali_test_data(data):
    print(data)
    data = [data.query("`Data-ID` == 1 and Split == @oneSplit") for oneSplit in ["train", "vali", "test"]]
    return [
        [
            {
                "Topic": oneRowData["Topic"], 
                "Content": i,
                "Correlation": 1 if i in oneRowData["Correlation"] else 0 
            } for _, oneRowData in oneData.iterrows() for i in rawContent.keys()
        ] for oneData in data
    ]

def build_batch(Data: list, batch_size: int = 128):

    """
    Input: a list including many dicts which includes topic id, content id, and correlation
    Output: each sublist for one batch
    """
    assert Data.__len__() >= batch_size, "The batch size is not bigger than the length of Data"
    
    batchResult = list()
    for oneBatch in range(math.ceil(Data.__len__() / batch_size)):
        batchResult.append( Data[batch_size * oneBatch:batch_size * (oneBatch+1)] )
    return batchResult

# def generate_TrainDataLoader(oneData, batch_size = 128):
#     oneData = pd.DataFrame(oneData).to_dict("list")
#     topicInputIdsAttentionMask = [rawTopics[oneTopic] for oneTopic in oneData["Topic"]]
#     print(topicInputIdsAttentionMask[0])
#     contentInputIdsAttentionMask = [rawContent[oneContent] for oneContent in oneData["Content"]]
#     tensorDataset = TensorDataset(
#         torch.LongTensor(topicInputIdsAttentionMask),
#         torch.LongTensor(contentInputIdsAttentionMask),
#         torch.FloatTensor(oneData["Correlation"])
#     )
#     return DataLoader(tensorDataset, batch_size = batch_size, shuffle = True) 

def generate_PredictionDataLoader(oneData: pd.DataFrame, batch_size):

    topicInputIdsAttentionMask = [rawTopics[oneTopic] for oneTopic in oneData["Topic"]]
    contentInputIdsAttentionMask = [rawContent[oneContent] for oneContent in oneData["Content"]]
    tensorDataset = TensorDataset(
        torch.LongTensor(topicInputIdsAttentionMask),
        torch.LongTensor(contentInputIdsAttentionMask)
    )
    return DataLoader(tensorDataset, batch_size = batch_size, shuffle = False)

def transform_topic_content_and_correlation_to_tensor(oneData, mode = "train"):

    """
    Parameters
    -------------
    oneData: a list including topic id, content id, and correlation

    Return
    -------------
    oneTopic: 
    oneContent:
    oneCorrelation: 
    """

    assert mode in ["train", "predict"], "The mode must includes train or predict. "

    topics = [rawTopics[oneBatch["Topic"]] for oneBatch in oneData]
    contents = [rawContent[oneBatch["Content"]] for oneBatch in oneData]
    correlations = [oneBatch["Correlation"] for oneBatch in oneData]

    if mode == "train":
        return [
            torch.LongTensor(topics),
            torch.LongTensor(contents),
            torch.FloatTensor(correlations)
        ]
    else:
        return [
            torch.LongTensor(topics),
            torch.LongTensor(contents)
        ]

def DL_model_training(trainDataLoader):

    """
    Parameters
    -------------
    trainDataLoader: a list including topic id, content id, and correlation

    Return
    -------------
    trainLossDict: the situation for model training
    """

    trainLossDict = list()
    for oneEpoch in range(epochs):
        LossMean = list()
        for oneBatchData in trainDataLoader:
            oneTopic, oneContent, oneCorrelation = transform_topic_content_and_correlation_to_tensor(oneData= oneBatchData)
            print(oneTopic.size(), oneContent.size(), oneCorrelation.size())
            optimizer.zero_grad()
            yhat = model(topicData = oneTopic.to(device), contentData = oneContent.to(device))
            loss = loss_func(yhat, oneCorrelation.to(device))
            loss.backward()
            optimizer.step()
            LossMean.append(loss.cpu().item())
            break
        trainLossDict.append({
            "Epoch": oneEpoch,
            "Loss": np.mean(LossMean)
        })
        break
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
    rawTopics["input_ids_attention_mask"] = rawTopics.copy().apply(lambda x: combine_input_ids_and_attention_mask_in_one_list(oneInputIds = eval(x["input_ids"]), oneAttentionMask = eval(x["attention_mask"])), axis = 1)
    rawContent["input_ids_attention_mask"] = rawContent.copy().apply(lambda x: combine_input_ids_and_attention_mask_in_one_list(oneInputIds = eval(x["input_ids"]), oneAttentionMask = eval(x["attention_mask"])), axis = 1)

    # convert the type of rawTopics and rawContent to json
    rawTopics = rawTopics.set_index("id")["input_ids_attention_mask"].to_dict()
    rawContent = rawContent.set_index("id")["input_ids_attention_mask"].to_dict()

    # Generate train, validation and test data
    trainData, valiData, testData = generate_three_obj_for_train_vali_test_data(data = rawData)
    del rawData
    gc.collect()

    # Shuffle Data
    random.shuffle(trainData)

    # Build Batch
    trainDataLoader = build_batch(trainData)
    valiDataLoader = build_batch(valiData, batch_size = list(rawContent.keys()).__len__() )
    testDataLoader = build_batch(testData, batch_size = list(rawContent.keys()).__len__())

    # # Generate DataLoader
    # trainDataLoader = generate_TrainDataLoader(oneData = trainData, batch_size = 128)
    # valiDataLoader, testDataLoader = [
    #     generate_PredictionDataLoader(oneData = oneData, batch_size = rawContent.shape[0]) for oneData in [valiData, testData]
    # ]

    # Model Training
    model = Transformer_RecSys_Model(transformer_model_name = "bert-base-cased").to(device)
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3) 
    trainLossDict = DL_model_training(trainDataLoader  =  trainDataLoader)

    # # Model Evaluation（先預測後計算評估指標）
    # testYhat = DL_model_prediction(onePredictionDataLoader=testDataLoader)
