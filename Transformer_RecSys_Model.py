import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class Transformer_RecSys_Model(nn.Module):
    def __init__(self, transformer_model_name):
        super(Transformer_RecSys_Model, self).__init__()
        self.topicTransformerModel = AutoModel.from_pretrained(transformer_model_name)
        self.contentTransformerModel = AutoModel.from_pretrained(transformer_model_name)
        self.decoderLayerNodes = [*[768*2//i for i in range(1, 11)], 1]
        self.decoder = nn.Sequential(
            *[nn.Linear(self.decoderLayerNodes[nodeIndex], self.decoderLayerNodes[nodeIndex+1]) for nodeIndex in range(len(self.decoderLayerNodes)-1)],
        )
        return 

    def forward(self, topicData, contentData):

        topicData = self.topicTransformerModel(input_ids = topicData[:, :, 0], 
                                               attention_mask = topicData[:, :, 1])["pooler_output"]
        contentData = self.contentTransformerModel(input_ids = contentData[:, :, 0],
                                                   attention_mask = contentData[:, :, 1])["pooler_output"]
        
        interactionData = torch.cat([topicData, contentData], dim = -1)
        return torch.sigmoid(self.decoder(interactionData))[:, 0]