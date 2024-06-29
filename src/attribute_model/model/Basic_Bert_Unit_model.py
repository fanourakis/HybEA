# from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from scipy.stats import truncnorm
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

torch.cuda.init()

class AttentionLayer(nn.Module):
    def __init__(self, input_size, attention_size):
        super(AttentionLayer, self).__init__()
        self.input_size = input_size
        self.attention_size = attention_size
        self.emb = nn.Embedding(self.input_size + 1, self.attention_size, padding_idx=(self.input_size))
        self.attribute_embeddings = nn.Linear(self.attention_size, self.attention_size)
        self.A = nn.Linear(self.attention_size, self.attention_size)
        self.dropout = nn.Dropout(0.1) 
        
    def initialize_weights(self):
        nn.init.normal_(self.attribute_embeddings.weight, mean=0, std=1)
        nn.init.normal_(self.A.weight, mean=0, std=1)

    def forward(self, padded_alpha_i, output):
        emb = self.emb(padded_alpha_i)
        alpha_i_embedding = self.attribute_embeddings(emb)
        A = self.A(emb)

        # Scaled ?
        scaled = True
        if scaled:
            scores = torch.bmm(A, alpha_i_embedding.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.attention_size, dtype=torch.float32))
        else:
            scores = torch.bmm(A, alpha_i_embedding.transpose(-2, -1))
        
        attention_weights = F.softmax(scores, dim=-1)
        # weighted_projection = torch.bmm(attention_weights, alpha_i_embedding)
        # etype = torch.sum(weighted_projection, dim=1)
        weighted_value = torch.bmm(attention_weights, output)
        evalue = torch.sum(weighted_value, dim=1)
        return evalue
        # return torch.cat([evalue, etype], dim=1)


class Basic_Bert_Unit_model(nn.Module):
    def __init__(self, input_size1, input_size2, attention_size, emb_dict):
        super(Basic_Bert_Unit_model, self).__init__()
        self.input_size1 = input_size1
        self.input_size2 = input_size2
        self.attention_size = attention_size
        self.attention_layer_1 = AttentionLayer(self.input_size1, attention_size)
        self.attention_layer_2 = AttentionLayer(self.input_size2, attention_size)
        self.out_linear_layer = nn.Linear(768, 768)
        self.dropout = nn.Dropout(p = 0.1)
        self.emb_dict = emb_dict

    def initialize_weights(self):
        nn.init.normal_(self.out_linear_layer.weight)
        
    def value_embs(self, alpha_i_batched,batch_sentences, input_size, llayer, padded_alpha_i):
        value_embs = []
        index = 0
        random_list = []
        for lst in padded_alpha_i:
            temp = []
            for elem in lst:
                if elem == input_size:
                    temp.append(np.zeros(768, dtype=np.float32))
                else:
                    temp.append(self.emb_dict[batch_sentences[index]])
                    index += 1
            value_embs.append(np.array(temp))
        value_embs = torch.tensor(np.array(value_embs)).cuda(1)
        # output = self.dropout(value_embs)
        # output_values = llayer(value_embs)
        # temp = torch.sum(output_values, dim=1)
        return value_embs
    
    def truncated_normal(size, threshold=0.02):
        values = truncnorm.rvs(-threshold, threshold, size=size)
        return values
    
    def forward(self, alpha_i_batched, batch_sentences, padded_alpha_i, which):
        if which == "1":
            output_values = self.value_embs(alpha_i_batched, batch_sentences, self.input_size1, self.out_linear_layer, padded_alpha_i)
            output = self.attention_layer_1(padded_alpha_i, output_values)
            return output
        elif which == "2":
            output_values = self.value_embs(alpha_i_batched, batch_sentences, self.input_size2, self.out_linear_layer, padded_alpha_i)
            output = self.attention_layer_2(padded_alpha_i, output_values)
            return output