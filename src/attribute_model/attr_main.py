import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW
import torch.optim as optim
import random
import pickle

from attribute_model.utils.Read_data_func import read_data
from attribute_model.Param import *
from attribute_model.model.Basic_Bert_Unit_model import Basic_Bert_Unit_model
from attribute_model.utils.Batch_TrainData_Generator import Batch_TrainData_Generator
from attribute_model.train.train_func import train
import numpy as np
from sentence_transformers import SentenceTransformer

torch.cuda.set_device(0)
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))

def fixed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def run(new_pairs):

    #read data
    print("start load data....")
    ent_ill, train_ill, test_ill, valid_ill, \
    index2rel, index2entity, rel2index, entity2index, \
    ent2data, rel_triples_1, rel_triples_2 = read_data()
    
    # ent2data_types = ent2data[1]
    # entid2data = ent2data[0]
    # result = set()
    # for lst in ent2data_types.values():
    #     for x in lst:
    #         result.add(x)
    # print(len(result))

    # ent2data_types = ent2data[3]
    # entid2data = ent2data[2]
    # result = set()
    # for lst in ent2data_types.values():
    #     for x in lst:
    #         result.add(x)
    # print(len(result))
    # exit()
    
    print("---------------------------------------")

    print("all entity ILLs num:",len(ent_ill))
    print("rel num:",len(index2rel))
    print("ent num:",len(index2entity))
    print("triple1 num:",len(rel_triples_1))
    print("triple2 num:",len(rel_triples_2))


    #get train/test ILLs from file.
    print("get train/test ILLs from file \"sup_pairs\", \"ref_pairs\" !")
    print("train ILL num: {}, test ILL num: {}, valid ILL num: {}".format(len(train_ill), len(test_ill), len(valid_ill)))
    print("train ILL | test ILL | valid ILL:", len(set(train_ill) | set(test_ill) | set(valid_ill)))
    print("train ILL & test ILL & valid ILL:", len(set(train_ill) & set(test_ill) & set(valid_ill)))
    
    print("Adding new pairs")
    print("#new_pairs: " + str(len(new_pairs)))
    train_inter = set(train_ill).intersection(new_pairs)
    print("Train Inter with new pairs: " + str(len(train_inter)))
    test_inter = set(test_ill).intersection(new_pairs)
    print("Test Inter with new_pairs: " + str(len(test_inter)))
    valid_inter = set(valid_ill).intersection(new_pairs)
    print("Valid Inter with new_pairs: " + str(len(valid_inter)))
    
    for pair in new_pairs:
        train_ill.append(pair)
        
    print("train ILL num: {}, test ILL num: {}, valid ILL num: {} after adding new pairs".format(len(train_ill), len(test_ill), len(valid_ill)))

    sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    attr_dict = {}
    for i in range(0, 3, 2):
        entid2data = ent2data[i]
        for eid in entid2data:
            attr_dict[eid] = [] 
            for x in entid2data[eid]:
                attr_dict[eid].append(x)
    sentences = [value for value_list in attr_dict.values() for value in value_list]
    
    print("Generating sentence embeddings...")
    sent_emb = sbert_model.encode(sentences)
    print("Done")
    
    emb_dict = {}
    for index, sentence in enumerate(sentences):
        emb_dict[sentence] = sent_emb[index]
    
    Model = Basic_Bert_Unit_model(INPUT_SIZE_1, INPUT_SIZE_2, 768, emb_dict)

    Model.cuda(CUDA_NUM)

    # Criterion = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    Criterion = nn.MarginRankingLoss(MARGIN,size_average=True)
    Optimizer = AdamW(Model.parameters(),lr=LEARNING_RATE)

    ent1 = [e1 for e1,e2 in ent_ill]
    ent2 = [e2 for e1,e2 in ent_ill]

    #training data generator(can generate batch-size training data)
    Train_gene = Batch_TrainData_Generator(train_ill, ent1, ent2,index2entity,batch_size=TRAIN_BATCH_SIZE,neg_num=NEG_NUM)

    ent_ill, res_mat, res_mat_2, loss_list, hits_1_list = train(Model,Criterion,Optimizer,Train_gene,train_ill,test_ill, valid_ill, ent2data)
    
    return ent_ill, res_mat, res_mat_2, loss_list, hits_1_list

def run_attr_model(new_pairs):
    fixed(SEED_NUM)
    return run(new_pairs)