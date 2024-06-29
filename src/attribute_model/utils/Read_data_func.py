import pickle
from transformers import BertTokenizer
import logging
from attribute_model.Param import *
import pickle
import numpy as np
import re
import random
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
import pandas as pd
from attribute_model.Param import *

def read_structure_datas(data_path, reversed=False):
    def read_id2object(file_paths):
        id2object = {}
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                print('loading a (id2object)file...    ' + file_path)
                for line in f:
                    th = line.strip('\n').split('\t')
                    id2object[int(th[0])] = th[1]
        return id2object
    
    index2entity = read_id2object([data_path + "ent_ids_1",data_path + "ent_ids_2"])
    entity2index = {e:idx for idx,e in index2entity.items()}
    
    if reversed:
        index2entity1 = read_id2object([data_path + "ent_ids_1"])
        ents1 = {idx:e for idx,e in index2entity1.items()}
        
        index2entity2 = read_id2object([data_path + "ent_ids_2"])
        ents2 = {idx:e for idx,e in index2entity2.items()}
    else:
        index2entity1 = read_id2object([data_path + "ent_ids_1"])
        ents1 = {e:idx for idx,e in index2entity1.items()}
        
        index2entity2 = read_id2object([data_path + "ent_ids_2"])
        ents2 = {e:idx for idx,e in index2entity2.items()}
    
    return entity2index, ents1, ents2


def read_att_data(kg_att_file_name, entity_list, entity2index, dataset, which, add_name_as_attTriples = True):
    """
    load attribute triples file.
    """
    print("loading attribute triples file from: ",kg_att_file_name)
    att_data = []
    added_ents = set()
    with open(kg_att_file_name,"r",encoding="utf-8") as f:
        for line in f:
            e,a,l = line.rstrip('\n').split('\t',2)
            e = e.strip('<>')
            a = a.strip('<>')
            if "/property/" in a:
                a = a.split(r'/property/')[-1]
            else:
                a = a.split(r'/')[-1]
            l = l.rstrip('@zhenjadefr .')
            if len(l.rsplit('^^',1)) == 2:
                l,l_type = l.rsplit("^^")
            else:
                l_type = 'string'
            l = l.strip("\"")
            att_data.append((entity2index[e],a,l)) #(entity,attribute,value)
            added_ents.add(entity2index[e])
    if add_name_as_attTriples:
        for e in entity_list:
            if entity2index[e] not in added_ents:
                l, a = get_name(e, dataset) #entity name
                att_data.append((entity2index[e], a, l))
    return att_data



def read_data(data_path = DATA_PATH):
    def read_idtuple_file(file_path):
        print('loading a idtuple file...   ' + file_path)
        ret = []
        with open(file_path, "r", encoding='utf-8') as f:
            for line in f:
                th = line.strip('\n').split('\t')
                x = []
                for i in range(len(th)):
                    x.append(int(th[i]))
                ret.append(tuple(x))
        return ret
    def read_id2object(file_paths):
        id2object = {}
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                print('loading a (id2object)file...  ' + file_path)
                for line in f:
                    th = line.strip('\n').split('\t')
                    id2object[int(th[0])] = th[1]
        return id2object
    def read_idobj_tuple_file(file_path):
        print('loading a idx_obj file...   ' + file_path)
        ret = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                th = line.strip('\n').split('\t')
                ret.append( ( int(th[0]),th[1] ) )
        return ret

    print("load data from... :", data_path)
    #ent_index(ent_id)2entity / relation_index(rel_id)2relation
    index2entity = read_id2object([data_path + "ent_ids_1",data_path + "ent_ids_2"])
    index2rel = read_id2object([data_path + "rel_ids_1",data_path + "rel_ids_2"])
    entity2index = {e:idx for idx,e in index2entity.items()}
    rel2index = {r:idx for idx,r in index2rel.items()}

    #triples
    rel_triples_1 = read_idtuple_file(data_path + 'triples_1')
    rel_triples_2 = read_idtuple_file(data_path + 'triples_2')
    index_with_entity_1 = read_idobj_tuple_file(data_path + 'ent_ids_1')
    index_with_entity_2 = read_idobj_tuple_file(data_path + 'ent_ids_2')

    #ill
    train_ill = read_idtuple_file(data_path + 'sup_pairs')
    test_ill = read_idtuple_file(data_path + 'ref_pairs')
    valid_ill = read_idtuple_file(data_path + 'valid_pairs')
    ent_ill = []
    ent_ill.extend(train_ill)
    ent_ill.extend(test_ill)
    ent_ill.extend(valid_ill)

    #ent_idx
    entid_1 = [entid for entid,_ in index_with_entity_1]
    entid_2 = [entid for entid,_ in index_with_entity_2]
    entids = list(range(len(index2entity)))

    Tokenizer = None
    ent2desTokens = None
    
    entity2index, ents1, ents2 = read_structure_datas(DATA_PATH)
    ents = [ent for ent in entity2index.keys()]

    
    attribute_triple_1_file_path = DATA_PATH + 'attr_triples1'
    attribute_triple_2_file_path = DATA_PATH + 'attr_triples2'
    
    att_datas_1  = read_att_data(attribute_triple_1_file_path,
                                     ents1,entity2index, DATASET, "1", add_name_as_attTriples = False)

    att_datas_2  = read_att_data(attribute_triple_2_file_path,
                                     ents2,entity2index, DATASET, "2", add_name_as_attTriples = False)
    attr_types_1 = {}
    i = 0
    for quad in att_datas_1:
        
        if quad[1] in attr_types_1:
            continue
        
        attr_types_1[quad[1]] = i
        i = i + 1
        
    attr_types_2 = {}
    i = 0
    for quad in att_datas_2:
        
        if quad[1] in attr_types_2:
            continue
        
        attr_types_2[quad[1]] = i
        i = i + 1

    ent2data_1 = {}
    ent2data_types_1 = {}
    for quad in att_datas_1:

        if quad[0] not in ent2data_1:
            ent2data_1[quad[0]] = []  
            ent2data_types_1[quad[0]] = []
            
        ent2data_1[quad[0]].append(quad[2])
        ent2data_types_1[quad[0]].append(attr_types_1[quad[1]])
        
    ent2data_2 = {}
    ent2data_types_2 = {}
    for quad in att_datas_2:

        if quad[0] not in ent2data_2:
            ent2data_2[quad[0]] = []  
            ent2data_types_2[quad[0]] = []
            
        ent2data_2[quad[0]].append(quad[2])
        ent2data_types_2[quad[0]].append(attr_types_2[quad[1]])
        
    # print(len(ent2data_1))
    # print(len(ent2data_2))
    # print(len(ent2data_types_1))
    # print(len(ent2data_types_2))
    # print(ent2data_1)
    
    # max_length = max(len(lst) for lst in ent2data_types_1.values())
    # keys_with_max_length = [key for key, value in ent2data_types_1.items() if len(value) == max_length]
    # print(max_length)
    # print(keys_with_max_length)
    # max_length = max(len(lst) for lst in ent2data_types_2.values())
    # keys_with_max_length = [key for key, value in ent2data_types_2.items() if len(value) == max_length]
    # print(keys_with_max_length)
    # print(max_length)
    # exit()
    
    attr_data = [ent2data_1, ent2data_types_1, ent2data_2, ent2data_types_2]
    
    return ent_ill, train_ill, test_ill, valid_ill, index2rel, index2entity, rel2index, entity2index, attr_data, rel_triples_1, rel_triples_2