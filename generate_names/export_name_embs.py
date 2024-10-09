from scipy.stats import truncnorm
from sentence_transformers import SentenceTransformer
import pandas as pd
import collections
import numpy as np
import pickle
from Param import *

"""
    Truncated normal initialization
"""
def truncated_normal(size, threshold=0.02):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values


def run_gen_sent_embs():
    
    ids1 = pd.read_csv("../knowformer_data/" + dataset + "/ent_ids_1", sep="\t", header=None, names=["id", "uri"])
    uris_to_ids1 = dict(zip(ids1["uri"], ids1["id"]))

    ids2 = pd.read_csv("../knowformer_data/" + dataset + "/ent_ids_2", sep="\t", header=None, names=["id", "uri"])
    uris_to_ids2 = dict(zip(ids2["uri"], ids2["id"]))

    if dataset == "D_W_15K_V1":

        names1 = pd.read_excel('./entity_names/' + dataset + '/DBpedia_names.xlsx', index_col=0)
        ids_to_names1 = dict(zip(names1["e1"], names1["name"]))
            
        names2 = pd.read_excel('./entity_names/' + dataset + '/Wikidata_names.xlsx', index_col=0)
        ids_to_names2 = dict(zip(names2["e1"], names2["name"]))

    elif dataset == "BBC_DB":

        names1 = pd.read_excel('./entity_names/' + dataset + '/BBC_names.xlsx', index_col=0)
        ids_to_names1 = dict(zip(names1["e1"], names1["name"]))
            
        names2 = pd.read_excel('./entity_names/' + dataset + '/DBpedia_names.xlsx', index_col=0)
        ids_to_names2 = dict(zip(names2["e1"], names2["name"]))
    
    elif dataset == "D_W_15K_V2":

        names1 = pd.read_excel('./entity_names/' + dataset + '/DBpedia_names.xlsx', index_col=0)
        ids_to_names1 = dict(zip(names1["e1"], names1["name"]))
            
        names2 = pd.read_excel('./entity_names/' + dataset + '/Wikidata_names.xlsx', index_col=0)
        ids_to_names2 = dict(zip(names2["e1"], names2["name"]))
        
    elif dataset == "SRPRS_D_W_15K_V1":

        names1 = pd.read_excel('./entity_names/' + dataset + '/DBpedia_names.xlsx', index_col=0)
        ids_to_names1 = dict(zip(names1["e1"], names1["name"]))
            
        names2 = pd.read_excel('./entity_names/' + dataset + '/Wikidata_names.xlsx', index_col=0)
        ids_to_names2 = dict(zip(names2["e1"], names2["name"]))
    
    elif dataset == "SRPRS_D_W_15K_V2":

        names1 = pd.read_excel('./entity_names/' + dataset + '/DBpedia_names.xlsx', index_col=0)
        ids_to_names1 = dict(zip(names1["e1"], names1["name"]))
            
        names2 = pd.read_excel('./entity_names/' + dataset + '/Wikidata_names.xlsx', index_col=0)
        ids_to_names2 = dict(zip(names2["e1"], names2["name"]))
        
    elif dataset == "fr_en":

        names1 = pd.read_excel('./entity_names/' + dataset + '/fr_names.xlsx', index_col=0)
        ids_to_names1 = dict(zip(names1["e1"], names1["name"]))
            
        names2 = pd.read_excel('./entity_names/' + dataset + '/en_names.xlsx', index_col=0)
        ids_to_names2 = dict(zip(names2["e1"], names2["name"]))
        
    elif dataset == "ja_en":

        names1 = pd.read_excel('./entity_names/' + dataset + '/ja_names.xlsx', index_col=0)
        ids_to_names1 = dict(zip(names1["e1"], names1["name"]))
            
        names2 = pd.read_excel('./entity_names/' + dataset + '/en_names.xlsx', index_col=0)
        ids_to_names2 = dict(zip(names2["e1"], names2["name"]))
        
    elif dataset == "zh_en":

        names1 = pd.read_excel('./entity_names/' + dataset + '/zh_names.xlsx', index_col=0)
        ids_to_names1 = dict(zip(names1["e1"], names1["name"]))
            
        names2 = pd.read_excel('./entity_names/' + dataset + '/en_names.xlsx', index_col=0)
        ids_to_names2 = dict(zip(names2["e1"], names2["name"]))
        
    sentences = []
    randoms = []
    for id in ids_to_names1:
        name = ids_to_names1[id]
        if name != "no_value":
            sentences.append(name)
        else:
            randoms.append(id)
        
    for id in ids_to_names2:
        name = ids_to_names2[id]
        if name != "no_value":
            sentences.append(name)
        else:
            randoms.append(id)
            
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = model.encode(sentences)

    embs = {}
    index = 0
    for id in ids_to_names1:
        name = ids_to_names1[id]
        if name != "no_value":
            embs[id] = embeddings[index]
            index += 1
        else:
            embs[id] = truncated_normal(768, 0.02)
            
    for id in ids_to_names2:
        name = ids_to_names2[id]
        if name != "no_value":
            embs[id] = embeddings[index]
            index += 1
        else:
            embs[id] = truncated_normal(768, 0.02)

    # with open("unknown_ents_list.pickle", "wb") as file:
    #     pickle.dump(randoms, file)
        
    # with open("unknown_ents_list.pickle", "rb") as file:
    #     unknown_ents = pickle.load(file)
        
    name_embs = []
    for id in embs:
        name_embs.append(embs[id])
    name_embs = np.array(name_embs)
    np.save('./entity_names/' + dataset + '/' + dataset + '_name_embs.npy', name_embs)

    print("Name embeddings shape " + str(name_embs.shape))
    print("# unknown entities " + str(len(randoms)))