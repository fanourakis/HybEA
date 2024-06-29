from scipy.stats import truncnorm
from sentence_transformers import SentenceTransformer
import pandas as pd
import collections
import numpy as np
import pickle

def truncated_normal(size, threshold=0.02):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values

ids1 = pd.read_csv("../../../knowformer_data/"+ "D_W_15K_V1_no_validation_set" +"/ent_ids_1", sep="\t", header=None, names=["id", "uri"])
uris_to_ids1 = dict(zip(ids1["uri"], ids1["id"]))

ids2 = pd.read_csv("../../../knowformer_data//"+ "D_W_15K_V1_no_validation_set" +"/ent_ids_2", sep="\t", header=None, names=["id", "uri"])
uris_to_ids2 = dict(zip(ids2["uri"], ids2["id"]))
    
names1 = pd.read_excel('../../../knowformer_data//entity_names/DBpedia_from_D_W_15K_V1_alt_desc.xlsx', index_col=0)
ids_to_names1 = dict(zip(names1["e1"], names1["name"]))
    
names2 = pd.read_excel('../../../knowformer_data//entity_names/Wikidata_from_D_W_15K_V1_alt_desc.xlsx', index_col=0)
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

name_embs = {}
index = 0
for id in ids_to_names1:
    name = ids_to_names1[id]
    if name != "no_value":
        name_embs[id] = embeddings[index]
        index += 1
    else:
        name_embs[id] = truncated_normal(768, 0.02)
        
for id in ids_to_names2:
    name = ids_to_names2[id]
    if name != "no_value":
        name_embs[id] = embeddings[index]
        index += 1
    else:
        name_embs[id] = truncated_normal(768, 0.02)
print(name_embs.keys())
exit()            
with open("names_emb.pickle", "wb") as file:
    pickle.dump(name_embs, file)

with open("unknown_ents_list.pickle", "wb") as file:
    pickle.dump(randoms, file)

with open("names_emb.pickle", "rb") as file:
    embs = pickle.load(file)
    
with open("unknown_ents_list.pickle", "rb") as file:
    unknown_ents = pickle.load(file)
    
name_embs = []
for id in embs:
    name_embs.append(embs[id])
name_embs = np.array(name_embs)

print(name_embs)
print(name_embs.shape)
print(unknown_ents)
print(len(unknown_ents))