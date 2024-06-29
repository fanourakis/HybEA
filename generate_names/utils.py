import re
from langdetect import detect
import os
import io
import numpy as np
import pandas as pd
from Param import *

"""
    Language detection
"""
def lang_detect(s):
    try:
        lang = detect(s)
    except:
        lang = "Other"

    return lang


def to_delete(lang):
    return lang in ["ne", "ml", "ja", "hi", "pa", "ko", "ar", "fa", "zh-tw", "zh-cn", "ta", "ru", "bg", "pt", "te", "Other"]



def clean_text(text):

    text = text.replace("@eng", "")
    text = text.replace("@en", "")
    text = text.replace("@la", "")
    text = text.replace("<http://www.w3.org/2001/XMLSchema#date>", "")
    text = text.replace("<http://www.w3.org/2001/XMLSchema#string>", "")

    if text == "None":
        text = "None"

    clean_text = re.sub(r'[^\w\s\']', ' ', text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    return clean_text


"""
    Find the name of the entity by the postfix of its url
"""    
def find_url_name(url):

    url = url.replace("dbp:", "")
    cleaned_text = clean_text(url.split("/")[-1])
    cleaned_text = cleaned_text.replace("_", " ")
        
    return cleaned_text


"""
    Map ids to uris
"""
def get_ids_to_uris(dataset, num):
    data_path = "../knowformer_data/" + dataset
    ids_to_uris = {}
    with open(data_path + "/ent_ids_" + num) as fp:
        for line in fp:
            ids_to_uris[int(line.split("\t")[0])] = line.split("\t")[1].rstrip()
    return ids_to_uris


"""
    dataframe with attribute triples
    the following dictionary with the attribute types for each entity:
        {
            "http://dbpedia.org/resource/Captain_Pirate": [ "http://dbpedia.org/ontology/imdbId", ...]
        }
"""
def load_attr_graph(kg_id):
    
    data_path = "../knowformer_data/" + dataset + "/attr_triples_" + kg_id
    attr_df = pd.read_csv(data_path,  sep='\t', names=["e1", "attr", "val"])
    attr_dict = {}
    with open(data_path, "r") as fp:
        for line in fp:
            ent = line.split("\t")[0]
            attr = line.split("\t")[1]

            if ent not in attr_dict.keys():
                attr_dict[ent] = list()

            attr_dict[ent].append(attr)

    return attr_df, attr_dict

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")