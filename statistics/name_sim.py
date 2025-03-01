import Levenshtein
import pandas as pd

def dataset_to_path(DATASET):
    if DATASET == "D_W_15K_V1":
        KG1_PATH_FOR_NAMES = 'DBpedia_names.xlsx'
        KG2_PATH_FOR_NAMES = 'Wikidata_names.xlsx'
    elif DATASET == "BBC_DB":
        KG1_PATH_FOR_NAMES = 'BBC_names.xlsx'
        KG2_PATH_FOR_NAMES = 'DBpedia_names.xlsx'
    elif DATASET == "D_W_15K_V2":
        KG1_PATH_FOR_NAMES = 'DBpedia_names.xlsx'
        KG2_PATH_FOR_NAMES = 'Wikidata_names.xlsx'
    elif DATASET == "SRPRS_D_W_15K_V1":
        KG1_PATH_FOR_NAMES = 'DBpedia_names.xlsx'
        KG2_PATH_FOR_NAMES = 'Wikidata_names.xlsx'
    elif DATASET == "SRPRS_D_W_15K_V2":
        KG1_PATH_FOR_NAMES = 'DBpedia_names.xlsx'
        KG2_PATH_FOR_NAMES = 'Wikidata_names.xlsx'
    elif DATASET == "fr_en":
        KG1_PATH_FOR_NAMES = 'fr_names.xlsx'
        KG2_PATH_FOR_NAMES = 'en_names.xlsx'
    elif DATASET == "ja_en":
        KG1_PATH_FOR_NAMES = 'ja_names.xlsx'
        KG2_PATH_FOR_NAMES = 'en_names.xlsx'
    elif DATASET == "zh_en":
        KG1_PATH_FOR_NAMES = 'zh_names.xlsx'
        KG2_PATH_FOR_NAMES = 'en_names.xlsx'
    elif DATASET == "ICEW_WIKI":
        KG1_PATH_FOR_NAMES = 'icew_names.xlsx'
        KG2_PATH_FOR_NAMES = 'wiki_names.xlsx'
    elif DATASET == "ICEW_YAGO":
        KG1_PATH_FOR_NAMES = 'icew_names.xlsx'
        KG2_PATH_FOR_NAMES = 'yago_names.xlsx'

    return KG1_PATH_FOR_NAMES, KG2_PATH_FOR_NAMES

def excel_to_dict(file_path, sheet_name=0):
    df = pd.read_excel(file_path, usecols=[1, 2])
    
    return dict(zip(df["e1"], df["name"]))
    

for dataset in ["ICEW_WIKI", "D_W_15K_V2", "SRPRS_D_W_15K_V1", "SRPRS_D_W_15K_V2", "BBC_DB", "ICEW_WIKI", "ICEW_YAGO", "fr_en", "zh_en", "ja_en"]:
    print(dataset)
    ents_ids = {}
    for i in range(2):
        
        with open("../../attribute_data/" + dataset + "/ent_ids_" + str(i + 1)) as fp:
            for line in fp:
                id = line.split("\t")[0]
                url = line.split("\t")[1].rstrip()
                ents_ids[url] = int(id)

    KG1_PATH_FOR_NAMES, KG2_PATH_FOR_NAMES = dataset_to_path(dataset)
    ent_1_names = excel_to_dict("../../generate_names/entity_names/" + dataset + "/" + KG1_PATH_FOR_NAMES)
    ent_2_names = excel_to_dict("../../generate_names/entity_names/" + dataset + "/" + KG2_PATH_FOR_NAMES)

    sim = 0
    counter = 0
    with open("../../knowformer_data/" + dataset + "/ent_ILLs.txt") as fp:
            for line in fp:
                url_1 = line.split("\t")[0]
                url_2 = line.split("\t")[1].rstrip()

                if url_1 in ents_ids and url_2 in ents_ids:
                    name_1 = ent_1_names[ents_ids[url_1]]
                    name_2 = ent_2_names[ents_ids[url_2]]
                    
                    print(name_1)
                    print(name_2)
                    print()
                    
                    if name_1 != "no_value" and name_2 != "no_value":
                        sim += Levenshtein.ratio(name_1, name_2)
                        counter += 1
                    else:
                        counter += 1
                    
    print(str(sim / counter)) 
    print(str(counter) + "\n")
    exit()