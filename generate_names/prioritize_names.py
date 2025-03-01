from utils import *

"""
    Find the url of an entity by mapping from id --> url (if need)
"""
def find_url(e1):
    
    if e1 in ids_to_uris_1:
        return ids_to_uris_1[e1]
    elif e1 in ids_to_uris_2:
        return ids_to_uris_2[e1]
    else:
        return e1

def gen_names(ids_to_uris, names_df, random_init_flag, url_priority, prior_attr):
    
    ent_names = {}
    for ent in ids_to_uris:

        rows = names_df[names_df["e1"] == ent]

        # in case entity not in dataframe from name_analysis.py, random init or find the URL name manually
        # depending of the URL is meaningful in this graph
        if len(rows) == 0:
            if random_init_flag:
                name = "no_value"
                src = "random"
                ent_names[ent] = (name, src)
            else:
                name = find_url_name(find_url(ent))
                src = "url"
                ent_names[ent] = (name, src)
            continue

        # give priority to url
        if url_priority:
            rand_row = rows.sample(n=1)
            name = rand_row["URL Names"].values[0]
            src = "url"
            ent_names[ent] = (name, src)
        
        # give priority to attributes
        else:
            prior_attr_rows = rows[rows["attr"].isin(prior_attr)]
            non_prior_attr_rows = rows[~rows["attr"].isin(prior_attr)]

            if len(prior_attr_rows) == 0:
                
                # alt_label_rows = non_prior_attr_rows[non_prior_attr_rows["attr"]
                #                                 == "https://www.scads.de/movieBenchmark/ontology/originalTitle"]
                # if len(alt_label_rows) != 0:
                #     rand_row = alt_label_rows.sample(n=1)
                # else:
                #     rand_row = non_prior_attr_rows.sample(n=1)

                rand_row = non_prior_attr_rows.sample(n=1)
                name = rand_row["replaced_puncs"].values[0]
                src = "non_prior"
                ent_names[ent] = (name, src)
            elif len(prior_attr_rows) > 0:
                rand_row = prior_attr_rows.sample(n=1)
                name = rand_row["replaced_puncs"].values[0]
                src = "prior"
                ent_names[ent] = (name, src)
            else:
                if random_init_flag:
                    name = "no_value"
                    src = "random"
                    ent_names[ent] = (name, src)
                else:
                    rand_row = prior_attr_rows.sample(n=1)
                    name = rand_row["URL Names"].values[0]
                    src = "url"
                    ent_names[ent] = (name, src)
    return ent_names

# Global to be used in utils
ids_to_uris_1 = get_ids_to_uris(dataset, "1")
ids_to_uris_2 = get_ids_to_uris(dataset, "2")

def run_prioritize():

    if dataset == "D_W_15K_V1":

            url_priority_1 = True
            url_priority_2 = False

            random_init_flag_1 = False
            random_init_flag_2 = True

            prior_attr_1 = []
            prior_attr_2 = ["http://www.wikidata.org/entity/P373", 'http://www.wikidata.org/entity/P1476']

            # paths to dataframes exported by name_analysis.py
            kg1_src_path = "entity_names/" + dataset + "/DBpedia_analysis.xlsx"
            kg2_src_path = "entity_names/" + dataset + "/Wikidata_analysis.xlsx"

            # path to dataframes with entity names
            kg1_dest_path = "entity_names/" + dataset + "/DBpedia_names.xlsx"
            kg2_dest_path = "entity_names/" + dataset + "/Wikidata_names.xlsx"
    
    elif dataset == "ICEW_WIKI":
            url_priority_1 = True
            url_priority_2 = True

            random_init_flag_1 = False
            random_init_flag_2 = False

            prior_attr_1 = []
            prior_attr_2 = []

            # paths to dataframes exported by name_analysis.py
            kg1_src_path = "entity_names/" + dataset + "/icew_analysis.xlsx"
            kg2_src_path = "entity_names/" + dataset + "/wiki_analysis.xlsx"

            # path to dataframes with entity names
            kg1_dest_path = "entity_names/" + dataset + "/icew_names.xlsx"
            kg2_dest_path = "entity_names/" + dataset + "/wiki_names.xlsx"

    elif dataset == "ICEW_YAGO":
            url_priority_1 = True
            url_priority_2 = True

            random_init_flag_1 = False
            random_init_flag_2 = False

            prior_attr_1 = []
            prior_attr_2 = []

            # paths to dataframes exported by name_analysis.py
            kg1_src_path = "entity_names/" + dataset + "/icew_analysis.xlsx"
            kg2_src_path = "entity_names/" + dataset + "/yago_analysis.xlsx"

            # path to dataframes with entity names
            kg1_dest_path = "entity_names/" + dataset + "/icew_names.xlsx"
            kg2_dest_path = "entity_names/" + dataset + "/yago_names.xlsx"

    elif dataset == "BBC_DB":

            url_priority_1 = False
            url_priority_2 = True

            random_init_flag_1 = True
            random_init_flag_2 = False

            prior_attr_1 = ['http://purl.org/dc/elements/1.1/title', 'http://xmlns.com/foaf/0.1/name', 'http://open.vocab.org/terms/sortlabel']
            prior_attr_2 = ['http://xmlns.com/foaf/0.1/name', 'prop:birthname', 'rdfs:label', 'prop:name']

            # paths to dataframes exported by name_analysis.py
            kg1_src_path = "entity_names/" + dataset + "/BBC_analysis.xlsx"
            kg2_src_path = "entity_names/" + dataset + "/DBpedia_analysis.xlsx"

            # path to dataframes with entity names
            kg1_dest_path = "entity_names/" + dataset + "/BBC_names.xlsx"
            kg2_dest_path = "entity_names/" + dataset + "/DBpedia_names.xlsx"
            
    elif dataset == "D_W_15K_V2":

            url_priority_1 = True
            url_priority_2 = False

            random_init_flag_1 = False
            random_init_flag_2 = True

            prior_attr_1 = []
            prior_attr_2 = ["http://www.wikidata.org/entity/P373", 'http://www.wikidata.org/entity/P1476']

            # paths to dataframes exported by name_analysis.py
            kg1_src_path = "entity_names/" + dataset + "/DBpedia_analysis.xlsx"
            kg2_src_path = "entity_names/" + dataset + "/Wikidata_analysis.xlsx"

            # path to dataframes with entity names
            kg1_dest_path = "entity_names/" + dataset + "/DBpedia_names.xlsx"
            kg2_dest_path = "entity_names/" + dataset + "/Wikidata_names.xlsx"
            
    elif dataset == "SRPRS_D_W_15K_V1":

            url_priority_1 = True
            url_priority_2 = False

            random_init_flag_1 = False
            random_init_flag_2 = True

            prior_attr_1 = []
            prior_attr_2 = ["http://www.wikidata.org/entity/P373", 'http://www.wikidata.org/entity/P1476']

            # paths to dataframes exported by name_analysis.py
            kg1_src_path = "entity_names/" + dataset + "/DBpedia_analysis.xlsx"
            kg2_src_path = "entity_names/" + dataset + "/Wikidata_analysis.xlsx"

            # path to dataframes with entity names
            kg1_dest_path = "entity_names/" + dataset + "/DBpedia_names.xlsx"
            kg2_dest_path = "entity_names/" + dataset + "/Wikidata_names.xlsx"
            
    elif dataset == "SRPRS_D_W_15K_V2":

            url_priority_1 = True
            url_priority_2 = False

            random_init_flag_1 = False
            random_init_flag_2 = True

            prior_attr_1 = []
            prior_attr_2 = ["http://www.wikidata.org/entity/P373", 'http://www.wikidata.org/entity/P1476']

            # paths to dataframes exported by name_analysis.py
            kg1_src_path = "entity_names/" + dataset + "/DBpedia_analysis.xlsx"
            kg2_src_path = "entity_names/" + dataset + "/Wikidata_analysis.xlsx"

            # path to dataframes with entity names
            kg1_dest_path = "entity_names/" + dataset + "/DBpedia_names.xlsx"
            kg2_dest_path = "entity_names/" + dataset + "/Wikidata_names.xlsx"
            
    elif dataset == "fr_en":
        
        url_priority_1 = True
        url_priority_2 = True
        
        random_init_flag_1 = False
        random_init_flag_2 = False
        
        prior_attr_1 = ["http://fr.dbpedia.org/property/titre", "http://xmlns.com/foaf/0.1/name", "http://fr.dbpedia.org/property/name", "http://fr.dbpedia.org/property/label"]

        prior_attr_2 = ["http://dbpedia.org/property/title", "http://xmlns.com/foaf/0.1/name", "http://dbpedia.org/property/name",  "http://xmlns.com/foaf/0.1/givenName", "http://dbpedia.org/ontology/birthName", "http://dbpedia.org/property/label"]

        # paths to dataframes exported by name_analysis.py
        kg1_src_path = "entity_names/" + dataset + "/fr_analysis.xlsx"
        kg2_src_path = "entity_names/" + dataset + "/en_analysis.xlsx"

        # path to dataframes with entity names
        kg1_dest_path = "entity_names/" + dataset + "/fr_names.xlsx"
        kg2_dest_path = "entity_names/" + dataset + "/en_names.xlsx"
        
        
    elif dataset == "ja_en":
        
        url_priority_1 = False
        url_priority_2 = False
        
        random_init_flag_1 = True
        random_init_flag_2 = False
        
        prior_attr_1 = ["http://ja.dbpedia.org/property/title", "http://xmlns.com/foaf/0.1/name", "http://ja.dbpedia.org/property/name", "http://xmlns.com/foaf/0.1/givenName", "http://ja.dbpedia.org/property/label"]

        prior_attr_2 = ["http://dbpedia.org/property/title", "http://xmlns.com/foaf/0.1/name", "http://dbpedia.org/property/name",  "http://xmlns.com/foaf/0.1/givenName", "http://dbpedia.org/property/label"]

        # paths to dataframes exported by name_analysis.py
        kg1_src_path = "entity_names/" + dataset + "/ja_analysis.xlsx"
        kg2_src_path = "entity_names/" + dataset + "/en_analysis.xlsx"

        # path to dataframes with entity names
        kg1_dest_path = "entity_names/" + dataset + "/ja_names.xlsx"
        kg2_dest_path = "entity_names/" + dataset + "/en_names.xlsx"
        
    elif dataset == "zh_en":
        
        url_priority_1 = False
        url_priority_2 = False
        
        random_init_flag_1 = True
        random_init_flag_2 = False
        
        prior_attr_1 = ["http://zh.dbpedia.org/property/title", "http://xmlns.com/foaf/0.1/name", "http://ja.dbpedia.org/property/name", "http://xmlns.com/foaf/0.1/givenName", "http://ja.dbpedia.org/property/label"]

        prior_attr_2 = ["http://dbpedia.org/property/title", "http://xmlns.com/foaf/0.1/name", "http://dbpedia.org/property/name",  "http://xmlns.com/foaf/0.1/givenName", "http://dbpedia.org/property/label"]

        # paths to dataframes exported by name_analysis.py
        kg1_src_path = "entity_names/" + dataset + "/zh_analysis.xlsx"
        kg2_src_path = "entity_names/" + dataset + "/en_analysis.xlsx"

        # path to dataframes with entity names
        kg1_dest_path = "entity_names/" + dataset + "/zh_names.xlsx"
        kg2_dest_path = "entity_names/" + dataset + "/en_names.xlsx"

    names_df_1 = pd.read_excel(kg1_src_path)
    names_df_2 = pd.read_excel(kg2_src_path)

    print("Preparing KG1 ...")
    ent_names_1 = gen_names(ids_to_uris_1, names_df_1, random_init_flag_1, url_priority_1, prior_attr_1)
    
    print("Preparing KG2 ...")
    ent_names_2 = gen_names(ids_to_uris_2, names_df_2, random_init_flag_2, url_priority_2, prior_attr_2)

    ent_names_df_1 = pd.DataFrame(([k, v[0], v[1]] for k,v in ent_names_1.items()), columns = ['e1', 'name', 'src'])
    ent_names_df_1.to_excel(kg1_dest_path)
    print(kg1_dest_path + " exported succesfully")

    ent_names_df_2 = pd.DataFrame(([k, v[0], v[1]] for k,v in ent_names_2.items()), columns = ['e1', 'name', 'src'])
    ent_names_df_2.to_excel(kg2_dest_path)
    print(kg2_dest_path + " exported succesfully")