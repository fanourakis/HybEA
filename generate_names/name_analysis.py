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

def export_names(df, kg_dest_path, attrs):
    
    # Keep only the manually selected attributes for name's information
    filtered_df_1 = df[df['attr'].isin(attrs)]
    # sort by entity
    sorted_df = filtered_df_1.sort_values(by='e1', ascending=True)
    # find the url of an entity
    sorted_df['URLs'] = sorted_df.apply(lambda x: find_url(x.e1), axis=1)

    # the following steps are for cleaning the text
    sorted_df['replaced_puncs'] = sorted_df.apply(lambda x: clean_text(x.val), axis=1)
    sorted_df["Lang"] = sorted_df.apply(lambda x: lang_detect(x.replaced_puncs), axis=1)
    sorted_df["Delete"] = sorted_df.apply(lambda x: to_delete(x.Lang), axis=1)

    # find name by url
    sorted_df['URL Names'] = sorted_df.apply(lambda x: find_url_name(x.URLs), axis=1)
    # export to excel
    sorted_df.to_excel(kg_dest_path)
    print(kg_dest_path + " exported successfully !")
    

# Global to be used in utils
ids_to_uris_1 = get_ids_to_uris(dataset, "1")
ids_to_uris_2 = get_ids_to_uris(dataset, "2")
    

def run_name_analysis():


    attr_df_1, attr_dict_kg1 = load_attr_graph("1")
    attr_df_2, attr_dict_kg2 = load_attr_graph("2")

    # Print to see all the available attributes of the two KGs
    # temp1 = [i for x in attr_dict_kg1.values() for i in x]
    # temp2 = [i for x in attr_dict_kg2.values() for i in x]
    # print(set(temp2))
    # exit()

    if dataset == "D_W_15K_V1":

        attrs_1 = ["http://xmlns.com/foaf/0.1/name", "http://xmlns.com/foaf/0.1/givenName", "http://dbpedia.org/ontology/birthName",
        "http://dbpedia.org/ontology/name", "http://dbpedia.org/ontology/longName", "http://dbpedia.org/ontology/otherName", "http://dbpedia.org/ontology/teamName"]

        attrs_2 = ['http://www.wikidata.org/entity/P373', 'http://www.wikidata.org/entity/P1476', 'http://www.w3.org/2004/02/skos/core#altLabel', 'http://schema.org/description']

        kg1_dest_path = "entity_names/" + dataset + "/DBpedia_analysis.xlsx"
        kg2_dest_path = "entity_names/" + dataset + "/Wikidata_analysis.xlsx"
    
    elif dataset == "BBC_DB":

        attrs_1 = ['http://purl.org/dc/elements/1.1/title', 'http://xmlns.com/foaf/0.1/name', 'http://open.vocab.org/terms/sortlabel']

        attrs_2 = ['http://xmlns.com/foaf/0.1/name', 'prop:birthname', 'rdfs:label', 'prop:name', 'http://xmlns.com/foaf/0.1/givenname', 'http://xmlns.com/foaf/0.1/surname', 'prop:label',
        
        'http://purl.org/dc/elements/1.1/description', 'prop:description', 'prop:shortdescription']

        kg1_dest_path = "entity_names/" + dataset + "/BBC_analysis.xlsx"
        kg2_dest_path = "entity_names/" + dataset + "/DBpedia_analysis.xlsx"
        
    elif dataset == "D_W_15K_V2":
        
        attrs_1 = ["http://xmlns.com/foaf/0.1/name", "http://xmlns.com/foaf/0.1/givenName", "http://dbpedia.org/ontology/birthName",
        "http://dbpedia.org/ontology/longName"]

        attrs_2 = ['http://www.wikidata.org/entity/P373', 'http://www.wikidata.org/entity/P1476', 'http://www.w3.org/2004/02/skos/core#altLabel', 'http://schema.org/description']

        kg1_dest_path = "entity_names/" + dataset + "/DBpedia_analysis.xlsx"
        kg2_dest_path = "entity_names/" + dataset + "/Wikidata_analysis.xlsx"

    elif dataset == "SRPRS_D_W_15K_V1":

        attrs_1 = ["http://xmlns.com/foaf/0.1/name", "http://xmlns.com/foaf/0.1/givenName", "http://dbpedia.org/ontology/birthName",
        "http://dbpedia.org/ontology/name", "http://dbpedia.org/ontology/longName", "http://dbpedia.org/ontology/otherName", "http://dbpedia.org/ontology/teamName"]

        attrs_2 = ['http://www.wikidata.org/entity/P373', 'http://www.wikidata.org/entity/P1476', 'http://www.w3.org/2004/02/skos/core#altLabel', 'http://schema.org/description']

        kg1_dest_path = "entity_names/" + dataset + "/DBpedia_analysis.xlsx"
        kg2_dest_path = "entity_names/" + dataset + "/Wikidata_analysis.xlsx"
        
    elif dataset == "SRPRS_D_W_15K_V2":
        
        attrs_1 = ["http://dbpedia.org/ontology/title", "http://dbpedia.org/ontology/birthName", "http://dbpedia.org/ontology/longName"]

        attrs_2 = ['http://www.wikidata.org/entity/P373', 'http://www.wikidata.org/entity/P1476']

        kg1_dest_path = "entity_names/" + dataset + "/DBpedia_analysis.xlsx"
        kg2_dest_path = "entity_names/" + dataset + "/Wikidata_analysis.xlsx"
    
    elif dataset == "fr_en":
        
        attrs_1 = ["http://fr.dbpedia.org/property/titre", "http://xmlns.com/foaf/0.1/name", "http://fr.dbpedia.org/property/name", "http://fr.dbpedia.org/property/label"]

        attrs_2 = ["http://dbpedia.org/property/title", "http://xmlns.com/foaf/0.1/name", "http://dbpedia.org/property/name",  "http://xmlns.com/foaf/0.1/givenName", "http://dbpedia.org/ontology/birthName", "http://dbpedia.org/property/label"]

        kg1_dest_path = "entity_names/" + dataset + "/fr_analysis.xlsx"
        kg2_dest_path = "entity_names/" + dataset + "/en_analysis.xlsx"
        
    elif dataset == "ja_en":
        
        attrs_1 = ["http://ja.dbpedia.org/property/title", "http://xmlns.com/foaf/0.1/name", "http://ja.dbpedia.org/property/name", "http://xmlns.com/foaf/0.1/givenName", "http://ja.dbpedia.org/property/label"]

        attrs_2 = ["http://dbpedia.org/property/title", "http://xmlns.com/foaf/0.1/name", "http://dbpedia.org/property/name",  "http://xmlns.com/foaf/0.1/givenName", "http://dbpedia.org/property/label"]

        kg1_dest_path = "entity_names/" + dataset + "/ja_analysis.xlsx"
        kg2_dest_path = "entity_names/" + dataset + "/en_analysis.xlsx"
        
    elif dataset == "zh_en":
        
        attrs_1 = ["http://zh.dbpedia.org/property/title", "http://xmlns.com/foaf/0.1/name", "http://ja.dbpedia.org/property/name", "http://xmlns.com/foaf/0.1/givenName", "http://ja.dbpedia.org/property/label"]

        attrs_2 = ["http://dbpedia.org/property/title", "http://xmlns.com/foaf/0.1/name", "http://dbpedia.org/property/name",  "http://xmlns.com/foaf/0.1/givenName", "http://dbpedia.org/property/label"]

        kg1_dest_path = "entity_names/" + dataset + "/zh_analysis.xlsx"
        kg2_dest_path = "entity_names/" + dataset + "/en_analysis.xlsx"
    elif dataset == "ICEW_WIKI":
        attrs_1 = ["has_name"]
        attrs_2 = ["has_name"]
        kg1_dest_path = "entity_names/" + dataset + "/icew_analysis.xlsx"
        kg2_dest_path = "entity_names/" + dataset + "/wiki_analysis.xlsx"
    elif dataset == "ICEW_YAGO":
        attrs_1 = ["has_name"]
        attrs_2 = ["has_name"]
        kg1_dest_path = "entity_names/" + dataset + "/icew_analysis.xlsx"
        kg2_dest_path = "entity_names/" + dataset + "/yago_analysis.xlsx"
        
    create_folder_if_not_exists("entity_names/" + dataset)

    print("Preparing name analysis on KG1 ...")
    export_names(attr_df_1, kg1_dest_path, attrs_1)

    print("Preparing name analysis on KG2 ...")
    export_names(attr_df_2, kg2_dest_path, attrs_2)

    # Read the exported .xslx files for validation
    # names_df_1 = pd.read_excel(kg1_dest_path)
    # names_df_2 = pd.read_excel(kg2_dest_path)