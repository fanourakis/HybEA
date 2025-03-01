# structure model param
DATASET = "D_W_15K_V1"

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

TASK = "entity-alignment"

RANDOM_INITIALIZATION = False
if RANDOM_INITIALIZATION:
    HIDDEN_SIZE = 256
else:
    HIDDEN_SIZE = 768
    
NUM_HIDDEN_LAYERS = 12
NUM_ATTENTION_HEADS = 4
INPUT_DROPOUT_PROB = 0.5
ATTENTION_DROPOUT_PROB = 0.1
HIDDEN_DROPOUT_PROB = 0.3
RESIDUAL_DROPOUT_PROB = 0.1
INITIALIZER_RANGE = 0.02
INTERMEDIATE_SIZE = 2048
RESIDUAL_W = 0.5
EPOCH = 200
MIN_EPOCHS = 10
LEARNING_RATE = 5e-4
BATCH_SIZE = 2048
EVAL_BATCH_SIZE = 4096
EARLY_STOP_MAX_TIMES = 3
SOFT_LABEL = 0.25
EVAL_FREQ = 5
START_EVAL = 0
SWA_PRE_NUM = 5
DO_TRAIN = True
DO_TEST = True
USE_GELU = False
ADDITION_LOSS_W = 0.1
RELATION_COMBINE_DROPOUT_PROB = 0.2
CSLS = 2

if DATASET == "D_W_15K_V1":
    VOC_LIM_1 = 15100
    VOC_LIM_2 = 27100
elif DATASET == "BBC_DB":
    VOC_LIM_1 = 9496
    VOC_LIM_2 = 17013
elif DATASET == "D_W_15K_V2":
    VOC_LIM_1 = 15100
    VOC_LIM_2 = 27100
elif DATASET == "SRPRS_D_W_15K_V1":
    VOC_LIM_1 = 15100
    VOC_LIM_2 = 27100
elif DATASET == "SRPRS_D_W_15K_V2":
    VOC_LIM_1 = 15100
    VOC_LIM_2 = 27100
elif DATASET == "fr_en":
    VOC_LIM_1 = 19761
    VOC_LIM_2 = 36754
elif DATASET == "ja_en":
    VOC_LIM_1 = 19914
    VOC_LIM_2 = 36694
elif DATASET == "zh_en":
    VOC_LIM_1 = 19488
    VOC_LIM_2 = 36060
elif DATASET == "ICEW_WIKI":
    VOC_LIM_1 = 11147
    VOC_LIM_2 = 25979
elif DATASET == "ICEW_YAGO":
    VOC_LIM_1 = 26954
    VOC_LIM_2 = 44443