print("In params:")

CUDA_NUM = 1 # used GPU num
MODEL_INPUT_DIM  = 768

SEED_NUM = 11037

EPOCH_NUM = 200 #training epoch num

NEAREST_SAMPLE_NUM = 128
CANDIDATE_GENERATOR_BATCH_SIZE = 128

NEG_NUM = 2 # negative sample num
MARGIN = 3 # margin
LEARNING_RATE = 1e-5 # learning rate
TRAIN_BATCH_SIZE = 24
TEST_BATCH_SIZE = 128

FOLD = "2"
DATASET = "D_W_15K_V1"
CSLS = 2

if DATASET == "D_W_15K_V1":
    TOPK = 1000
    INPUT_SIZE_1 = 341
    INPUT_SIZE_2 = 649
elif DATASET == "BBC_DB":
    TOPK = 939
    INPUT_SIZE_1 = 4
    INPUT_SIZE_2 = 723
elif DATASET == "D_W_15K_V2":
    TOPK = 1000
    INPUT_SIZE_1 = 175
    INPUT_SIZE_2 = 457
elif DATASET == "SRPRS_D_W_15K_V1":
    TOPK = 1000
    INPUT_SIZE_1 = 363
    INPUT_SIZE_2 = 652
elif DATASET == "SRPRS_D_W_15K_V2":
    TOPK = 1000
    INPUT_SIZE_1 = 256
    INPUT_SIZE_2 = 531

DATA_PATH = r"../attribute_data/" + DATASET + "/"