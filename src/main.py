from attribute_model.attr_main import *
from attribute_model.utils.Read_data_func import *
from structure_model.structure_main import *
from structure_model.reader.helper import *
import torch
import os
import matplotlib.pyplot as plt
import sys
from calculate_performance import *
from calculate_cummulative import *
import pickle

def get_item(x):
    if torch.is_tensor(x):
        return x.item()
    else:
        return x
            
def reciprocity(ents_1, ents_2, res_mat_1, res_mat_2):
    max_indices_1 = np.argmax(res_mat_1, axis=1)
    max_indices_2 = np.argmax(res_mat_2, axis=1)
    pairs = set()
    for i in range(len(ents_1)):
        if max_indices_1[i] == max_indices_2[i]:
            pairs.add((ents_1[i], ents_2[i]))
    return pairs


def main(args):

    print(args[1])
    print(DATASET)
    
    MYPATH = "experiments/" + DATASET + "_" + args[1]
    if not os.path.exists(MYPATH):
        os.makedirs(MYPATH)

    if args[1] == "Hybea":
        turn = 2
        stop_structure = False
        stop_attribute = False
        stop_first_iteration = False
    elif args[1] == "Hybea_struct_first":
        turn = 1
        stop_structure = False
        stop_attribute = False
        stop_first_iteration = False
    elif args[1] == "Hybea_without_structure":
        turn = 2
        stop_structure = False
        stop_attribute = True
        stop_first_iteration = False
    elif args[1] == "Hybea_without_factual":
        turn = 1
        stop_structure = True
        stop_attribute = False
        stop_first_iteration = False
    elif args[1] == "Hybea_basic":
        turn = 2
        stop_structure = False
        stop_attribute = False
        stop_first_iteration = True
    elif args[1] == "Hybea_basic_structure_first":
        turn = 1
        stop_structure = False
        stop_attribute = False
        stop_first_iteration = True

    added_ents_of_KG1 = set()
    while True:
        
        new_pairs = set()
        
        # DATA_PATH is from attribute model Param.py
        _, ent_ids_1, ent_ids_2 = read_structure_datas(DATA_PATH, reversed=True)
        _, ent_ids_1_nrev, ent_ids_2_nrev = read_structure_datas(DATA_PATH, reversed=False)

        if turn % 2 == 0:
            
            newp_struct = []
            for i in range(3):
                if os.path.exists(MYPATH + "/" + "rec_new_pairs_from_structure" + str(i) + ".pickle"):
                    with open(MYPATH + "/" + "rec_new_pairs_from_structure" + str(i) + ".pickle", "rb") as file:
                        pstruct = pickle.load(file)
                    for pair in pstruct:
                        newp_struct.append((ent_ids_1_nrev[pair[0]], ent_ids_2_nrev[pair[1]]))
                        added_ents_of_KG1.add(pair[0])
                        
            newp_attr = []
            for i in range(3):
                if os.path.exists(MYPATH + "/" + "rec_new_pairs_from_attr" + str(i) + ".pickle"):
                    with open(MYPATH + "/" + "rec_new_pairs_from_attr" + str(i) + ".pickle", "rb") as file:
                        pattr = pickle.load(file)
                    for pair in pattr:
                        newp_attr.append((ent_ids_1_nrev[pair[0]], ent_ids_2_nrev[pair[1]]))
                        added_ents_of_KG1.add(pair[0])
                    
            print(len(newp_attr))
            print(len(newp_struct))
            newp = set(newp_attr).union(set(newp_struct))
            print(len(newp))
            
            ent_ill, res_mat_1, res_mat_2, loss_list, hits_1_list = run_attr_model(newp)
            ents_1 = [e1 for e1,e2 in ent_ill]
            ents_2 = [e2 for e1,e2 in ent_ill]
            results = reciprocity(ents_1, ents_2, res_mat_1, res_mat_2)
            
            for pairs in results:
                
                if ent_ids_1[pairs[0]] not in added_ents_of_KG1:
                    added_ents_of_KG1.add(ent_ids_1[pairs[0]])
                    new_pairs.add((ent_ids_1[pairs[0]], ent_ids_2[pairs[1]]))
            
            print(len(new_pairs))        
                    
            index = 0
            while os.path.exists(MYPATH + "/" + "rec_new_pairs_from_attr" + str(index) + ".pickle"):
                index += 1
            
            with open(MYPATH + "/" + "rec_new_pairs_from_attr" + str(index) + ".pickle", "wb") as file:
                pickle.dump(new_pairs, file)
                
            with open(MYPATH + "/" + "res_mat_1_attr" + str(index) + ".pickle", "wb") as file:
                pickle.dump(res_mat_1, file)

            if stop_attribute:
                calc_hits(DATASET, MYPATH, mode)
                exit()
            
            turn += 1

        else:
            
            newp_attr = []
            for i in range(3):
                if os.path.exists(MYPATH + "/" + "rec_new_pairs_from_attr" + str(i) + ".pickle"):
                    with open(MYPATH + "/" + "rec_new_pairs_from_attr" + str(i) + ".pickle", "rb") as file:
                        newp_attr_pickle = list(pickle.load(file))
                                    
                    for pair in newp_attr_pickle:
                        newp_attr.append((pair[0], pair[1]))
                        added_ents_of_KG1.add(pair[0])
                        
            newp_struct = []
            for i in range(3):
                if os.path.exists(MYPATH + "/" + "rec_new_pairs_from_structure" + str(i) + ".pickle"):
                    with open("rec_new_pairs_from_structure" + str(i) + ".pickle", "rb") as file:
                        newp_struct_pickle = list(pickle.load(file))
                                        
                    for pair in newp_struct_pickle:
                        newp_struct.append((pair[0], pair[1]))
                        added_ents_of_KG1.add(pair[0])
                
            print(len(newp_attr))
            print(len(newp_struct))
            newp = set(newp_attr).union(set(newp_struct))
            print(len(newp))    

            
            res_mat_1, res_mat_2, ents_1, ents_2, vocab, epoch_loss_list, hits_1_list = run_structure_model(set(newp))

            results = reciprocity(ents_1, ents_2, res_mat_1, res_mat_2)
                
            vocab = {v: k for k, v in vocab.items()}
            for pairs in results:
                    
                if vocab[get_item(pairs[0])] not in added_ents_of_KG1:
                    added_ents_of_KG1.add(vocab[get_item(pairs[0])])
                    new_pairs.add((vocab[get_item(pairs[0])], vocab[get_item(pairs[1])]))
            
            print(len(new_pairs))
            
            index = 0
            while os.path.exists(MYPATH + "/" + "rec_new_pairs_from_structure" + str(index) + ".pickle"):
                index += 1
                
            with open(MYPATH + "/" + "rec_new_pairs_from_structure" + str(index) + ".pickle", "wb") as file:
                pickle.dump(new_pairs, file)
                
            with open("res_mat_1_struct" + str(index) + ".pickle", "wb") as file:
                pickle.dump(res_mat_1, file)

            if stop_structure:
                calc_hits(DATASET, MYPATH, mode)
                exit()

            turn += 1

            
        if len(new_pairs) == 0 or stop_first_iteration:
            break
    
    calc_hits(DATASET, MYPATH + "/", mode)

if __name__ == "__main__":
    main(sys.argv)