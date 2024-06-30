import os
import pickle
import numpy as np
import sys

def calc_hits(DATASET, MYPATH, mode):
    
    print(DATASET)
    print(MYPATH)
    print(mode)

    test_set = []
    with open("../knowformer_data/" + DATASET + "/ref_ents.txt", "r") as fp:
        for line in fp:
            e1 = line.split("\t")[0]
            e2 = line.split("\t")[1].rstrip()
            test_set.append((e1, e2))
            
    ents_1 = [e1 for e1,e2 in test_set]
    ents_2 = [e2 for e1,e2 in test_set]

    if mode == "Hybea":
        index = 0
        while os.path.exists(MYPATH + "/" + "res_mat_1_struct" + str(index) + ".pickle"):
                index += 1
        with open(MYPATH + "res_mat_1_struct" + str(index - 1) + ".pickle", "rb") as file:
            res_mat_1 = pickle.load(file)
    elif mode == "Hybea_struct_first":
        index = 0
        while os.path.exists(MYPATH + "/" + "res_mat_1_attr" + str(index) + ".pickle"):
                index += 1
        with open(MYPATH + "res_mat_1_attr" + str(index - 1) + ".pickle", "rb") as file:
            res_mat_1 = pickle.load(file).detach().cpu().numpy()
    elif mode == "Hybea_basic":
        with open(MYPATH + "res_mat_1_struct0.pickle", "rb") as file:
            res_mat_1 = pickle.load(file)
    elif mode == "Hybea_basic_structure_first":
        with open(MYPATH + "res_mat_1_attr0.pickle", "rb") as file:
            res_mat_1 = pickle.load(file).detach().cpu().numpy()
    else:
        print("Wrong mode !")
        exit()

    index = 0
    while os.path.exists(MYPATH + "/" + "rec_new_pairs_from_structure" + str(index) + ".pickle"):
        index += 1
    
    newp_struct_list = []
    for i in range(0, index):
        with open(MYPATH + "rec_new_pairs_from_structure" + str(i) + ".pickle", "rb") as file:
            newp_struct = pickle.load(file)
        temp = [ pair[0] for pair in newp_struct]

        for t in temp:
            newp_struct_list.append(t)

    index = 0
    while os.path.exists(MYPATH + "/" + "rec_new_pairs_from_attr" + str(index) + ".pickle"):
        index += 1

    newp_attr_list = []
    for i in range(0, index):
        with open(MYPATH + "rec_new_pairs_from_attr" + str(i) + ".pickle", "rb") as file:
            newp_attr = pickle.load(file)
        temp = [ pair[0] for pair in newp_attr]

        for t in temp:
            newp_attr_list.append(t)


    print(len(newp_struct_list))
    print(len(newp_attr_list))
    
    c = 0
    cn = 0
    hits_1 = 0
    hits_10 = 0
    MRR = 0
    for i in range(len(ents_1)):

        if ents_1[i] in newp_struct_list or ents_1[i] in newp_attr_list:
            c += 1
            hits_1 +=1
            hits_10 += 1
            MRR += 1
            cn += 1
            continue
        
        rank = (-res_mat_1[i, :]).argsort()
        rank_index = np.where(rank == i)[0][0]
        if rank_index == 0:
            hits_1 +=1
            hits_10 += 1
        elif rank_index < 5:
            hits_10 +=1
        elif rank_index < 10:
            hits_10 += 1

        MRR += 1/ (rank_index + 1)

        if i == rank[0]:
            c += 1
    
    total = len(ents_1)
    print(str(c/total * 100))

    print("Hits@1: " + str(hits_1/total * 100))
    print("Hits@10: " + str(hits_10/total * 100))
    print("MRR: " + str(MRR/total))
    
    print(str(cn) + " out of " + str(len(set(newp_struct_list).union(set(newp_attr_list)))) + " new proposed pairs are correct")

if __name__ == "__main__":
    
    args = sys.argv
    DATASET = args[1]
    MYPATH = args[2]
    mode = args[3]

    calc_hits(DATASET, MYPATH, mode)