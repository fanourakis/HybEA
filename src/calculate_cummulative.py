import os
import pickle
import sys

def measures(DATASET, model, iteration, MYPATH):
    
    test_set = []
    with open("../knowformer_data/" + DATASET + "/ref_ents.txt", "r") as fp:
        for line in fp:
            e1 = line.split("\t")[0]
            e2 = line.split("\t")[1].rstrip()
            test_set.append((e1, e2))
    
    ents_1 = [e1 for e1,e2 in test_set]
    ents_2 = [e2 for e1,e2 in test_set]
    
    with open(MYPATH + "rec_new_pairs_from_" + model + str(iteration) + ".pickle", "rb") as file:
            newp= pickle.load(file)
    newp_list = [ pair[0] for pair in newp]
    
    e1_proposed_pairs = set(newp_list)
    correct_proposed_pairs = e1_proposed_pairs.intersection(set(ents_1))
    print(len(e1_proposed_pairs))
    print(len(ents_1))
    fp = set()
    for e1 in e1_proposed_pairs:
        if e1 not in ents_1:
            fp.add(e1)
    
    tn = set()
    fn = set()
    e1_not_proposed= set(ents_1) - set(e1_proposed_pairs)
    for e1 in e1_not_proposed:
        if e1 not in ents_1:
            tn.add(e1)
        elif e1 in ents_1:
            fn.add(e1)

    # print("tp: " + str(len(correct_proposed_pairs)))
    # print("fp: " + str(len(fp)))
    # print("tn: " + str(len(tn)))
    # print("fn: " + str(len(fn)))
    
    precision = len(correct_proposed_pairs) / (len(correct_proposed_pairs) + len(fp))
    recall = len(correct_proposed_pairs) / (len(correct_proposed_pairs) + len(fn))
    f1_score = (2 * precision * recall) / (precision + recall)

    # print("\nprecision: " + str(precision))
    # print("recall: " + str(recall))
    # print("f1_score: " + str(f1_score))
    
    return set(e1_proposed_pairs)
 
def calc_measures(DATASET, MYPATH):

    cpr = 0
    crec = 0

    test_set = []
    with open("../knowformer_data/" + DATASET + "/ref_ents.txt", "r") as fp:
        for line in fp:
            e1 = line.split("\t")[0]
            e2 = line.split("\t")[1].rstrip()
            test_set.append((e1, e2))
    
    ents_1 = [e1 for e1,e2 in test_set]
    ents_2 = [e2 for e1,e2 in test_set]
    
    models = ["structure", "attr"]
    for ch_model in models:
        e1_pr_pairs = set()

        index = 0
        while os.path.exists(MYPATH + "/" + "rec_new_pairs_from_" + ch_model + str(index) + ".pickle"):
            index += 1
        print(index)

        print("Model: " + ch_model)
        if ch_model == "attr" && (DATASET == "ICEW_WIKI" or DATASET == "ICEW_YAGO)":
            dif = 1
        else:
            dif = 0
        for i in range(0, index - dif):
            # print("iteration: " + str(i))
            iter_new_pairs = measures(DATASET, ch_model, i, MYPATH)
            e1_pr_pairs = e1_pr_pairs.union(iter_new_pairs)
            # print("-------")
            
        tp = e1_pr_pairs.intersection(set(ents_1))
        fp = set()
        for e1 in e1_pr_pairs:
            if e1 not in ents_1:
                fp.add(e1)
        
        tn = set()
        fn = set()
        e1_not_pr_pairs= set(ents_1) - set(e1_pr_pairs)
        for e1 in e1_not_pr_pairs:
            if e1 not in ents_1:
                tn.add(e1)
            elif e1 in ents_1:
                fn.add(e1)

        # print("tp: " + str(len(tp)))
        # print("fp: " + str(len(fp)))
        # print("tn: " + str(len(tn)))
        # print("fn: " + str(len(fn)))
        # if len(tp) == 0 and len(fp) == 0:
        #     print("no proposed pairs")
        #     continue
        precision = len(tp) / (len(tp) + len(fp))
        cpr += precision
        recall = len(tp) / (len(tp) + len(fn))
        f1_score = (2 * precision * recall) / (precision + recall)
        crec += recall
        print("\nprecision: " + str(precision))
        print("recall: " + str(recall))
        print("f1_score: " + str(f1_score))
        print()
    cpr = cpr / 2
    print("cummulative")
    print("\nprecision: " + str(cpr))
    print("recall: " + str(crec))
    print("f1_score: " + str((2 * cpr * crec) / (cpr + crec)))

if __name__ == "__main__":
    
    args = sys.argv
    DATASET = args[1]
    MYPATH = args[2]

    calc_measures(DATASET, MYPATH)