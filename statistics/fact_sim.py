import Levenshtein

def max_lit_sim(l1, l2):
    max_similarities = {}
    
    for word1 in l1:
        max_similarity = 0
        best_match = None
        
        for word2 in l2:
            similarity = Levenshtein.ratio(word1, word2)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_1 = word1
                best_match_2 = word2

    return max_similarity

for dataset in ["ICEW_WIKI", "D_W_15K_V2", "SRPRS_D_W_15K_V1", "SRPRS_D_W_15K_V2", "BBC_DB", "ICEW_WIKI", "ICEW_YAGO", "fr_en", "zh_en", "ja_en"]:
    print(dataset)
    attrs = {}
    for i in range(2):
        with open("../../attribute_data/" + dataset + "/attr_triples" + str(i + 1)) as fp:
            for line in fp:
                ent = line.split("\t")[0]
                lit = line.split("\t")[2].rstrip()
                if ent not in attrs:
                    attrs[ent] = list()
                attrs[ent].append(lit)

    avg_max_sim = 0
    counter = 0
    with open("../../knowformer_data/" + dataset + "/ent_ILLs.txt") as fp:
            for line in fp:
                ent_1 = line.split("\t")[0]
                ent_2 = line.split("\t")[1].rstrip()
                
                if ent_1 in attrs and ent_2 in attrs:
                    ent_1_lits = attrs[ent_1]
                    ent_2_lits = attrs[ent_2]
                    print(ent_1_lits)
                    print(ent_2_lits)
                    print()

                    avg_max_sim += max_lit_sim(ent_1_lits, ent_2_lits)
                    counter += 1
                else:
                    counter += 1
                                 
    print(str(avg_max_sim / counter)) 
    print(str(counter) + "\n")
    exit()