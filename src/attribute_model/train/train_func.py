import torch
import torch.nn as nn
import torch.nn.functional as F
from attribute_model.Param import *
import numpy as np
import time
import pickle
from attribute_model.valid.eval_function import cos_sim_mat_generate,batch_topk,hit_res

def entlist2emb(Model,entids,entid2data,cuda_num, which):
    """
    return basic bert unit output embedding of entities
    """
    batch_sentences = []
    batch_attr_types = []
    
    if which == "1":
        ent2data_types = entid2data[1]
        entid2data = entid2data[0]
        # 341
        input_size = INPUT_SIZE_1
        # result = set()
        # for lst in ent2data_types.values():
        #     for x in lst:
        #         result.add(x)
        # print(len(result))
        # exit()
    elif which == "2":
        ent2data_types = entid2data[3]
        entid2data = entid2data[2]
        # 649
        input_size = INPUT_SIZE_2
        # result = set()
        # for lst in ent2data_types.values():
        #     for x in lst:
        #         result.add(x)
        # print(len(result))
        # exit()
    for eid in entids:
        if eid in entid2data:
            for x in entid2data[eid]:
                batch_sentences.append(x)
            temp_attr_types = ent2data_types[eid]
        else:
            temp_attr_types = []

        batch_attr_types.append(temp_attr_types)
        
    max_len = max(len(lst) for lst in batch_attr_types)
    alpha_i_list = [torch.tensor(alpha_i) for alpha_i in batch_attr_types]
    padded_alpha_i = []
    for x in alpha_i_list:
        padding_values_to_add = torch.full((max_len - len(x),), input_size, dtype=torch.float32)
        padded_alpha_i.append(np.array(torch.cat((x, padding_values_to_add))))
    padded_alpha_i = torch.tensor(np.array(padded_alpha_i)).int().cuda(1)

    batch_emb = Model(batch_attr_types, batch_sentences, padded_alpha_i, which)
    del batch_sentences
    del padded_alpha_i
    del alpha_i_list
    del batch_attr_types
    return batch_emb


def generate_candidate_dict(Model,train_ent1s,train_ent2s,for_candidate_ent1s,for_candidate_ent2s,
                                entid2data,index2entity,
                                nearest_sample_num = NEAREST_SAMPLE_NUM, batch_size = CANDIDATE_GENERATOR_BATCH_SIZE):

    start_time = time.time()
    Model.eval()
    torch.cuda.empty_cache()
    candidate_dict = dict()
    with torch.no_grad():
        print("0")
        #langauge1 (KG1)
        train_emb1 = []
        for_candidate_emb1 = []

        for i in range(0,len(train_ent1s),batch_size):
            temp_emb = entlist2emb(Model,train_ent1s[i:i+batch_size],entid2data,CUDA_NUM, "1").cuda(1).tolist()
            train_emb1.extend(temp_emb)
        for i in range(0,len(for_candidate_ent2s),batch_size):
            temp_emb = entlist2emb(Model,for_candidate_ent2s[i:i+batch_size],entid2data,CUDA_NUM, "2").cuda(1).tolist()
            for_candidate_emb1.extend(temp_emb)
        print("1")
        
        #language2 (KG2)
        train_emb2 = []
        for_candidate_emb2 = []
        for i in range(0,len(train_ent2s),batch_size):
            temp_emb = entlist2emb(Model,train_ent2s[i:i+batch_size],entid2data,CUDA_NUM, "2").cuda(1).tolist()
            train_emb2.extend(temp_emb)
        for i in range(0,len(for_candidate_ent1s),batch_size):
            temp_emb = entlist2emb(Model,for_candidate_ent1s[i:i+batch_size],entid2data,CUDA_NUM, "1").cuda(1).tolist()
            for_candidate_emb2.extend(temp_emb)
        torch.cuda.empty_cache()
        print("2")

        #cos sim
        cos_sim_mat1 = cos_sim_mat_generate(train_emb1,for_candidate_emb1)
        cos_sim_mat2 = cos_sim_mat_generate(train_emb2,for_candidate_emb2)
        torch.cuda.empty_cache()

        #topk index
        _,topk_index_1 = batch_topk(cos_sim_mat1,topn=nearest_sample_num,largest=True)
        topk_index_1 = topk_index_1.tolist()

        _,topk_index_2 = batch_topk(cos_sim_mat2,topn=nearest_sample_num,largest=True)
        topk_index_2 = topk_index_2.tolist()

        #get candidate
        for x in range(len(topk_index_1)):
            e = train_ent1s[x]
            candidate_dict[e] = []
            for y in topk_index_1[x]:
                c = for_candidate_ent2s[y]
                candidate_dict[e].append(c)

        for x in range(len(topk_index_2)):
            e = train_ent2s[x]
            candidate_dict[e] = []
            for y in topk_index_2[x]:
                c = for_candidate_ent1s[y]
                candidate_dict[e].append(c)
        #show
        # def rstr(string):
        #     return string.split(r'/resource/')[-1]
        # for e in train_ent1s[100:105]:
        #     print(rstr(index2entity[e]),"---",[rstr(index2entity[eid]) for eid in candidate_dict[e][:6]])
        # for e in train_ent2s[100:105]:
        #     print(rstr(index2entity[e]),"---",[rstr(index2entity[eid]) for eid in candidate_dict[e][:6]])
    print("get candidate using time: {:.3f}".format(time.time()-start_time))
    torch.cuda.empty_cache()
    return candidate_dict

def train(Model,Criterion,Optimizer,Train_gene,train_ill,test_ill,valid_ill,entid2data):
    print("start training...")
    loss_list = []
    hits_1_list = []
    max_hits1 = 0
    for epoch in range(EPOCH_NUM):
        print("+++++++++++")
        print("Epoch: ",epoch)
        print("+++++++++++")
        #generate candidate_dict
        #(candidate_dict is used to generate negative example for train_ILL)
        train_ent1s = [e1 for e1,e2 in train_ill]
        train_ent2s = [e2 for e1,e2 in train_ill]
        for_candidate_ent1s = Train_gene.ent_ids1
        for_candidate_ent2s = Train_gene.ent_ids2
        print("train ent1s num: {} train ent2s num: {} for_Candidate_ent1s num: {} for_candidate_ent2s num: {}"
              .format(len(train_ent1s),len(train_ent2s),len(for_candidate_ent1s),len(for_candidate_ent2s)))
        candidate_dict = generate_candidate_dict(Model,train_ent1s,train_ent2s,for_candidate_ent1s,
                                                     for_candidate_ent2s,entid2data,Train_gene.index2entity)

        Train_gene.train_index_gene(candidate_dict) #generate training data with candidate_dict
        #train
        epoch_loss,epoch_train_time = ent_align_train(Model,Criterion,Optimizer,Train_gene,entid2data)
        loss_list.append(epoch_loss)
        Optimizer.zero_grad()
        torch.cuda.empty_cache()
        print("Epoch {}: loss {:.3f}, using time {:.3f}".format(epoch,epoch_loss,epoch_train_time))
        if epoch % 5 == 0:
            _, _, hits_1 = test(Model, valid_ill, entid2data, TEST_BATCH_SIZE, context="EVAL IN VALID SET:", second_mat = False, csls=CSLS)
            hits_1_list.append(hits_1)
                
            if hits_1 > max_hits1:
                max_hits1 = hits_1
                times = 0
            else:
                times += 1
            if times >= 3 and epoch >= 5:
                print("early stop at this epoch")
                break
            
    _, _, _, = test(Model, test_ill, entid2data, TEST_BATCH_SIZE, context="EVAL IN TEST SET WITHOUT CSLS:", second_mat = False, csls=0) 
    ent_ill, res_mat, res_mat_2, hits_1 = test(Model, test_ill, entid2data, TEST_BATCH_SIZE, context="EVAL IN TEST SET WITH CSLS:", second_mat = True, csls=CSLS)             
    return ent_ill, res_mat, res_mat_2, loss_list, hits_1_list


def test(Model,ent_ill,entid2data,batch_size,context = "", second_mat=False, csls=CSLS):
    print("-----test start-----")
    start_time = time.time()
    print(context)
    Model.eval()
    with torch.no_grad():
        ents_1 = [e1 for e1,e2 in ent_ill]
        ents_2 = [e2 for e1,e2 in ent_ill]

        emb1 = []
        for i in range(0,len(ents_1),batch_size):
            batch_ents_1 = ents_1[i: i+batch_size]
            batch_emb_1 = entlist2emb(Model,batch_ents_1,entid2data,CUDA_NUM, "1").detach().cuda(1).tolist()
            emb1.extend(batch_emb_1)
            del batch_emb_1

        emb2 = []
        for i in range(0,len(ents_2),batch_size):
            batch_ents_2 = ents_2[i: i+batch_size]
            batch_emb_2 = entlist2emb(Model,batch_ents_2,entid2data,CUDA_NUM, "2").detach().cuda(1).tolist()
            emb2.extend(batch_emb_2)
            del batch_emb_2

        print("Cosine similarity of embedding res:")
        res_mat = cos_sim_mat_generate(emb1,emb2,batch_size,cuda_num=CUDA_NUM, csls=csls)
        score,top_index = batch_topk(res_mat,batch_size,topn = TOPK,largest=True,cuda_num=CUDA_NUM)
        hits_1 = hit_res(top_index)
        if second_mat:
            res_mat_2 = cos_sim_mat_generate(emb2,emb1,batch_size,cuda_num=CUDA_NUM, csls=csls)
            print("test using time: {:.3f}".format(time.time()-start_time))
            print("--------------------")
            return ent_ill, res_mat, res_mat_2, hits_1
        
    print("test using time: {:.3f}".format(time.time()-start_time))
    print("--------------------")
    return ent_ill, res_mat, hits_1

def ent_align_train(Model,Criterion,Optimizer,Train_gene,entid2data):
    start_time = time.time()
    all_loss = 0
    Model.train()
    for pe1s, pe2s, ne1s, ne2s in Train_gene:
        Optimizer.zero_grad()
        pos_emb1 = entlist2emb(Model,pe1s,entid2data,CUDA_NUM, "1")
        pos_emb2 = entlist2emb(Model,pe2s,entid2data,CUDA_NUM, "2")
        batch_length = pos_emb1.shape[0]
        pos_score = F.pairwise_distance(pos_emb1,pos_emb2,p=2,keepdim=True)
        del pos_emb1
        del pos_emb2

        neg_emb1 = entlist2emb(Model,ne1s,entid2data,CUDA_NUM, "1")
        neg_emb2 = entlist2emb(Model,ne2s,entid2data,CUDA_NUM, "2")
        neg_score = F.pairwise_distance(neg_emb1,neg_emb2,p=2,keepdim=True)
        del neg_emb1
        del neg_emb2

        sum1 = torch.sum(0.2 * pos_score)
        sum2 = torch.sum(0.8 * torch.clamp(3.0 - neg_score, min=0.0))


        batch_loss = sum1 + sum2
    
        batch_loss.backward()
        Optimizer.step()

        all_loss += batch_loss.item()

    all_using_time = time.time() - start_time
    return np.mean(all_loss), all_using_time

# def ent_align_train(Model,Criterion,Optimizer,Train_gene,entid2data):
#     start_time = time.time()
#     all_loss = 0
#     Model.train()
#     for pe1s,pe2s,ne1s,ne2s in Train_gene:
#         Optimizer.zero_grad()
#         pos_emb1 = entlist2emb(Model,pe1s,entid2data,CUDA_NUM, "1")
#         pos_emb2 = entlist2emb(Model,pe2s,entid2data,CUDA_NUM, "2")
#         batch_length = pos_emb1.shape[0]
#         pos_score = F.pairwise_distance(pos_emb1,pos_emb2,p=1,keepdim=True)#L1 distance
#         del pos_emb1
#         del pos_emb2

#         neg_emb1 = entlist2emb(Model,ne1s,entid2data,CUDA_NUM, "1")
#         neg_emb2 = entlist2emb(Model,ne2s,entid2data,CUDA_NUM, "2")
#         neg_score = F.pairwise_distance(neg_emb1,neg_emb2,p=1,keepdim=True)
#         del neg_emb1
#         del neg_emb2

#         label_y = -torch.ones(pos_score.shape).cuda(CUDA_NUM) #pos_score < neg_score
#         batch_loss = Criterion( pos_score , neg_score , label_y )
#         del pos_score
#         del neg_score
#         del label_y
#         batch_loss.backward()
#         Optimizer.step()

#         all_loss += batch_loss.item() * batch_length
#     all_using_time = time.time()-start_time
#     return all_loss,all_using_time