import torch
import os
import numpy as np
from structure_model.reader.kg_reader import KGDataReader
from structure_model.reader.ea_reader import EADataReader
from structure_model.reader.helper import load_vocab
from structure_model.valid.evaluate_kbc import kbc_predict
from structure_model.valid.evaluate_ea import ea_evaluation
from structure_model.utils.tools import device
from structure_model.Param import CSLS

def entity_alignment_test(args, my_model, logger, csls=CSLS, valid=False):
    
    if valid:
        test_path = args.ea_validation_file
    else:
        test_path = args.ea_ref_file
        
    top_k = [1, 3, 10]
    test_data_reader = EADataReader(vocab_path=os.path.join(args.dataset_root_path, args.dataset, args.vocab_file),
                                    data_path=os.path.join(args.dataset_root_path, args.dataset, test_path))
    test_data = np.array(test_data_reader.ent_pairs)

    kg1_entities = test_data[:, 0]
    kg2_entities = test_data[:, 1]
    this_vocab = load_vocab(os.path.join(args.dataset_root_path, args.dataset, args.vocab_file))
    # add_kg2_entities_index = set()
    # with open(os.path.join(args.dataset_root_path, args.dataset, args.ea_target_triples_file), encoding="utf-8") as f:
    #     this_vocab = load_vocab(os.path.join(args.dataset_root_path, args.dataset, args.vocab_file))
    #     for line in f:
    #         h, r, t = line.strip().split()
    #         if h in this_vocab.keys():
    #             add_kg2_entities_index.add(this_vocab[h])
    #         if t in this_vocab.keys():
    #             add_kg2_entities_index.add(this_vocab[t])
    # add_kg2_entities_index -= set(list(kg2_entities))
    # add_kg2_entities_index = list(add_kg2_entities_index)

    kg1_entities_index = list(kg1_entities)
    # kg2_entities_index = list(kg2_entities) + add_kg2_entities_index
    kg2_entities_index = list(kg2_entities)

    assert len(kg2_entities_index) == len(set(kg2_entities_index))

    my_model.eval()
    with torch.no_grad():
        kg1_entities_ids = torch.LongTensor(kg1_entities_index).to(device)
        kg2_entities_ids = torch.LongTensor(kg2_entities_index).to(device)
        embeds1 = my_model.ele_embedding.lut(kg1_entities_ids)
        embeds2 = my_model.ele_embedding.lut(kg2_entities_ids)
        embeds1 = embeds1.cpu().numpy()
        embeds2 = embeds2.cpu().numpy()
  
    alignment_rest_12, hits1_12, mrr_12, msg, sim_mat, sim_mat_2 = ea_evaluation(embeds1, embeds2, None, top_k, threads_num=4,
                                                             metric='inner', normalize=False, csls_k=csls, accurate=True)

    # logger.info("{}\n".format(msg))
    # _, _, _, msg, sim_mat = ea_evaluation(embeds1, embeds2, None, top_k, threads_num=4,
    #                                                          metric='inner', normalize=False, csls_k=2, accurate=True)

    # print("csls_k=2 : {}\n".format(msg))
    return hits1_12, (sim_mat, sim_mat_2, kg1_entities_ids, kg2_entities_ids, this_vocab)
