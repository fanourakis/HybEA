import os
from structure_model.reader.helper import read_triples
from structure_model.reader.helper import write_triples
from structure_model.reader.helper import read_entity_paris
from structure_model.reader.helper import write_vocab_path
from structure_model.reader.helper import load_vocab


def prepare_entity_alignment_data(args, new_pairs):
    sup_ents_path = os.path.join(args.dataset_root_path, args.dataset, "sup_ents.txt")
    ref_ents_path = os.path.join(args.dataset_root_path, args.dataset, args.ea_ref_file)
    ent_ILLs_path = os.path.join(args.dataset_root_path, args.dataset, args.ea_all_file)
    s_triples_path = os.path.join(args.dataset_root_path, args.dataset, args.ea_source_triples_file)
    t_triples_path = os.path.join(args.dataset_root_path, args.dataset, args.ea_target_triples_file)
    train_triples_path = os.path.join(args.dataset_root_path, args.dataset, "train.triples.txt")
    vocab_path = os.path.join(args.dataset_root_path, args.dataset, args.vocab_file)
    assert os.path.exists(s_triples_path) and os.path.exists(t_triples_path) and os.path.exists(
        ref_ents_path) and os.path.exists(ent_ILLs_path)


    if len(new_pairs) > 0:
        
        sup_new_ents_path = os.path.join(args.dataset_root_path, args.dataset, "sup_new_ents.txt")
        train_new_triples_path = os.path.join(args.dataset_root_path, args.dataset, "train.new.triples.txt")

        sup_set = []
        with open(sup_ents_path, "r") as fp:
            for line in fp:
                sup_set.append((line.split("\t")[0], line .split("\t")[1].rstrip()))

        for pair in new_pairs:
            sup_set.append(pair)

        with open(sup_new_ents_path, "w") as fp:
            for pair in sup_set:
                fp.write(pair[0])
                fp.write("\t")
                fp.write(pair[1])
                fp.write("\n")

        def merge(train_new_triples_path_, s_triples_path_, t_triples_path_, sup_new_ents_path):
            s_triples = read_triples(s_triples_path_)
            t_triples = read_triples(t_triples_path_)
            sup_ents = read_entity_paris(sup_new_ents_path)
            t_s_map = {}
            for s, t in sup_ents:
                assert t not in t_s_map.keys()
                t_s_map[t] = s
            train_triples = s_triples.copy()
            for t_triple in t_triples:
                t_triple_h, t_triple_r, t_triple_t = t_triple
                t_triple_h = t_s_map.get(t_triple_h, t_triple_h)
                t_triple_t = t_s_map.get(t_triple_t, t_triple_t)
                train_triples.append((t_triple_h, t_triple_r, t_triple_t))
            assert len(train_triples) == len(s_triples) + len(t_triples)
            write_triples(train_new_triples_path_, train_triples, is_add_mask=True)

        merge(train_new_triples_path, s_triples_path, t_triples_path, sup_new_ents_path)

    else:

        if not os.path.exists(train_triples_path):
            def merge(train_triples_path_, s_triples_path_, t_triples_path_, sup_ents_path_):
                s_triples = read_triples(s_triples_path_)
                t_triples = read_triples(t_triples_path_)
                sup_ents = read_entity_paris(sup_ents_path_)
                t_s_map = {}
                for s, t in sup_ents:
                    assert t not in t_s_map.keys()
                    t_s_map[t] = s
                train_triples = s_triples.copy()
                for t_triple in t_triples:
                    t_triple_h, t_triple_r, t_triple_t = t_triple
                    t_triple_h = t_s_map.get(t_triple_h, t_triple_h)
                    t_triple_t = t_s_map.get(t_triple_t, t_triple_t)
                    train_triples.append((t_triple_h, t_triple_r, t_triple_t))
                assert len(train_triples) == len(s_triples) + len(t_triples)
                write_triples(train_triples_path_, train_triples, is_add_mask=True)

            merge(train_triples_path, s_triples_path, t_triples_path, sup_ents_path)

    entities_set = set()
    relations_set = set()
    # if len(new_pairs) > 0:
    #     train_triples = read_triples(train_new_triples_path)
    # else:
    
    train_triples = read_triples(train_triples_path)
    for triple in train_triples:
        for i, label in enumerate(triple):

            if label.startswith("MASK"):
                continue
            if i % 2 == 0:
                entities_set.add(label)
            else:
                relations_set.add(label)

    def custom_sort_key(entity):
        if 'http' in entity:
            return 0
        elif 'dbp:' in entity:
            return 1
        else:
            return 2
        
    def custom_sort_key_fr_en(entity):
        if 'http://fr.dbpedia' in entity:
            return 0
        elif 'http://dbpedia.org' in entity:
            return 1
        else:
            return 2
        
    def custom_sort_key_ja_en(entity):
        if 'http://ja.dbpedia' in entity:
            return 0
        elif 'http://dbpedia.org' in entity:
            return 1
        else:
            return 2
        
    def custom_sort_key_zh_en(entity):
        if 'http://zh.dbpedia' in entity:
            return 0
        elif 'http://dbpedia.org' in entity:
            return 1
        else:
            return 2

    entities_list = sorted(list(entities_set))
    relations_list = sorted(list(relations_set))

    if args.dataset == "BBC_DB":
        entities_list = sorted(list(entities_set), key=custom_sort_key)
        relations_list = sorted(list(relations_set), key=custom_sort_key)
        
    if args.dataset == "fr_en":
        entities_list = sorted(list(entities_set), key=custom_sort_key_fr_en)
        relations_list = sorted(list(relations_set), key=custom_sort_key_fr_en)
        
    if args.dataset == "ja_en":
        entities_list = sorted(list(entities_set), key=custom_sort_key_ja_en)
        relations_list = sorted(list(relations_set), key=custom_sort_key_ja_en)
        
    if args.dataset == "zh_en":
        entities_list = sorted(list(entities_set), key=custom_sort_key_zh_en)
        relations_list = sorted(list(relations_set), key=custom_sort_key_zh_en)

    args.vocab_size = 100 + len(entities_list) + len(relations_list)
    args.num_relations = len(relations_list)
    if not os.path.exists(vocab_path):
        write_vocab_path(vocab_path, entities_list, relations_list)
    else:
        assert args.vocab_size == len(load_vocab(vocab_path))