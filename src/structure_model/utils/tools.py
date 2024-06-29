import torch
import torch.nn as nn
from scipy.stats import truncnorm
from rich.console import Console
from rich.table import Table
import numpy as np
from structure_model.reader.helper import load_vocab
import os
from sentence_transformers import SentenceTransformer
import pandas as pd

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    # Set the GPU device number (assuming you want to use GPU device 0)
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
else:
    # If CUDA is not available, use CPU
    device = torch.device('cpu')


def truncated_normal(size, threshold=0.02):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values


def truncated_normal_init(x, size, initializer_range):
    """
    Init a module x, init x.weight with truncated normal, and init x.bias with 0 if x has bias.

    Args:
        x: a pytorch module, it has attr weight or bias, we init the weight tensor or bias tensor with truncated normal.
        size: a list, depicts the shape of x.weight
        initializer_range: the range for truncated normal.

    Returns:
        None
    """
    x.weight.data.copy_(
        torch.from_numpy(truncated_normal(size, initializer_range)))
    if hasattr(x, "bias"):
        nn.init.constant_(x.bias, 0.0)
        
def generate_attr_dict(attr):
    result_dict = {}

    for index, row in attr.iterrows():
        key = row['ent']
        value = row['val']
        value = str(value)

        if key in result_dict:
            result_dict[key].append(value)
        else:
            result_dict[key] = [value]
    
    return result_dict
        
def name_init(x, size, initializer_range, kg, args):
    
    ids1 = pd.read_csv("../knowformer_data/"+ args.dataset +"/ent_ids_1", sep="\t", header=None, names=["id", "uri"])
    uris_to_ids1 = dict(zip(ids1["uri"], ids1["id"]))
    
    ids2 = pd.read_csv("../knowformer_data//"+ args.dataset +"/ent_ids_2", sep="\t", header=None, names=["id", "uri"])
    uris_to_ids2 = dict(zip(ids2["uri"], ids2["id"]))
    
    names1 = pd.read_excel('../generate_names/entity_names/' + args.dataset + '/' + args.KG1_PATH_FOR_NAMES, index_col=0)
    ids_to_names1 = dict(zip(names1["e1"], names1["name"]))
    
    names2 = pd.read_excel('../generate_names/entity_names/' + args.dataset + '/' + args.KG2_PATH_FOR_NAMES, index_col=0)
    ids_to_names2 = dict(zip(names2["e1"], names2["name"]))

    this_vocab = load_vocab(os.path.join(args.dataset_root_path, args.dataset, args.vocab_file))
    
    sentences = []
    for k in this_vocab:

        # for D_W_15K_V1 27100
        # imdb-tmdb 3580
        # bbc-db 17013
        if this_vocab[k] > 99 and this_vocab[k] < args.VOC_LIM_2:
            uri = k
            if uri in uris_to_ids1.keys():
                id = uris_to_ids1[uri]
                name = ids_to_names1[id]
            elif uri in uris_to_ids2.keys():
                id = uris_to_ids2[uri]
                name = ids_to_names2[id]
                
            if name != "no_value":
                sentences.append(name)
                
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = model.encode(sentences)
    
    
    name_embs = []
    index = 0
    for k in this_vocab:
        if this_vocab[k] < 100:
            name_embs.append(truncated_normal(768, initializer_range))
        
        # for D_W_15K_V1 15100
        # imdb-tmdb 2033
        # bbc_db 9496
        elif this_vocab[k] < args.VOC_LIM_1:
            uri = k
            id = uris_to_ids1[k]
            name = ids_to_names1[id]
            if name != "no_value":
                name_embs.append(embeddings[index])
                index = index + 1
            else:
                name_embs.append(truncated_normal(768, initializer_range))
        
        # for D_W_15K_V1 27100
        # imdb-tmdb 3580
        # bbc_db 17013
        elif this_vocab[k] < args.VOC_LIM_2:
            uri = k
            id = uris_to_ids2[k]
            name = ids_to_names2[id]
            if name != "no_value":
                name_embs.append(embeddings[index])
                index = index + 1
            else:
                name_embs.append(truncated_normal(768, initializer_range))
        else:
                name_embs.append(truncated_normal(768, initializer_range))
    x.weight.data.copy_(
        torch.from_numpy(np.array(name_embs)))


def norm_layer_init(x):
    """
    Init a norm layer x, init x.weight with 1, and init x.bias with 0.

    Args:
        x: nn.LayerNorm.

    Returns:
        None
    """
    nn.init.constant_(x.weight, 1.0)  # init
    nn.init.constant_(x.bias, 0.0)  # init


def print_results(dataset_name, eval_performance, k=3):
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Dataset", justify="center")
    table.add_column("MRR", justify="center")
    table.add_column("Hits@1", justify="center")
    table.add_column("Hits@3", justify="center")
    table.add_column("Hits@10", justify="center")
    table.add_row(dataset_name, str(round(eval_performance['fmrr'], k)), str(round(eval_performance['fhits1'], k)),
                  str(round(eval_performance['fhits3'], k)), str(round(eval_performance['fhits10'], k)))
    console.print(table)
