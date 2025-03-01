import Levenshtein as lev
import pickle
from sklearn import preprocessing
import numpy.matlib 
import numpy as np
import re
import string
from KnowledgeGraph import *

def ent_dict(path, part):
    counter = 0
    triples = {}
    with open(path, "r") as fp:
        for line in fp:
            counter += 1
            if part == "left" or part == "both":
                if line.split("\t", 2)[0] not in triples.keys():
                    triples[line.split("\t", 2)[0]] = list()
                    triples[line.split("\t", 2)[0]].append(
                        (line.split("\t", 2)[1], line.split("\t", 2)[2].rstrip("\n")))
                else:
                    triples[line.split("\t", 2)[0]].append(
                        (line.split("\t", 2)[1], line.split("\t", 2)[2].rstrip("\n")))
            if part == "right" or part == "both":
                if line.split("\t", 2)[2].rstrip("\n") not in triples.keys():
                    triples[line.split("\t", 2)[2].rstrip("\n")] = list()
                    triples[line.split("\t", 2)[2].rstrip("\n")].append(
                        (line.split("\t", 2)[0], line.split("\t", 2)[1]))
                else:
                    triples[line.split("\t", 2)[2].rstrip("\n")].append(
                        (line.split("\t", 2)[0], line.split("\t", 2)[1]))
    return triples


def ent_dict_attr(path):
    triples = {}
    counter = 0
    with open(path, "r") as fp:
        for line in fp:
            counter += 1
            if line.split("\t", 2)[0] not in triples.keys():
                triples[line.split("\t", 2)[0]] = list()
                triples[line.split("\t", 2)[0]].append(
                    (line.split("\t", 2)[1], line.split("\t", 2)[2].rstrip("\n")))
            else:
                triples[line.split("\t", 2)[0]].append(
                    (line.split("\t", 2)[1], line.split("\t", 2)[2].rstrip("\n")))
    return triples


def num_of_triples(path):
    counter = 0
    with open(path, "r") as fp:
        for line in fp:
            counter += 1
    return counter

def entity_pairs(path):
    ent_dict = {}
    with open(path + "ent_ILLs.txt", "r") as fp:
        for line in fp:
            ent_dict[line.split("\t")[0]] = line.split("\t")[1].rstrip()
    return len(ent_dict.keys())

def splitting(path):
    ent_dict = {}
    with open(path + "sup_ents.txt", "r") as fp:
        for line in fp:
            ent_dict[line.split("\t")[0]] = line.split("\t")[1].rstrip()
    train = len(ent_dict.keys())

    ent_dict = {}
    with open(path + "ref_ents.txt", "r") as fp:
        for line in fp:
            ent_dict[line.split("\t")[0]] = line.split("\t")[1].rstrip()
    test = len(ent_dict.keys())

    ent_dict = {}
    with open(path + "valid_ents.txt", "r") as fp:
        for line in fp:
            ent_dict[line.split("\t")[0]] = line.split("\t")[1].rstrip()
    valid = len(ent_dict.keys())

    return train, valid, test


def weakly_conn_comps(kg_mdi):
    return nx.number_weakly_connected_components(kg_mdi.graph)/kg_mdi.graph.number_of_nodes()

def max_comp(kg_mdi):
    comps = sorted(nx.weakly_connected_components(kg_mdi.graph), key=len)
    max_len = len(comps[-1])/kg_mdi.graph.number_of_nodes()
    return max_len


datasets = "D_W_15K_V1"
for name in ["ICEW_YAGO"]:

    path = "../knowformer_data/" + name +"/"

    print("Dataset: " + name)
    print()

    num_of_rel_triples_1 = num_of_triples(path + "/s_triples.txt")
    num_of_rel_triples_2 = num_of_triples(path + "/t_triples.txt")

    num_of_attr_triples_1 = num_of_triples(path + "/attr_triples_1")
    num_of_attr_triples_2 = num_of_triples(path + "/attr_triples_2")

    ent_dict_1 = ent_dict(path + "/s_triples.txt", "both")
    ent_dict_2 = ent_dict(path + "/t_triples.txt", "both")

    ent_dict_attr_1 = ent_dict_attr(path + "/attr_triples_1")
    ent_dict_attr_2 = ent_dict_attr(path + "/attr_triples_2")

    attr_types = set()
    for id in ent_dict_attr_1:
        for pair in ent_dict_attr_1[id]:
            attr_types.add(pair[0])
    print("#Attribute types kg1: " + str(len(attr_types)))

    attr_types = set()
    for id in ent_dict_attr_2:
        for pair in ent_dict_attr_2[id]:
            attr_types.add(pair[0])
    print("#Attribute types kg2: " + str(len(attr_types)))

    print()

    ent_dict_left_1 = ent_dict(path + "/s_triples.txt", "left")
    ent_dict_left_2 = ent_dict(path + "/t_triples.txt", "left")

    rel_types = set()
    for id in ent_dict_left_1:
        for pair in ent_dict_left_1[id]:
            rel_types.add(pair[0])
    print("#Relation types kg1: " + str(len(rel_types)))

    rel_types = set()
    for id in ent_dict_left_2:
        for pair in ent_dict_left_2[id]:
            rel_types.add(pair[0])
    print("#Relation types kg2: " + str(len(rel_types)))

    ent_dict_attr_left_1 = ent_dict(path + "/attr_triples_1", "left")
    ent_dict_attr_left_2 = ent_dict(path + "/attr_triples_2", "left")

    print()
    print("#Relation triples of kg1: " + str(num_of_rel_triples_1))
    print("#Relation triples of kg2: " + str(num_of_rel_triples_2))
    print()

    print()
    print("#Attribute triples of kg1: " + str(num_of_attr_triples_1))
    print("#Attribute triples of kg2: " + str(num_of_attr_triples_2))
    print()

    # Seed Alignment Size
    print("Seed Alignment Size")
    print(entity_pairs(path))

    print()
    print("Splitting train/test/valid")
    print(splitting(path))
    print()

    kg1_mdi = KnowledgeGraph("1", name, "multi_directed", "original", "original", "RDGCN")
    kg2_mdi = KnowledgeGraph("2", name, "multi_directed", "original", "original", "RDGCN")

    wccR1 = weakly_conn_comps(kg1_mdi)
    wccR2 = weakly_conn_comps(kg2_mdi)

    maxcs1 = max_comp(kg1_mdi)
    maxcs2 = max_comp(kg2_mdi)   
        
    print("wccR KG1: " + str(wccR1))
    print("wccR KG2: " + str(wccR2))
    print("maxCS KG1: " + str(maxcs1))
    print("maxCS KG2: " + str(maxcs2))
    print()