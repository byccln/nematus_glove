#!/bin/env python3

import numpy
import json

def pre_load_data(vector_path):
    #load data
    source_data = open(vector_path+"/vectors.txt", "r").readlines()
    source_data_dict = {}
    vocab = {}
    emb_list = []
    for idx, line in enumerate(source_data):
        line = line.strip().split(' ')
        vocab[line[0]] = idx
        emb_list.append([x for x in line[1:]])
    emb = numpy.array(emb_list).astype('float32')
    numpy.save(vector_path+"/vectors.npy", emb)
    with open(vector_path+"/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

path="/media/ntfs-3/EXP/MULTI/mix/zh-en/data3/glove"
pre_load_data(path)
