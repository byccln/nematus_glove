'''
Utility functions
'''

import pickle as pkl
import exception
import json
import logging
import numpy
import random
import sys
import copy
import hashlib
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Source:
# https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
def get_available_gpus():
    """
        Returns a list of the identifiers of all visible GPUs.
    """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

#add
#获取tsne
def get_tsne(vectors, savepath):
    tsne = TSNE(n_components=2, init='pca', verbose=1).fit_transform(vectors)
    numpy.save(savepath, tsne)
    return tsne

#绘制预训练词向量和glove词向量的二维图
def plot_tsne(vocab, embed, gvocab, gembed, top=20):
    plt.rcParams['font.sans-serif']=['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.figure()
    #先绘制预训练词向量
    plt.scatter(embed[:top, 0], embed[:top, 1], c='blue', marker='o')
    for i in range(top):
        x = embed[i][0]
        y = embed[i][1]
        plt.text(x, y, vocab[i])
    #绘制glove词向量
    xx, yy = [], []
    for token in vocab[:top]:
        ids = gvocab[token]
        xx.append(gembed[ids][0])
        yy.append(gembed[ids][1])
    plt.scatter(xx, yy, c='yellow', marker='v')
    for i in range(top):
        plt.text(xx[i], yy[i], vocab[i])
    plt.legend(["pretrain", "glove"])
    plt.show()

#add
def pre_load_data(vocabpath, vectorspath):
    vocab = json.load(open(vocabpath, "r"))
    vectors = numpy.load(vectorspath)
    return vocab, vectors

#add
def token_convert_to_utf8(token):
    return [u for char in token for u in hex(ord(char))[2:]]

def get_md5(chars):
    md5 = hashlib.md5()
    md5.update(chars.encode('utf-8'))
    csl = md5.hexdigest()
    return [u for u in csl]

#add
def get_data(vocab_list, batch_size, vocab, vectors, utf8_dict, shuffle=True):
    #vocab_list = copy.deepcopy(pre_vocab_list)
    if shuffle:
        random.shuffle(vocab_list)
    #将词表按batch_size分割
    split_n = len(vocab_list) // batch_size
    source_x = [vocab_list[i*batch_size:i*batch_size+batch_size] for i in range(split_n)]
    source_y = [[vectors[vocab[tok]] for tok in vocab_list[i*batch_size:i*batch_size+batch_size]] for i in range(split_n)]
    if len(vocab_list) % batch_size:
        source_x.append(vocab_list[split_n*batch_size:])
        source_y.append([vectors[vocab[tok]] for tok in vocab_list[split_n*batch_size:]])
    #把source_x变为id
    #source_xx = [[[utf8_dict[u] for u in token_convert_to_utf8(x)] for x in xx] for xx in source_x]
    source_xx = [[[utf8_dict[u] for u in get_md5(x)] for x in xx] for xx in source_x]
    return source_xx, source_y, vocab_list

#add
def pre_prepare_data(seqs_x, seqs_y): #(batch_size, u_len)
    lengths_x = [len(s) for s in seqs_x]
    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x)
    x = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    y = numpy.array(seqs_y).astype('float32')
    for idx, s_x in enumerate(seqs_x):
        x[idx, :lengths_x[idx]] = s_x
    return x, y   

# batch preparation
def prepare_data(seqs_x, seqs_y, n_factors, seqs_px=None, maxlen=None):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x] #[15, 12, 10, 20]
    lengths_xx = [[len(ss) for ss in s] for s in seqs_x] #[[...](15), [...](12), ...]
    lengths_y = [len(s) for s in seqs_y]

    n_samples = len(seqs_x) #batch_size
    maxlen_x = numpy.max(lengths_x) + 1 #最长句子长度
    maxlen_xx = max(map(max, lengths_xx))
    maxlen_y = numpy.max(lengths_y) + 1

    if seqs_px is not None:
        px = numpy.zeros((n_factors, maxlen_x, n_samples)).astype('int64')
    x = numpy.zeros((n_factors, maxlen_x, maxlen_xx, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, maxlen_xx, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        if seqs_px is not None:
            px[:, :lengths_x[idx], idx] = list(zip(*(seqs_px[idx])))
        for j in range(lengths_x[idx]):
            x[:, j, :lengths_xx[idx][j], idx] = s_x[j]
            x_mask[j, :lengths_xx[idx][j], idx] = 1.
        x_mask[j+1, :, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    if seqs_px is not None:
        return px, x, x_mask, y, y_mask
    return x, x_mask, y, y_mask


def load_dict(filename, model_type):
    try:
        # build_dictionary.py writes JSON files as UTF-8 so assume that here.
        with open(filename, 'r', encoding='utf-8') as f:
            d = json.load(f)
    except:
        # FIXME Should we be assuming UTF-8?
        with open(filename, 'r', encoding='utf-8') as f:
            d = pkl.load(f)

    # The transformer model requires vocab dictionaries to use the new style
    # special symbols. If the dictionary looks like an old one then tell the
    # user to update it.
    if model_type == 'transformer' and ("<GO>" not in d or d["<GO>"] != 1):
        logging.error('you must update \'{}\' for use with the '
                      '\'transformer\' model type. Please re-run '
                      'build_dictionary.py to generate a new vocabulary '
                      'dictionary.'.format(filename))
        sys.exit(1)

    return d


def seq2words(seq, inverse_dictionary, join=True):
    seq = numpy.array(seq, dtype='int64')
    assert len(seq.shape) == 1
    return factoredseq2words(seq.reshape([seq.shape[0], 1]),
                             [inverse_dictionary],
                             join)

def factoredseq2words(seq, inverse_dictionaries, join=True):
    assert len(seq.shape) == 2
    assert len(inverse_dictionaries) == seq.shape[1]
    words = []
    eos_reached = False
    for i, w in enumerate(seq):
        if eos_reached:
            break
        factors = []
        for j, f in enumerate(w):
            if f == 0:
                eos_reached = True
                break
                # This assert has been commented out because it's possible for
                # non-zero values to follow zero values for Transformer models.
                # TODO Check why this happens
                #assert (i == len(seq) - 1) or (seq[i+1][j] == 0), \
                #       ('Zero not at the end of sequence', seq)
            elif f in inverse_dictionaries[j]:
                factors.append(inverse_dictionaries[j][f])
            else:
                factors.append('UNK')
        word = '|'.join(factors)
        words.append(word)
    return ' '.join(words) if join else words

def reverse_dict(dictt):
    keys, values = list(zip(*list(dictt.items())))
    r_dictt = dict(list(zip(values, keys)))
    return r_dictt


def load_dictionaries(config):
    model_type = config.model_type
    source_to_num = [load_dict(d, model_type) for d in config.source_dicts]
    target_to_num = load_dict(config.target_dict, model_type)
    num_to_source = [reverse_dict(d) for d in source_to_num]
    num_to_target = reverse_dict(target_to_num)
    return source_to_num, target_to_num, num_to_source, num_to_target

def read_all_lines(config, sentences, batch_size):
    source_to_num, _, _, _ = load_dictionaries(config)

    with open(config.pretrain_vocab, 'r', encoding='utf-8') as f:
        pretrain_vocab = json.load(f)

    if config.source_vocab_sizes != None:
        assert len(config.source_vocab_sizes) == len(source_to_num)
        for d, vocab_size in zip(source_to_num, config.source_vocab_sizes):
            if vocab_size != None and vocab_size > 0:
                for key, idx in list(d.items()):
                    if idx >= vocab_size:
                        del d[key]

    prelines, lines = [], []
    for sent in sentences:
        preline, line = [], []
        for w in sent.strip().split():
            if config.utf8_type == 'utf8':
                ww = token_convert_to_utf8(w)
            else:
                ww = get_md5(w)
            if config.factors == 1:
                ww = [source_to_num[0][c] for c in ww]
                pw = [pretrain_vocab[w] if w in pretrain_vocab else pretrain_vocab['<unk>']]
            line.append(ww)
            preline.append(pw)
        lines.append(line)
        prelines.append(preline)
    lines = numpy.array(lines)
    prelines = numpy.array(prelines)
    lengths = numpy.array([len(l) for l in lines])
    idxs = lengths.argsort()
    lines = lines[idxs]
    prelines = prelines[idxs]

    #merge into batches
    prebatches, batches = [], []
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i+batch_size]
        batches.append(batch)
        prebatch = prelines[i:i+batch_size]
        prebatches.append(prebatch)

    return prebatches, batches, idxs
