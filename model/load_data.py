# coding: utf-8

import io, nltk
import gluonnlp as nlp
from typing import List, Tuple
import mxnet as mx
from nltk import word_tokenize
nltk.download('punkt')


relation_types = [
    "Component-Whole",
    "Component-Whole-Inv",
    "Instrument-Agency",
    "Instrument-Agency-Inv",
    "Member-Collection",
    "Member-Collection-Inv",
    "Cause-Effect",
    "Cause-Effect-Inv",
    "Entity-Destination",
    "Entity-Destination-Inv",
    "Content-Container",
    "Content-Container-Inv",
    "Message-Topic",
    "Message-Topic-Inv",
    "Product-Producer",
    "Product-Producer-Inv",
    "Entity-Origin",
    "Entity-Origin-Inv",
    "Other"
    ]


def load_tsv_to_array(fname: str) -> List[Tuple[str, int, int, str]]:
    """
    load tsv file to array
    :param fname: input file path
    :return: a list of tuples, (relation, entity1 index, entity2 index, data)
    """
    arr = []
    with io.open(fname, 'r') as fp:
        for line in fp:
            els = line.split('\t')
            els[3] = els[3].strip().lower()
            els[2] = int(els[2])
            els[1] = int(els[1])
            arr.append(tuple(els))
    return arr


def load_dataset(file: str, max_length: int = 100) \
        -> Tuple[nlp.Vocab, List[Tuple[str, int, int, List[int]]], 'BasicTransform']:
    """
    parse the input data by getting the word sequence and the argument position ids for entity1 and entity2
    :param file: training file in TSV format. Split the file later. Cross validation
    :param max_length: vocabulary (with attached embedding), training, validation and test datasets
            ready for neural net training
    :return: vocab, dataset, data transform object
    """
    train_array = load_tsv_to_array(file)

    vocabulary = build_vocabulary(train_array)
    dataset = preprocess_dataset(train_array, vocabulary, max_length)
    data_transform = BasicTransform(relation_types, max_length)
    return vocabulary, dataset, data_transform


def tokenize(txt: str) -> List[str]:
    """
    Tokenize an input string.
    """
    return word_tokenize(txt)


def build_vocabulary(array: List[Tuple[str, int, int, List[str]]]) -> nlp.Vocab:
    """
    Inputs: arrays representing the training, validation or test data
    Outputs: vocabulary (Tokenized text as in-place modification of input arrays or returned as new arrays)
    """
    array, tokens = _get_tokens(array)
    counter = nlp.data.count_tokens(tokens)
    vocab = nlp.Vocab(counter)
    return vocab


def _get_tokens(array: Tuple[str, int, int, str]) -> Tuple[List[Tuple[str, int, int, List[str]]], List[str]]:
    """
    an internal function that maps word tokens to indices
    this method also marks the start and end of each entity
    """
    all_tokens = []
    for i, instance in enumerate(array):
        # e1 - entity 1; e2 - entity 2
        label, e1, e2, text = instance
        tokens = text.split(" ")
        # mark start and end of entities
        tokens.insert(e2 + 1, "e2_end")
        tokens.insert(e2, "e2_start")
        tokens.insert(e1 + 1, "e1_end")
        tokens.insert(e1, "e1_start")
        text = ' '.join(tokens)
        tokens = tokenize(text)
        inds = [tokens.index("e1_start")+1, tokens.index("e2_start")+1]
        tokens = [token for token in tokens if token not in["e1_start", "e2_start", "e1_end", "e2_end"]]
        inds[0] = inds[0] - 1
        inds[1] = inds[1] - 3
        array[i] = (label, inds[0], inds[1], tokens)
        all_tokens.extend(tokens)
    return array, all_tokens


def _preprocess(x: Tuple[str, int, int, List[str]], vocab: nlp.Vocab, max_len: int) -> Tuple[str, int, int, List[int]]:
    """
    Inputs: data instance x (tokenized), vocabulary, maximum length of input (in tokens)
    Outputs: data mapped to token IDs, with corresponding label
    """
    label, ind1, ind2, text_tokens = x
    data = vocab[text_tokens]   ## map tokens (strings) to unique IDs
    data = data[:max_len]       ## truncate to max_len

    return label, ind1, ind2, data


def preprocess_dataset(dataset: Tuple[str, int, int, List[str]], vocab: nlp.Vocab, max_len: int)\
        -> Tuple[str, int, int, List[int]]:
    """
    map data to token ids with corresponding labels
    """
    preprocessed_dataset = [ _preprocess(x, vocab, max_len) for x in dataset]
    return preprocessed_dataset


class BasicTransform(object):
    """
    This is a callable object used by the transform method for a dataset. It will be
    called during data loading/iteration.  

    Parameters
    ----------
    labels : list string
        List of the valid strings for classification labels
    max_len : int, default 32
        Maximum sequence length - longer seqs will be truncated and shorter ones padded
    
    """
    def __init__(self, labels: List[str], max_len: int = 32):
        self._max_seq_length = max_len
        self.label_map = {}
        for (i, label) in enumerate(labels):
            self.label_map[label] = i
        self.label_map['?'] = i+1
    
    def __call__(self, label: str, ind1: int, ind2: int, data: str):
        label_id = self.label_map[label]
        padded_data = data + [0] * (self._max_seq_length - len(data))
        inds = mx.nd.array([ind1, ind2])
        return mx.nd.array(padded_data, dtype='int32'), inds, mx.nd.array([label_id], dtype='int32')


if __name__=="__main__":
    load_tsv_to_array("../data/semevalTrain.tsv")
