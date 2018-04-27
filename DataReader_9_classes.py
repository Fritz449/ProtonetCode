from typing import Dict, List, Sequence, Iterable
import itertools
import logging

from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _is_divider(line: str) -> bool:
    line = line.strip()
    return not line or line == """-DOCSTART- -X- -X- O"""


_VALID_LABELS = {'ner', 'pos', 'chunk'}

import os
import json
from glob import glob
import ftplib
import numpy as np
import random
import urllib.request
import tarfile
import re
import pandas as pd
import copy

SNIPS_URL = 'http://share.ipavlov.mipt.ru:8080/repository/datasets/ner/SNIPS2017.tar.gz'


def tokenize(s):
    return re.findall(r"[\w']+|[‑–—“”€№…’\"#$%&\'()+,-./:;<>?]", s)


def snips_reader(file='train', dataset_download_path='ontonotes/', return_intent=False):
    # param: dataset_download_path - path to the existing dataset or if there is no
    #   dataset there the dataset it will be downloaded to this path
    sentences = []
    ys = []
    with open(dataset_download_path + file, "r") as data_file:
        for is_divider, lines in itertools.groupby(data_file, _is_divider):
            # Ignore the divider chunks, so that `lines` corresponds to the words
            # of a single sentence.
            if not is_divider:
                fields = [line.strip().split() for line in lines]
                tokens, ner_tags = [list(field) for field in zip(*fields)]
                sentences.append([[token, tag] for token, tag in zip(tokens, ner_tags)])
                ys += ner_tags

    random_seed = 1
    np.random.seed(random_seed)
    np.random.shuffle(sentences)
    valid_classes = ['LANGUAGE', 'PRODUCT', 'QUANTITY', 'FAC', 'LOC', 'PERCENT', 'MONEY', 'GPE', 'PERSON']
    train_classes = ['LAW', 'ORDINAL', 'EVENT', 'TIME', 'WORK_OF_ART', 'NORP', 'CARDINAL', 'DATE', 'ORG']
    if file == 'train.txt':
        classes_here = train_classes
    else:
        classes_here = valid_classes

    total_sentences = 100
    train_sentences = 50
    support_sentences = sentences[:(len(sentences) // 2)]
    query_sentences = sentences[(len(sentences) // 2):]

    super_list = []
    for sent in sentences:
        for word in sent:
            if word[1][2:] not in classes_here:
                word[1] = 'O'

    if file != 'train.txt':
        train = []
        validation = []
        ys_sup = np.array([])
        for sentence in sentences:
            ys_here = [xy[1] for xy in sentence]
            if len(train) < train_sentences:
                train.append(sentence)
                ys_sup = np.unique(np.concatenate((ys_sup, np.unique(ys_here))))
            else:
                if ys_sup.shape[0] < np.unique(np.concatenate((ys_sup, np.unique(ys_here)))).shape[0]:
                    pass
                else:
                    validation.append(sentence)

        for i in range(len(validation) // 50):
            if (i + 1) * 50 > len(validation):
                break
            super_list += copy.deepcopy(train + validation[i * 50:(i + 1) * 50])
    else:
        n_episodes = 100
        super_list = []
        for episode in range(n_episodes):
            np.random.seed(episode)
            query = []
            support = np.random.choice(support_sentences, size=train_sentences, replace=False).tolist()
            ys_here = np.concatenate([[xy[1] for xy in sentence] for sentence in support])
            ys_sup = np.unique(ys_here)

            np.random.shuffle(query_sentences)
            for sentence in query_sentences:
                ys_here = [xy[1] for xy in sentence]
                if all([y in ys_sup for y in ys_here]):
                    query.append(sentence)
                if len(query) == total_sentences - train_sentences:
                    break
            super_list += support + query
    return super_list


@DatasetReader.register("onto_pnet")
class PnetOntoDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    WORD POS-TAG CHUNK-TAG NER-TAG

    with a blank line indicating the end of each sentence
    and '-DOCSTART- -X- -X- O' indicating the end of each article,
    and converts it into a ``Dataset`` suitable for sequence tagging.

    Each ``Instance`` contains the words in the ``"tokens"`` ``TextField``.
    The values corresponding to the ``tag_label``
    values will get loaded into the ``"tags"`` ``SequenceLabelField``.
    And if you specify any ``feature_labels`` (you probably shouldn't),
    the corresponding values will get loaded into their own ``SequenceLabelField`` s.

    This dataset reader ignores the "article" divisions and simply treats
    each sentence as an independent ``Instance``. (Technically the reader splits sentences
    on any combination of blank lines and "DOCSTART" tags; in particular, it does the right
    thing on well formed inputs.)

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    tag_label: ``str``, optional (default=``ner``)
        Specify `ner`, `pos`, or `chunk` to have that tag loaded into the instance field `tag`.
    feature_labels: ``Sequence[str]``, optional (default=``()``)
        These labels will be loaded as features into the corresponding instance fields:
        ``pos`` -> ``pos_tags``, ``chunk`` -> ``chunk_tags``, ``ner`` -> ``ner_tags``
        Each will have its own namespace: ``pos_labels``, ``chunk_labels``, ``ner_labels``.
        If you want to use one of the labels as a `feature` in your model, it should be
        specified here.
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tag_label: str = "ner",
                 feature_labels: Sequence[str] = (),
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        if tag_label is not None and tag_label not in _VALID_LABELS:
            raise ConfigurationError("unknown tag label type: {}".format(tag_label))
        for label in feature_labels:
            if label not in _VALID_LABELS:
                raise ConfigurationError("unknown feature label type: {}".format(label))

        self.tag_label = tag_label
        self.feature_labels = set(feature_labels)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        if file_path[-8:] == 'test.txt':
            data = snips_reader('test.txt')
        elif file_path[-9:] == 'train.txt':
            data = snips_reader('train.txt')
        else:
            data = snips_reader('valid.txt')
        # if file_path[-9:] == 'train.txt':
        #     print(data[:10])

        for fields in data:
            # unzipping trick returns tuples, but our Fields need lists

            tokens, ner_tags = [list(field) for field in zip(*fields)]
            # TextField requires ``Token`` objects
            tokens = [Token(token) for token in tokens]
            sequence = TextField(tokens, self._token_indexers)

            instance_fields: Dict[str, Field] = {'tokens': sequence}
            # Add "feature labels" to instance
            if 'ner' in self.feature_labels:
                instance_fields['ner_tags'] = SequenceLabelField(ner_tags, sequence, "ner_tags")
            # Add "tag label" to instance
            instance_fields['tags'] = SequenceLabelField(ner_tags, sequence)
            yield Instance(instance_fields)

    def text_to_instance(self, tokens: List[Token]) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        return Instance({'tokens': TextField(tokens, token_indexers=self._token_indexers)})

    @classmethod
    def from_params(cls, params: Params) -> 'PnetOntoDatasetReader':
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        tag_label = params.pop('tag_label', None)
        feature_labels = params.pop('feature_labels', ())
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return PnetOntoDatasetReader(token_indexers=token_indexers,
                                     tag_label=tag_label,
                                     feature_labels=feature_labels,
                                     lazy=lazy)
