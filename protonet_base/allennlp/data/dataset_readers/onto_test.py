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

SNIPS_URL = 'http://share.ipavlov.mipt.ru:8080/repository/datasets/ner/SNIPS2017.tar.gz'


def tokenize(s):
    return re.findall(r"[\w']+|[‑–—“”€№…’\"#$%&\'()+,-./:;<>?]", s)


# def download_and_extract_archive(url, extract_path):
#     archive_filename = url.split('/')[-1]
#     archive_path = os.path.join(extract_path, archive_filename)
#     os.makedirs(extract_path, exist_ok=True)
#     urllib.request.urlretrieve(SNIPS_URL, archive_path)
#     f = tarfile.open(archive_path)
#     f.extractall(extract_path)
#     f.close()
#     os.remove(archive_path)
#
#
# def parse_snips_utterance(utterance):
#     if 'data' in utterance:
#         # utterance_tokens = list()
#         # utterance_tags = list()
#         instances = []
#         for item in utterance['data']:
#             # print(item)
#             text = item['text']
#             tokens = tokenize(text)
#             if 'entity' in item:
#                 entity = item['entity']
#                 tags = list()
#                 for n in range(len(tokens)):
#                     if n == 0:
#                         tags.append('B-' + entity)
#                     else:
#                         tags.append('I-' + entity)
#             else:
#                 tags = ['O' for _ in range(len(tokens))]
#             instances += [[to, ta] for ta, to in zip(tags, tokens)]
#             # utterance_tags.extend(tags)
#             # utterance_tokens.extend(tokens)
#     return instances


def snips_reader(file='train', dataset_download_path='ontonotes/', return_intent=False):
    # param: dataset_download_path - path to the existing dataset or if there is no
    #   dataset there the dataset it will be downloaded to this path

    sentences = []
    for file_name in ['valid.txt']:
        with open(dataset_download_path + file_name, "r") as data_file:
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider:
                    fields = [line.strip().split() for line in lines]
                    tokens, ner_tags = [list(field) for field in zip(*fields)]
                    sentences.append([[token, tag] for token, tag in zip(tokens, ner_tags)])

    target_relation = 10.
    total_sentences = 500
    random_seed = 1

    np.random.seed(random_seed)
    np.random.shuffle(sentences)

    ys = []
    for sent in sentences:
        for word in sent:
            ys.append(word[1])
    ys = np.unique([y[2:] for y in np.unique(ys)[:-1]])

    np.random.shuffle(ys)

    sentences = sentences[(len(sentences) // 2):]

    if True:
        ys = ys[(ys.shape[0] // 2):]
        for sent in sentences:
            for word in sent:
                if word[1][2:] not in ys:
                    word[1] = 'O'

    ys_sup = np.array([])
    support = []
    query = []
    np.random.seed(random_seed)
    for i in range(total_sentences):
        sentence = random.choice(sentences)
        ys_here = [xy[1] for xy in sentence]
        if (ys_sup.shape[0] < np.unique(np.concatenate((ys_sup, np.unique(ys_here)))).shape[0]) or (
            (len(query) + 1) / len(support)) > target_relation:
            support.append(sentence)
        else:
            query.append(sentence)
        ys_sup = np.unique(np.concatenate((ys_sup, np.unique(ys_here))))

    if file == 'train.txt':
        return support
    else:
        return query


@DatasetReader.register("onto_test")
class TestOntoDatasetReader(DatasetReader):
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
    def from_params(cls, params: Params) -> 'TestOntoDatasetReader':
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        tag_label = params.pop('tag_label', None)
        feature_labels = params.pop('feature_labels', ())
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return TestOntoDatasetReader(token_indexers=token_indexers,
                                  tag_label=tag_label,
                                  feature_labels=feature_labels,
                                  lazy=lazy)
