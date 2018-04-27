from typing import Dict, Optional
from collections import defaultdict, Iterable
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear

from torch.autograd import Variable
from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import SpanBasedF1Measure
import numpy as np


@Model.register("pnet_crf_tagger")
class PnetCrfTagger(Model):
    """
    The ``CrfTagger`` encodes a sequence of text with a ``Seq2SeqEncoder``,
    then uses a Conditional Random Field model to predict a tag for each token in the sequence.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the tokens ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder that we will use in between embedding tokens and predicting output tags.
    label_namespace : ``str``, optional (default=``labels``)
        This is needed to compute the SpanBasedF1Measure metric.
        Unless you did something unusual, the default value should be what you want.
    constraint_type : ``str``, optional (default=``None``)
        If provided, the CRF will be constrained at decoding time
        to produce valid labels based on the specified type (e.g. "BIO", or "BIOUL").
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 label_namespace: str = "labels",
                 constraint_type: str = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_tags = self.vocab.get_vocab_size(label_namespace)
        self.encoder = encoder

        self.last_layer = Linear(400, 64)

        self.bias_outside = torch.nn.Parameter(torch.zeros(1) - 1., requires_grad=True)

        self.stds = torch.autograd.Variable(torch.ones(self.num_tags))
        self.sums = np.zeros(self.num_tags) + 10
        self.amount = np.zeros(self.num_tags) + 11
        self.loss = torch.nn.CrossEntropyLoss()

        self.tag_projection_layer = TimeDistributed(Linear(self.encoder.get_output_dim(),
                                                           self.num_tags))

        if constraint_type is not None:
            labels = self.vocab.get_index_to_token_vocabulary(label_namespace)
            constraints = allowed_transitions(constraint_type, labels)
        else:
            constraints = None

        self.crf = ConditionalRandomField(self.num_tags, constraints)

        self.span_metric = SpanBasedF1Measure(vocab, tag_namespace=label_namespace)

        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        initializer(self)

        self.hash = 0
        self.new = True

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                tags: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : ``Dict[str, torch.LongTensor]``, required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        tags : ``torch.LongTensor``, optional (default = ``None``)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.

        Returns
        -------
        An output dictionary consisting of:

        logits : ``torch.FloatTensor``
            The logits that are the output of the ``tag_projection_layer``
        mask : ``torch.LongTensor``
            The text field mask for the input tokens
        tags : ``List[List[str]]``
            The predicted tags using the Viterbi algorithm.
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised. Only computed if gold label ``tags`` are provided.
        """

        embedded_text_input = self.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)
        encoded_text = self.encoder(embedded_text_input, mask)

        embeddings = self.last_layer(encoded_text)
        target_relation = 1.

        if not tokens['tokens'].volatile:
            for i in range(embeddings.shape[0]):
                if (embeddings.shape[0] - i - 1) / (i + 1) > target_relation:
                    continue
                tags_support = tags[:(i + 1)]
                tags_query = tags[(i + 1):]
                uniq_support = np.unique(tags_support.cpu().data.numpy())
                uniq_query = np.unique(tags_query.cpu().data.numpy())
                is_ok = True
                for tag in uniq_query:
                    if tag not in uniq_support:
                        is_ok = False
                        break
                if not is_ok:
                    continue
                else:
                    support = embeddings[:(i + 1)]
                    query = embeddings[(i + 1):]
                    support_mask = mask[:(i + 1)]
                    query_mask = mask[(i + 1):]
                    break
        else:
            i = 49
            if self.hash != np.sum(tags[:(i + 1)].data.cpu().numpy()):
                self.hash = np.sum(tags[:(i + 1)].data.cpu().numpy())
                self.new = True
            else:
                self.new = False
            tags_support = tags[:(i + 1)]
            tags_query = tags[(i + 1):]
            uniq_support = np.unique(tags_support.cpu().data.numpy())
            support = embeddings[:(i + 1)]
            query = embeddings[(i + 1):]
            query_mask = mask[(i + 1):]
            support_mask = mask[:(i + 1)]

        mask_query = query_mask.data.numpy()
        mask_support = support_mask.data.numpy()
        decoder = dict(zip(uniq_support, np.arange(uniq_support.shape[0])))
        encoder = dict(zip(np.arange(uniq_support.shape[0]), uniq_support))

        embeds_per_class = [[] for _ in np.arange(np.unique(uniq_support).shape[0])]

        for i_sen, sentence in enumerate(support):
            for i_word, word in enumerate(sentence):
                if mask_support[i_sen, i_word] == 1:
                    tag = tags_support[i_sen][i_word].data.cpu().numpy()[0]
                    if tag > 0:
                        embeds_per_class[decoder[tag]].append(word)

        prototypes = [torch.zeros_like(embeds_per_class[1][0]) for _ in range(len(embeds_per_class))]

        for i in range(len(embeds_per_class)):
            for embed in embeds_per_class[i]:
                prototypes[i] += embed / len(embeds_per_class[i])

        support_distances = [[] for _ in range(self.num_tags)]
        for i_sen, sentence in enumerate(support):
            for i_word, word in enumerate(sentence):
                if mask_support[i_sen, i_word] == 1:
                    i_class = tags_support.data.cpu().numpy()[i_sen][i_word]
                    if i_class > 0:
                        distance = (torch.sum(torch.pow(word - prototypes[decoder[i_class]], 2)).cpu().data.numpy()[0])
                        support_distances[i_class].append(distance)
        vars = [1]
        for i in range(self.num_tags)[1:]:
            if len(support_distances[i]) > 1:
                sum = np.sum(support_distances[i])
                # self.sums[i] = 0.9 * self.sums[i] + sum
                # self.amount[i] = (0.9 * (self.amount[i] - 1)) + 1 + len(support_distances[i])
            vars.append((self.sums[i]/(self.amount[i]-1)))
        n_words = 0
        for i_sen, sentence in enumerate(query):
            for i_word, word in enumerate(sentence):
                if mask_query[i_sen, i_word] == 1:
                    n_words += 1

        logits = Variable(torch.zeros((n_words, len(embeds_per_class))))
        answers = np.zeros((n_words,))
        i_logit = 0
        query_tags = []
        for i_sen, sentence in enumerate(query):
            query_tags.append([])
            for i_word, word in enumerate(sentence):
                if mask_query[i_sen, i_word] == 1:
                    answers[i_logit] = decoder[tags_query.data.cpu().numpy()[i_sen][i_word]]
                    logits[i_logit, 0] = self.bias_outside
                    for i_class in range(len(embeds_per_class))[1:]:
                        distance = torch.sum(torch.pow(word - prototypes[i_class], 2))
                        logits[i_logit, i_class] = -distance/vars[encoder[i_class]] - np.log(vars[encoder[i_class]])/2.
                    query_tags[-1].append(encoder[np.argmax(logits[i_logit].data.numpy())])
                    i_logit += 1
        answers = Variable(torch.LongTensor(answers))
        if not tokens['tokens'].volatile:
            loss = self.loss(logits, answers)
        else:
            loss = self.loss(logits, answers).detach()

        class_probabilities = torch.zeros((tags.shape[0], tags.shape[1], self.num_tags))
        for i, instance_tags in enumerate(query_tags):
            for j, tag_id in enumerate(instance_tags):
                class_probabilities[i, j, tag_id] = 1
        self.span_metric(class_probabilities, tags_query, query_mask)

        output = {"mask": mask, "loss": loss}

        return output

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        output_dict["tags"] = [
            [self.vocab.get_token_from_index(tag, namespace="labels")
             for tag in instance_tags]
            for instance_tags in output_dict["tags"]
            ]

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = self.span_metric.get_metric(reset=reset)
        return {x: y for x, y in metric_dict.items() if "overall" in x}

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'CrfTagger':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        label_namespace = params.pop("label_namespace", "labels")
        constraint_type = params.pop("constraint_type", None)
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        params.assert_empty(cls.__name__)

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   encoder=encoder,
                   label_namespace=label_namespace,
                   constraint_type=constraint_type,
                   initializer=initializer,
                   regularizer=regularizer)
