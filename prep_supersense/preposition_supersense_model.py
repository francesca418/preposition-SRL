# model

import sys

from typing import Dict, List, Optional, Any, Union

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel
from allennlp.data.tokenizers import Token

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.models.srl_util import convert_bio_tags_to_conll_format

@Model.register("preposition_supersense_bert")
class PrepSupersenseClassifierBert(Model): # Model inherits from torch.nn.Module and Registrable
  '''
  # Parameters

  Doesn't train anything beyond the Linear layer.

  vocab: `Vocabulary`, required
    A Vocabulary, required in order to compute sizes for input / output projections.
  model: `Union[str, BertModel]`, required
    A string describing the BERT model to load or an already constructed BertModel.
  initializer: `InitializerApplicator`, optional (default=`InitializerApplicator()`)
    Used to initialize the model parameters.
  regularizer: `RegularizerApplicator`, optional (default=None)
    If provided, will be used to calculate the regularization penalty during training.
  '''

  def __init__(
      self,
      vocab: Vocabulary,
      bert_model: Union[str, BertModel],
      embedding_dropout: float = 0.0,
      initializer: InitializerApplicator = InitializerApplicator(),
      regularizer: Optional[RegularizerApplicator] = None,
  ) -> None:
      super(PrepSupersenseClassifierBert, self).__init__(vocab, regularizer)

      if isinstance(bert_model, str):
        self.bert_model = BertModel.from_pretrained(bert_model)
      else:
        self.bert_model = bert_model
      
      in_features = (16, self.bert_model.config.hidden_size)
      out_features1 = self.vocab.get_vocab_size("supersense1_labels")
      out_features2 = self.vocab.get_vocab_size("supersense2_labels")

      self.num_classes1 = 55 #self.vocab.get_vocab_size("supersense1_labels") 
      self.num_classes2 = 52 #47 #38 #self.vocab.get_vocab_size("supersense2_labels")
      
      self.classification_layer1 = Linear(self.bert_model.config.hidden_size + 1, self.num_classes1)
      self.classification_layer2 = Linear(769, self.num_classes2)
      self.dropout = Dropout(p=embedding_dropout)
      self.accuracy1 = CategoricalAccuracy()
      self.loss1 = torch.nn.CrossEntropyLoss()
      self.accuracy2 = CategoricalAccuracy()
      self.loss2 = torch.nn.CrossEntropyLoss()
      initializer(self) # example does input as self.classification_layer

  def forward(
      self,
      tokens: Dict[str, torch.Tensor],
      prep_indicator: torch.Tensor,
      metadata: List[Any],
      tags: torch.LongTensor = None,
      supersense1_labels: str = None,
      supersense2_labels: str = None
  ):
      '''
      # Parameters

      tokens: Dict[str, torch.Tensor], required 
        The output of `TextField.as_array()`, which should typically be passed directly to a 
        `TextFieldEmbedder`. For this model, this must be a `SingleIdTokenIndexer` which
        indexes wordpieces from the BERT voacabulary.
      prep_indicator: torch.LongTensor, required.
        An integer `SequenceFeatureField` representation of the position of the preposition
        in the sentence. Shape is (batch_size, num_tokens) and can be all zeros,
        if the sentence has no preposition predicate.
      tags: torch.LongTensor, optional (default=None)
        Torch tensor representing sequence of integer gold class labels of shape `(batch_size, num_tokens)`.
      metadata: `List[Dict[str, Any]]` optional (default=None)
        metadata containing the original words of the sentence, the preposition to compute the frame for,
        and start offsets to convert wordpieces back to a sequence of words.

      # Returns

      Output dictionary consisting of:
      logits: torch.FloatTensor
        A tensor of shape `(batch_size, num_tokens, tag_vocab_size)` representing unnormalized log
        probabilities of the tag classes.
      class_probabilities: torch.FloatTensor
        A tensor of shape `(batch_size, num_tokens, tag_vocab_size)` representing a 
        distribution of the tag classes per word
      loss: torch.FloatTensor, optional
        A scalar loss to be optimized, during training phase.
      '''
      mask = get_text_field_mask(tokens)
      
      bert_embeddings, pooled = self.bert_model(
          input_ids=tokens["tokens"], 
          token_type_ids=prep_indicator,
          attention_mask=mask,
          output_all_encoded_layers=False
      )
      prep_indicator = prep_indicator.float()
      prep_indicator = prep_indicator.unsqueeze(2)
      bert_embeddings = torch.cat((bert_embeddings, prep_indicator), 2)

      embedded_text = torch.autograd.Variable(self.dropout(bert_embeddings)) #vs using embedded text input?

      logits1 = self.classification_layer1(embedded_text)
      logits2 = self.classification_layer2(embedded_text)

      prep_indicator = prep_indicator.squeeze()
      prep_indices = torch.nonzero(prep_indicator)
      val = -1
      idx_list = []
      for i in range(prep_indices.shape[0]):
        if prep_indices[i][0] == val:
          continue
        val = prep_indices[i][0]
        idx_list.append(prep_indices[i])

      prep_indices = torch.stack(idx_list)

      logits1_list = []
      logits2_list = []
      for i in range(prep_indices.shape[0]):
        current_idx = prep_indices[i][1]
        labels1 = logits1[i][current_idx] # the ith instance, the jth word (corresponds to the first word of the preposition)
        labels2 = logits2[i][current_idx]
        logits1_list.append(labels1)
        logits2_list.append(labels2)

      logits1 = torch.stack(logits1_list)
      logits2 = torch.stack(logits2_list)

      class_probabilities1 = torch.nn.functional.softmax(logits1, dim=-1)
      class_probabilities2 = torch.nn.functional.softmax(logits2, dim=-1)
      
      output_dict = {"ss1_logits": logits1, "ss2_logits": logits2, "ss1_class_probabilities": class_probabilities1, "ss2_class_probabilities": class_probabilities2}
      # retail the mask in the output dictionary so we can remove padding
      output_dict["mask"] = mask
      # add in offsets to compute un-wordpieced tags.
      words, prepositions, offsets, prep_indices = zip(*[(x["words"], x["preposition"], x["offsets"], x["prep_index"]) for x in metadata])
      output_dict["words"] = list(words)
      output_dict["preposition"] = list(prepositions)
      output_dict["wordpiece_offsets"] = list(offsets)
      output_dict["preposition_indices"] = prep_indices

      if (supersense1_labels is not None) and (supersense2_labels is not None):
        loss1 = self.loss1(logits1, supersense1_labels.long().view(-1))
        loss2 = self.loss2(logits2, supersense2_labels.long().view(-1))
        loss = loss1 + loss2
        output_dict["loss"] = loss
      elif supersense1_label is not None:
        loss = self.loss1(logits1, supersense1_labels.long().view(-1))
        output_dict["loss"] = loss
      elif supersense2_label is not None:
        loss = self.loss2(logits2, supersense2_labels.long().view(-1))
        output_dict["loss"] = loss
      
      self.accuracy1(logits1, supersense1_labels)
      self.accuracy2(logits2, supersense2_labels)

      return output_dict

  @overrides
  def decode(
      self, output_dict: Dict[str, torch.Tensor]
  ) -> Dict[str, torch.Tensor]:
      '''
      Does a simple argmax over the probabilities, converts index to string label, and add ``"label"`` key to the dictionary with the result.

      LATER ON: Could incorporate rules in decision-making as well (if needed)
      '''

      ss1_predictions = output_dict["ss1_class_probabilities"]
      ss2_predictions = output_dict["ss2_class_probabilities"]

      if ss1_predictions.dim() == self.num_classes1: 
        ss1_predictions_list = [ss1_predictions[i] for i in range(ss1_predictions.shape[0])]
      else:
        ss1_predictions_list = [ss1_predictions]

      if ss2_predictions.dim() == self.num_classes2:
        ss2_predictions_list = [ss2_predictions[i] for i in range(ss2_predictions.shape[0])]
      else:
        ss2_predictions_list = [ss2_predictions]

      ss1_classes = []
      ss2_classes = []

      for prediction in ss1_predictions_list:
        label_idx = prediction.argmax(dim=-1).item()
        label_str = (self.vocab.get_index_to_token_vocabulary("supersense1_labels").get(label_idx, str(label_idx)))
        ss1_classes.append(label_str)

      for prediction in ss2_predictions_list:
        label_idx = prediction.argmax(dim=-1).item()
        label_str = (self.vocab.get_index_to_token_vocabulary("supersense2_labels").get(label_idx, str(label_idx)))
        ss2_classes.append(label_str)

      output_dict["supersense1_label"] = ss1_classes
      output_dict["supersense2_label"] = ss2_classes

      return output_dict

  def get_metrics(self, reset: bool = False):
      return_dict = {}
      return_dict["ss1_accuracy"] = self.accuracy1.get_metric(reset)
      return_dict["ss2_accuracy"] = self.accuracy2.get_metric(reset)
      return_dict["combined_accuracy_score"] = 0.5 * (return_dict["ss1_accuracy"] + return_dict["ss2_accuracy"])
      return return_dict

  default_predictor = "preposition_supersense"

