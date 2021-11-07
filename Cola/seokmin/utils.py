import torch
import torch.nn as nn
from transformers import XLMRobertaModel
import numpy as np
import pandas as pd


class get_similarity(XLMRobertaModel):
  def __init__(self, config, args):
    super(get_similarity, self).__init__(config)
    self.xlmroberta = XLMRobertaModel.from_pretrained("xlm-roberta-large")
    self.num_labels = config.num_labels
  
  def forward(self, input_ids, attention_mask, token_type_ids, labels):
    outputs = self.xlmroberta(
      input_ids, attention_mask=attention_mask
    )
    print("forward outputs: ", outputs)


def check_arch(model_type):
  archs = {
    "encoder" : ["Bert", "Electra", "XLMRoberta", "Electra_BoolQ", "Roberta"],
    "encoder-decoder" : ["T5", "Bart", "Bart_BoolQ"]
  }
  for arch in archs:
    if model_type in archs[arch]:
      return arch
  raise ValueError(f"Model [{model_type}] no defined archtecture")