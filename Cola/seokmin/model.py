import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, ElectraModel, ElectraPreTrainedModel


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class PoolingHead(nn.Module):

    def __init__(
            self,
            input_dim: int,
            inner_dim: int,
            pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        return hidden_states


class Electra(ElectraPreTrainedModel):
    def __init__(self, config):
        super(Electra, self).__init__(config)
        self.electra = ElectraModel(config)  # Load pretrained Electra

        self.num_labels = config.num_labels

        self.pooling = PoolingHead(input_dim=config.hidden_size,
                                   inner_dim=config.hidden_size,
                                   pooler_dropout=0.1)
        self.qa_classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.electra(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        pooled_output = outputs[0][:, 0, :]  # [CLS]

        pooled_output = self.pooling(pooled_output)
        # pooled_output_cat = torch.cat([pooled_output, pooled_output2], dim=1)

        logits = self.qa_classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # logits, (hidden_states), (attentions)
