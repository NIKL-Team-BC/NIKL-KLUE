import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, ElectraModel, ElectraPreTrainedModel, XLMRobertaModel, BartModel, BartPretrainedModel, T5Model, RobertaModel 
from transformers import MBartModel, MBartConfig
from transformers import BertTokenizer, BertModel


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


class Bert(BertPreTrainedModel):
    def __init__(self, config, args):
        super(Bert, self).__init__(config)
        self.bert = BertModel(config=config)  # Load pretrained bert
        
        self.num_labels = config.num_labels

        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, args.dropout_rate)
        # l2 norm, similarity add
        self.label_classifier = FCLayer(
            config.hidden_size,
            config.num_labels,
            args.dropout_rate,
            use_activation=False,
        )

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)

        # Concat -> fc_layer
        logits = self.label_classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # logits, (hidden_states), (attentions)


class Electra(ElectraPreTrainedModel):
    def __init__(self, config, args):
        super(Electra, self).__init__(config)
        self.electra = ElectraModel(config)  # Load pretrained Electra

        self.num_labels = config.num_labels

        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, args.dropout_rate)
        self.label_classifier = FCLayer(
            config.hidden_size,
            config.num_labels,
            dropout_rate = 0,
            use_activation=False,
        )

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        outputs = self.electra(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        pooled_output = outputs[0][:, 0, :]  # [CLS]
        pooled_output = self.cls_fc_layer(pooled_output)

        logits = self.label_classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # (loss), logits, (hidden_states), (attentions)


class XLMRoberta(XLMRobertaModel):
    def __init__(self, config, args):
        super(XLMRoberta, self).__init__(config)
        self.xlmroberta = XLMRobertaModel.from_pretrained("xlm-roberta-large", config=config)  # Load pretrained Electra

        self.num_labels = config.num_labels

        self.pooling = PoolingHead(input_dim=config.hidden_size,
            inner_dim=config.hidden_size,
            pooler_dropout=0.1)
        self.qa_classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.xlmroberta(
            input_ids, attention_mask=attention_mask
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[0][:, 0, :]  # [CLS]

        pooled_output = self.pooling(pooled_output)
        logits = self.qa_classifier(pooled_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # logits, (hidden_states), (attentions)


class Electra_BoolQ(ElectraPreTrainedModel):
    def __init__(self, config, args):
        super(Electra_BoolQ, self).__init__(config)

        #self.num_labels = config.num_labels
        self.num_labels = config.num_labels
        self.model = ElectraModel.from_pretrained('monologg/koelectra-base-v3-discriminator', config=config)
        self.pooling = PoolingHead(input_dim=config.hidden_size,
            inner_dim=config.hidden_size,
            pooler_dropout=0.1)
        self.qa_classifier = nn.Linear(config.hidden_size, self.num_labels)
        #self.sparse = Sparsemax()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None
    ):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0][:,0,:] #cls
        sequence_output = self.pooling(sequence_output)
        logits = self.qa_classifier(sequence_output)
        #logits = self.sparse(logits)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # (loss), logits, (hidden_states), (attentions)
        

class Roberta(RobertaModel):
    def __init__(self, config, args):
        super(Roberta, self).__init__(config)
        self.roberta = RobertaModel.from_pretrained("klue/roberta-large", config=config)  # Load pretrained Electra

        self.num_labels = config.num_labels

        self.pooling = PoolingHead(input_dim=config.hidden_size,
            inner_dim=config.hidden_size,
            pooler_dropout=0.1)
        self.qa_classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        pooled_output = outputs[0][:, 0, :]  # [CLS]

        pooled_output = self.pooling(pooled_output)
        # pooled_output_cat = torch.cat([pooled_output, pooled_output2], dim=1)
        
        logits = self.qa_classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # logits, (hidden_states), (attentions)