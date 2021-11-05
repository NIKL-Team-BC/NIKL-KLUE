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

        self.pooling = PoolingHead(input_dim=config.hidden_size,
            inner_dim=config.hidden_size,
            pooler_dropout=0.1)
        self.qa_classifier = nn.Linear(config.hidden_size, self.num_labels -1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, input_ids2=None, attention_mask2=None, token_type_ids2=None, labels=None):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        outputs2 = self.bert(
            input_ids2, attention_mask=attention_mask2
        )
        sequence_output = outputs[0]
        sequence_output2 = outputs2[0]
        pooled_output = outputs[0][:, 0, :]  # [CLS]
        pooled_output2 = outputs2[0][:, 0, :]

        sentence_representation = torch.cat([pooled_output, pooled_output2], dim=1)

        pooled_output = self.pooling(pooled_output)
        pooled_output2 = self.pooling(pooled_output2)
        
        logits1 = self.qa_classifier(pooled_output)
        logits2 = self.qa_classifier(pooled_output2)

        logits = torch.cat([logits1, logits2], dim=1)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # logits, (hidden_states), (attentions)


class XLMRoberta(XLMRobertaModel):
    def __init__(self, config, args):
        super(XLMRoberta, self).__init__(config)
        self.xlmroberta = XLMRobertaModel.from_pretrained("xlm-roberta-large", config=config)  # Load pretrained Electra

        self.num_labels = config.num_labels

        self.pooling = PoolingHead(input_dim=config.hidden_size,
            inner_dim=config.hidden_size,
            pooler_dropout=0.1)
        self.qa_classifier = nn.Linear(config.hidden_size, self.num_labels -1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, input_ids2=None, attention_mask2=None, token_type_ids2=None, labels=None):
        outputs = self.xlmroberta(
            input_ids, attention_mask=attention_mask
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        outputs2 = self.xlmroberta(
            input_ids2, attention_mask=attention_mask2
        )
        sequence_output = outputs[0]
        sequence_output2 = outputs2[0]
        pooled_output = outputs[0][:, 0, :]  # [CLS]
        pooled_output2 = outputs2[0][:, 0, :]

        sentence_representation = torch.cat([pooled_output, pooled_output2], dim=1)

        pooled_output = self.pooling(pooled_output)
        pooled_output2 = self.pooling(pooled_output2)
        
        logits1 = self.qa_classifier(pooled_output)
        logits2 = self.qa_classifier(pooled_output2)

        logits = torch.cat([logits1, logits2], dim=1)

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
        self.qa_classifier = nn.Linear(config.hidden_size, self.num_labels -1)
        #self.sparse = Sparsemax()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, input_ids2=None, attention_mask2=None, token_type_ids2=None, labels=None):
        outputs = self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        outputs2 = self.model(
            input_ids2, attention_mask=attention_mask2, token_type_ids=token_type_ids2
        )
        sequence_output = outputs[0]
        sequence_output2 = outputs2[0]
        pooled_output = outputs[0][:, 0, :]  # [CLS]
        pooled_output2 = outputs2[0][:, 0, :]

        sentence_representation = torch.cat([pooled_output, pooled_output2], dim=1)

        pooled_output = self.pooling(pooled_output)
        pooled_output2 = self.pooling(pooled_output2)
        
        logits1 = self.qa_classifier(pooled_output)
        logits2 = self.qa_classifier(pooled_output2)

        logits = torch.cat([logits1, logits2], dim=1)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # logits, (hidden_states), (attentions)


class Roberta(RobertaModel):
    def __init__(self, config, args):
        super(Roberta, self).__init__(config)
        self.roberta = RobertaModel.from_pretrained("klue/roberta-large", config=config)  # Load pretrained Electra

        self.num_labels = config.num_labels

        self.pooling = PoolingHead(input_dim=config.hidden_size,
            inner_dim=config.hidden_size,
            pooler_dropout=0.1)
        self.qa_classifier = nn.Linear(config.hidden_size, self.num_labels -1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, input_ids2=None, attention_mask2=None, token_type_ids2=None, labels=None):
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        outputs2 = self.roberta(
            input_ids2, attention_mask=attention_mask2
        )
        sequence_output = outputs[0]
        sequence_output2 = outputs2[0]
        pooled_output = outputs[0][:, 0, :]  # [CLS]
        pooled_output2 = outputs2[0][:, 0, :]

        sentence_representation = torch.cat([pooled_output, pooled_output2], dim=1)

        pooled_output = self.pooling(pooled_output)
        pooled_output2 = self.pooling(pooled_output2)
        
        logits1 = self.qa_classifier(pooled_output)
        logits2 = self.qa_classifier(pooled_output2)

        logits = torch.cat([logits1, logits2], dim=1)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # logits, (hidden_states), (attentions)


class CustomPreTrainModel(RobertaModel):
    def __init__(self, config, model):
        super(CustomPreTrainModel, self).__init__(config)
        self.model = model
        self.num_labels = 2
        self.qa_classifier_final = nn.Linear(self.num_labels*2, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, input_ids2=None, attention_mask2=None, token_type_ids2=None, labels=None):
        out1 = self.model(input_ids, attention_mask=attention_mask)[0]   # B, C   (여기서 C는 2라서 concat후 다시 2차원 변환 필요)
        out2 = self.model(input_ids2, attention_mask=attention_mask2)[0]

        out_cat = torch.cat([out1, out2], dim=1)   # B, C*2
        logits = self.qa_classifier_final(out_cat)

        outputs = (logits,) + (0,)

        return outputs


