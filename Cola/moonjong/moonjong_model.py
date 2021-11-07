from torch import nn
import torch
from transformers import XLMRobertaModel, RobertaModel, RobertaForCausalLM, AutoModel
from transformers.modeling_bert import BertOnlyMLMHead


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        if self.use_activation:
            x = self.activation(x)
        return x


class CoLA(nn.Module):
    def __init__(self, model_name, config, dropout_rate):
        super(CoLA, self).__init__()
        self.model = AutoModel.from_pretrained(model_name, config=config)  # Load pretrained XLMRoberta

        self.num_labels = config.num_labels
        self.decode_token = FCLayer(config.hidden_size, config.vocab_size, dropout_rate)
        self.eos_fc_layer = FCLayer(config.hidden_size, self.num_labels, dropout_rate)
        self.mlm_head = BertOnlyMLMHead(config)

    def forward_pretrain(self, input_ids):
        encoder_outputs, _ = self.model(input_ids=input_ids)
        mlm_logits = self.mlm_head(encoder_outputs)

        return mlm_logits

    def forward(self, input_ids, attention_mask, target_ids=None, labels=None):
        outputs = self.model(
            input_ids, attention_mask=attention_mask
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        sequence_representation = torch.mean(sequence_output, 1)
        logits = self.eos_fc_layer(sequence_representation)
        outputs = (logits,) + outputs[2:]
        # Softmax

        ## token_loss

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                # loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)

            outputs = (loss,) + outputs

        return outputs  # (loss), (logits etc...)
