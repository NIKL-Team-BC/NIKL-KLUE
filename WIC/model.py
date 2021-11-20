import torch
import torch.nn as nn
from transformers import AutoModel, RobertaPreTrainedModel



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


class R_RoBERTa_WiC(RobertaPreTrainedModel):
    def __init__(self,  model_name, config, dropout_rate):
        super(R_RoBERTa_WiC, self).__init__(config)
        self.model = AutoModel.from_pretrained(model_name, config=config)  # Load pretrained XLMRoberta

        self.num_labels = config.num_labels

        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, dropout_rate)
        self.entity_fc_layer1 = FCLayer(config.hidden_size, config.hidden_size, dropout_rate)
        self.entity_fc_layer2 = FCLayer(config.hidden_size, config.hidden_size, dropout_rate)

        self.label_classifier = FCLayer(
            config.hidden_size * 3,
            config.num_labels,
            dropout_rate,
            use_activation=False,
        )

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, labels, e1_mask, e2_mask):
        outputs = self.model(
            input_ids, attention_mask=attention_mask
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0] #batch, max_len, hidden_size  

        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)
        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        sentence_representation = self.cls_fc_layer(outputs.pooler_output)

        e1_h = self.entity_fc_layer1(e1_h)
        e2_h = self.entity_fc_layer2(e2_h)
        # Concat -> fc_layer
        concat_h = torch.cat([sentence_representation, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        # Softmax
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)