import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np


class CoLA_Dataset(Dataset):
    def __init__(self, tokenized_dataset, labels=None, test=False):
        self.tokenized_dataset = tokenized_dataset
        self.test = test
        self.labels = labels

    def __getitem__(self, idx):
        if self.test:
            item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
            return item
        else:
            item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.tokenized_dataset['input_ids'])


def convert_trains_to_features(train_dataset, tokenizer, max_len, mode='train'):
    max_seq_len = max_len
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token_id
    add_sep_token = False
    mask_padding_with_zero = True

    all_input_ids = []
    all_attention_mask = []
    all_label = []
    all_target_ids = []
    m_len = 0

    for idx in tqdm(range(len(train_dataset))):
        input_sentence = cls_token  + train_dataset['sentence'][idx] + sep_token

        # print(sentence)

        input_token = tokenizer.tokenize(input_sentence)
#         input_token = np.array([[cls_token, ele] for ele in input_token]).reshape(-1).tolist() # 모든 토큰 앞에 cls
        m_len = max(m_len, len(input_token))

        if len(input_token) < max_seq_len:
            input_ids = tokenizer.convert_tokens_to_ids(input_token)
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            padding_length = max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

            assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids),
                                                                                            max_seq_len)
            assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
                len(attention_mask), max_seq_len
            )

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)

        all_label.append(train_dataset['acceptability_label'][idx])

    print(m_len)
    all_features = {
        'input_ids': torch.tensor(all_input_ids),
        'attention_mask': torch.tensor(all_attention_mask),
    }
    return CoLA_Dataset(all_features, all_label)


def convert_devs_to_features(train_dataset, tokenizer, max_len, mode='train'):
    max_seq_len = max_len
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token_id
    add_sep_token = False
    mask_padding_with_zero = True

    all_input_ids = []
    all_attention_mask = []
    all_label = []
    m_len = 0

    for idx in tqdm(range(len(train_dataset))):
        sentence = cls_token + train_dataset['sentence'][idx] + sep_token
        # print(sentence)

        token = tokenizer.tokenize(sentence)
        m_len = max(m_len, len(token))

        if len(token) < max_seq_len:
            input_ids = tokenizer.convert_tokens_to_ids(token)
#             input_token = np.array([[cls_token, ele] for ele in input_token]).reshape(-1).tolist() # 토큰 앞에 cls
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            padding_length = max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

            assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids),
                                                                                            max_seq_len)
            assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
                len(attention_mask), max_seq_len
            )

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_label.append(train_dataset['acceptability_label'][idx])

    print(m_len)
    all_features = {
        'input_ids': torch.tensor(all_input_ids),
        'attention_mask': torch.tensor(all_attention_mask)
    }
    return CoLA_Dataset(all_features, all_label)


def convert_test_to_features(train_dataset, tokenizer, max_len, mode='train'):
    max_seq_len = max_len
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token_id
    add_sep_token = False
    mask_padding_with_zero = True

    all_input_ids = []
    all_attention_mask = []
    m_len = 0

    for idx in tqdm(range(len(train_dataset))):
        sentence = cls_token + train_dataset['sentence'][idx] + sep_token
        # print(sentence)

        token = tokenizer.tokenize(sentence)
#         token = np.array([[cls_token, ele] for ele in token]).reshape(-1).tolist() # 토큰 앞에 cls
        m_len = max(m_len, len(token))

        if len(token) < max_seq_len:
            input_ids = tokenizer.convert_tokens_to_ids(token)
            
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            padding_length = max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

            assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids),
                                                                                            max_seq_len)
            assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
                len(attention_mask), max_seq_len
            )

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)

    print(m_len)
    all_features = {
        'input_ids': torch.tensor(all_input_ids),
        'attention_mask': torch.tensor(all_attention_mask)
    }
    return CoLA_Dataset(all_features, test=True)

# def convert_devs_to_features(train_dataset, tokenizer, max_len, mode='train'):
#     max_seq_len = max_len
#     cls_token = tokenizer.cls_token
#     sep_token = tokenizer.sep_token
#     pad_token = tokenizer.pad_token_id
#     add_sep_token = False
#     mask_padding_with_zero = True
#
#     all_input_ids = []
#     all_target_ids = []
#     all_attention_mask = []
#     all_label = []
#     m_len = 0
#
#     for idx in tqdm(range(len(train_dataset))):
#         sentence = cls_token + train_dataset['sentence'][idx] + sep_token
#         # print(sentence)
#
#         token = tokenizer.tokenize(sentence)
#         m_len = max(m_len, len(token))
#
#         if len(token) < max_seq_len:
#             input_ids = tokenizer.convert_tokens_to_ids(token)
#             attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
#
#             padding_length = max_seq_len - len(input_ids)
#             input_ids = input_ids + ([pad_token] * padding_length)
#             attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
#
#             assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids),
#                                                                                             max_seq_len)
#             assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
#                 len(attention_mask), max_seq_len
#             )
#
#             all_input_ids.append(input_ids)
#             all_target_ids.append(input_ids)
#             all_attention_mask.append(attention_mask)
#             all_label.append(train_dataset['acceptability_label'][idx])
#
#     print(m_len)
#     all_features = {
#         'input_ids': torch.tensor(all_input_ids),
#         'attention_mask': torch.tensor(all_attention_mask),
#         'target_ids': torch.tensor(all_target_ids)
#     }
#     return CoLA_Dataset(all_features, all_label)
