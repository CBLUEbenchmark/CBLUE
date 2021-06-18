import numpy as np
import torch
from torch.utils.data import Dataset
from cblue.models import convert_examples_to_features_for_tokens


class EEDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128,
            ignore_label=-100,
            model_type='bert',
            ngram_dict=None
    ):
        super(EEDataset, self).__init__()

        self.orig_text = samples['orig_text']
        self.texts = samples['text']
        if mode != "test":
            self.labels = samples['label']
        else:
            self.labels = None

        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.ignore_label = ignore_label
        self.max_length = max_length
        self.mode = mode
        self.ngram_dict = ngram_dict
        self.model_type = model_type

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.model_type == 'zen':
            inputs = convert_examples_to_features_for_tokens(text, max_seq_length=self.max_length,
                                                             ngram_dict=self.ngram_dict, tokenizer=self.tokenizer,
                                                             return_tensors=True)
        else:
            inputs = self.tokenizer.encode_plus(text, max_length=self.max_length, padding='max_length', truncation=True)
        if self.mode != "test":
            label = [self.data_processor.label2id[label_] for label_ in
                     self.labels[idx].split('\002')]  # find index from label list
            label = ([-100] + label[:self.max_length - 2] + [-100] +
                     [self.ignore_label] * self.max_length)[:self.max_length]  # use ignore_label padding CLS+label+SEP
            if self.model_type == 'zen':
                return inputs['input_ids'], inputs['token_type_ids'], \
                       inputs['attention_mask'], torch.tensor(label), inputs['input_ngram_ids'], \
                       inputs['ngram_attention_mask'], inputs['ngram_token_type_ids'], \
                       inputs['ngram_position_matrix']
            else:
                return np.array(inputs['input_ids']), np.array(inputs['token_type_ids']), \
                    np.array(inputs['attention_mask']), np.array(label)
        else:
            if self.model_type == 'zen':
                return inputs['input_ids'], inputs['token_type_ids'], \
                       inputs['attention_mask'], inputs['input_ngram_ids'], \
                       inputs['ngram_attention_mask'], inputs['ngram_token_type_ids'], \
                       inputs['ngram_position_matrix']
            else:
                return np.array(inputs['input_ids']), np.array(inputs['token_type_ids']), \
                       np.array(inputs['attention_mask']),

    def __len__(self):
        return len(self.texts)


class REDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128,
            model_type='bert',
            ngram_dict=None
    ):
        super(REDataset, self).__init__()

        self.texts = samples['text']
        self.flags = samples['flag']

        if mode != "test":
            self.labels = samples['label']

        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.model_type = model_type
        self.ngram_dict = ngram_dict

    def __getitem__(self, idx):
        text, flag = self.texts[idx], self.flags[idx]
        if self.model_type == 'zen':
            inputs = convert_examples_to_features_for_tokens(text, max_seq_length=self.max_length,
                                                             ngram_dict=self.ngram_dict, tokenizer=self.tokenizer,
                                                             return_tensors=False)
        else:
            inputs = self.tokenizer.encode_plus(text, max_length=self.max_length, padding='max_length', truncation=True)

        s_encode = self.tokenizer.encode(flag[0])
        s_start_idx = self.data_processor.search(inputs['input_ids'], s_encode[1:-1])

        o_encode = self.tokenizer.encode(flag[1])
        o_start_idx = self.data_processor.search(inputs['input_ids'], o_encode[1:-1])
        if self.mode != "test":
            label = self.labels[idx]
            if self.model_type == 'zen':
                return torch.tensor(inputs['input_ids']), torch.tensor(inputs['token_type_ids']), \
                       torch.tensor(inputs['attention_mask']), torch.tensor([s_start_idx, o_start_idx]), \
                       torch.tensor(label), torch.tensor(inputs['input_ngram_ids']), \
                       torch.tensor(inputs['ngram_attention_mask']), torch.tensor(inputs['ngram_token_type_ids']), \
                       torch.tensor(inputs['ngram_position_matrix'])
            else:
                return torch.tensor(inputs['input_ids']), \
                       torch.tensor(inputs['token_type_ids']), \
                       torch.tensor(inputs['attention_mask']), \
                       torch.tensor([s_start_idx, o_start_idx]).long(), \
                       torch.tensor(label).long()
        else:
            if self.model_type == 'zen':
                return torch.tensor(inputs['input_ids']), torch.tensor(inputs['token_type_ids']), \
                       torch.tensor(inputs['attention_mask']), torch.tensor([s_start_idx, o_start_idx]), \
                       torch.tensor(inputs['input_ngram_ids']), \
                       torch.tensor(inputs['ngram_attention_mask']), torch.tensor(inputs['ngram_token_type_ids']), \
                       torch.tensor(inputs['ngram_position_matrix'])
            else:
                return torch.tensor(inputs['input_ids']), \
                       torch.tensor(inputs['token_type_ids']).long(), \
                       torch.tensor(inputs['attention_mask']).float(), \
                       torch.tensor([s_start_idx, o_start_idx]).long()

    def __len__(self):
        return len(self.texts)


class ERDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128,
            model_type='bert',
            ngram_dict=None
    ):
        super(ERDataset, self).__init__()

        self.texts = samples['text']
        if mode != 'test':
            self.spo_lists = samples['spo_list']

        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.model_type = model_type
        self.ngram_dict = ngram_dict

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.model_type == 'zen':
            inputs = convert_examples_to_features_for_tokens(text, max_seq_length=self.max_length,
                                                             ngram_dict=self.ngram_dict, tokenizer=self.tokenizer,
                                                             return_tensors=False)
        else:
            inputs = self.tokenizer.encode_plus(text, max_length=self.max_length, padding='max_length', truncation=True)
        if self.mode != "test":
            spo_list = self.spo_lists[idx]

            sub_start_label = np.zeros((self.max_length,), dtype=int)
            sub_end_label = np.zeros((self.max_length,), dtype=int)
            obj_start_label = np.zeros((self.max_length,), dtype=int)
            obj_end_label = np.zeros((self.max_length,), dtype=int)
            for spo in spo_list:
                sub_encode = self.tokenizer.encode(spo[0])
                sub_start_idx = self.data_processor.search(inputs['input_ids'], sub_encode[1:-1])  # 去掉CLS SEP
                sub_end_idx = sub_start_idx + len(sub_encode[1:-1]) - 1
                obj_encode = self.tokenizer.encode(spo[2])
                obj_start_idx = self.data_processor.search(inputs['input_ids'], obj_encode[1:-1])
                obj_end_idx = obj_start_idx + len(obj_encode[1:-1]) - 1

                sub_start_label[sub_start_idx] = 1
                sub_end_label[sub_end_idx] = 1
                obj_start_label[obj_start_idx] = 1
                obj_end_label[obj_end_idx] = 1

            if self.model_type == 'zen':
                return torch.tensor(inputs['input_ids']), torch.tensor(inputs['token_type_ids']), \
                       torch.tensor(inputs['attention_mask']), torch.tensor(sub_start_label).long(), \
                       torch.tensor(sub_end_label).long(), \
                       torch.tensor(obj_start_label).long(), \
                       torch.tensor(obj_end_label).long(), torch.tensor(inputs['input_ngram_ids']), \
                       torch.tensor(inputs['ngram_attention_mask']), torch.tensor(inputs['ngram_token_type_ids']), \
                       torch.tensor(inputs['ngram_position_matrix'])
            else:
                return torch.tensor(inputs['input_ids']), \
                       torch.tensor(inputs['token_type_ids']), \
                       torch.tensor(inputs['attention_mask']), \
                       torch.tensor(sub_start_label).long(), \
                       torch.tensor(sub_end_label).long(), \
                       torch.tensor(obj_start_label).long(), \
                       torch.tensor(obj_end_label).long()
        else:
            if self.model_type == 'zen':
                return torch.tensor(inputs['input_ids']), torch.tensor(inputs['token_type_ids']), \
                       torch.tensor(inputs['attention_mask']), torch.tensor(inputs['input_ngram_ids']), \
                       torch.tensor(inputs['ngram_attention_mask']), torch.tensor(inputs['ngram_token_type_ids']), \
                       torch.tensor(inputs['ngram_position_matrix'])
            else:
                return torch.tensor(inputs['input_ids']).long(), \
                       torch.tensor(inputs['token_type_ids']).long(), \
                       torch.tensor(inputs['attention_mask']).long()

    def __len__(self):
        return len(self.texts)


class CDNDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            mode='train',
            dtype='cls'
    ):
        super(CDNDataset, self).__init__()

        self.text1 = samples['text1']

        if dtype == 'cls':
            self.text2 = samples['text2']
            if mode != 'test':
                self.label = samples['label']
        else:
            if mode != 'test':
                self.label = samples['label']

        self.data_processor = data_processor
        self.dtype = dtype
        self.mode = mode

    def __getitem__(self, item):
        if self.dtype == 'cls':
            if self.mode != 'test':
                return self.text1[item], self.text2[item], self.label[item]
            else:
                return self.text1[item], self.text2[item]
        else:
            if self.mode != 'test':
                return self.text1[item], self.label[item]
            else:
                return self.text1[item]

    def __len__(self):
        return len(self.text1)


class CTCDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            max_length=128,
            mode='train',
            model_type='bert',
            ngram_dict=None
    ):
        super(CTCDataset, self).__init__()

        self.texts = [text.split("\002") for text in samples['text']]
        self.ids = samples['id']

        if mode != 'test':
            self.labels = samples['label']
        self.data_processor = data_processor
        self.mode = mode
        self.ngram_dict = ngram_dict
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.model_type = model_type

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.model_type == 'zen':
            inputs = convert_examples_to_features(text1=text, ngram_dict=self.ngram_dict,
                                                  tokenizer=self.tokenizer, max_seq_length=self.max_length)
        else:
            inputs = self.tokenizer.encode_plus(text, padding='max_length', max_length=self.max_length, truncation=True)

        if self.mode != 'test':
            if self.model_type == 'zen':
                return inputs['input_ids'], inputs['token_type_ids'], \
                       inputs['attention_mask'], self.labels[idx], inputs['input_ngram_ids'], \
                       inputs['ngram_attention_mask'], inputs['ngram_token_type_ids'], \
                       inputs['ngram_position_matrix']
            else:
                return np.array(inputs['input_ids']), np.array(inputs['token_type_ids']), \
                    np.array(inputs['attention_mask']), self.labels[idx]
        else:
            if self.model_type == 'zen':
                return inputs['input_ids'], inputs['token_type_ids'], \
                       inputs['attention_mask'], inputs['input_ngram_ids'], \
                       inputs['ngram_attention_mask'], inputs['ngram_token_type_ids'], \
                       inputs['ngram_position_matrix']
            else:
                return np.array(inputs['input_ids']), np.array(inputs['token_type_ids']), \
                       np.array(inputs['attention_mask']),

    def __len__(self):
        return len(self.texts)


class STSDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            mode='train'
    ):
        super(STSDataset, self).__init__()

        self.text1 = samples['text1']
        self.text2 = samples['text2']
        self.ids = samples['id']
        self.category = samples['category']

        if mode != 'test':
            self.labels = samples['label']

        self.data_processor = data_processor
        self.mode = mode

    def __getitem__(self, item):
        if self.mode != 'test':
            return self.text1[item], self.text2[item], self.labels[item]
        else:
            return self.text1[item], self.text2[item]

    def __len__(self):
        return len(self.text1)


class QQRDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            mode='train'
    ):
        super(QQRDataset, self).__init__()

        self.text1 = samples['text1']
        self.text2 = samples['text2']
        self.ids = samples['id']

        if mode != 'test':
            self.labels = samples['label']

        self.data_processor = data_processor
        self.mode = mode

    def __getitem__(self, item):
        if self.mode != 'test':
            return self.text1[item], self.text2[item], self.labels[item]
        else:
            return self.text1[item], self.text2[item]

    def __len__(self):
        return len(self.text1)


class QICDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            mode='train'
    ):
        super(QICDataset, self).__init__()

        self.text = samples['text']
        self.ids = samples['id']

        if mode != 'test':
            self.labels = samples['label']

        self.data_processor = data_processor
        self.mode = mode

    def __getitem__(self, item):
        if self.mode != 'test':
            return self.text[item], self.labels[item]
        else:
            return self.text[item]

    def __len__(self):
        return len(self.text)


class QTRDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            mode='train'
    ):
        super(QTRDataset, self).__init__()

        self.text1 = samples['text1']
        self.text2 = samples['text2']
        self.ids = samples['id']

        if mode != 'test':
            self.labels = samples['label']

        self.data_processor = data_processor
        self.mode = mode

    def __getitem__(self, item):
        if self.mode != 'test':
            return self.text1[item], self.text2[item], self.labels[item]
        else:
            return self.text1[item], self.text2[item]

    def __len__(self):
        return len(self.text1)


