import numpy as np
import torch
from torch.utils.data import Dataset


class EEDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128,
            ignore_label=-100
    ):
        super(EEDataset, self).__init__()

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

    def __getitem__(self, idx):
        text = self.texts[idx]

        if self.mode != "test":
            label = [self.data_processor.label2id[label_] for label_ in
                     self.labels[idx].split('\002')]  # find index from label list
            label = ([-100] + label[:self.max_length - 2] + [-100] +
                     [self.ignore_label] * self.max_length)[:self.max_length]  # use ignore_label padding CLS+label+SEP
            return text, np.array(label)
        else:
            return text

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
    ):
        super(REDataset, self).__init__()

        self.texts = samples['text']
        self.flags = samples['flag']
        self.labels = samples['label']
        if mode != "test":
            self.labels = samples['label']

        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

    def __getitem__(self, idx):
        text, flag = self.texts[idx], self.flags[idx]
        inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length',
                                truncation=True, add_special_tokens=False)
        input_ids, token_type_ids, attention_mask = inputs['input_ids'], inputs['token_type_ids'], \
                                                    inputs['attention_mask']
        s_encode = self.tokenizer.encode(flag[0])
        s_start_idx = self.data_processor.search(input_ids, s_encode[1:-1])

        o_encode = self.tokenizer.encode(flag[1])
        o_start_idx = self.data_processor.search(input_ids, o_encode[1:-1])
        if self.mode != "test":
            label = self.labels[idx]
            return torch.tensor(input_ids).long(), \
                   torch.tensor(token_type_ids).long(), \
                   torch.tensor(attention_mask).float(), \
                   torch.tensor([s_start_idx, o_start_idx]).long(), \
                   torch.tensor(label).long()
        else:
            return torch.tensor(input_ids).long(), \
                   torch.tensor(token_type_ids).long(), \
                   torch.tensor(attention_mask).float(), \
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
    ):
        super(ERDataset, self).__init__()

        self.texts = samples['text']
        if mode != 'test':
            self.spo_lists = samples['spo_list']

        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, max_length=self.max_length, padding="max_length",
                                truncation=True, add_special_tokens=False)
        input_ids, token_type_ids, attention_mask = inputs['input_ids'], inputs['token_type_ids'], \
                                                    inputs['attention_mask']
        if self.mode != "test":
            spo_list = self.spo_lists[idx]

            sub_start_label = np.zeros((self.max_length,), dtype=int)
            sub_end_label = np.zeros((self.max_length,), dtype=int)
            obj_start_label = np.zeros((self.max_length,), dtype=int)
            obj_end_label = np.zeros((self.max_length,), dtype=int)
            for spo in spo_list:
                sub_encode = self.tokenizer.encode(spo[0])
                sub_start_idx = self.data_processor.search(input_ids, sub_encode[1:-1])  # 去掉CLS SEP
                sub_end_idx = sub_start_idx + len(sub_encode[1:-1]) - 1
                obj_encode = self.tokenizer.encode(spo[2])
                obj_start_idx = self.data_processor.search(input_ids, obj_encode[1:-1])
                obj_end_idx = obj_start_idx + len(obj_encode[1:-1]) - 1

                sub_start_label[sub_start_idx] = 1
                sub_end_label[sub_end_idx] = 1
                obj_start_label[obj_start_idx] = 1
                obj_end_label[obj_end_idx] = 1
            return torch.tensor(input_ids).long(), \
                   torch.tensor(token_type_ids).long(), \
                   torch.tensor(attention_mask).float(), \
                   torch.tensor(sub_start_label).long(), \
                   torch.tensor(sub_end_label).long(), \
                   torch.tensor(obj_start_label).long(), \
                   torch.tensor(obj_end_label).long()
        else:
            return torch.tensor(input_ids).long(), \
                   torch.tensor(token_type_ids).long(), \
                   torch.tensor(attention_mask).float()

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
                return self.text1[item], self.label[item]

    def __len__(self):
        return len(self.text1)


class CTCDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            mode='train'
    ):
        super(CTCDataset, self).__init__()

        self.texts = samples['text']
        if mode != 'test':
            self.labels = samples['label']
        self.data_processor = data_processor
        self.mode = mode

    def __getitem__(self, idx):
        text = self.texts[idx]

        if self.mode != 'test':
            label = self.labels[idx]
            return text, label
        else:
            return text

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


