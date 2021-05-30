import os
import json
import pandas as pd
import jieba
import numpy as np
from gensim import corpora, models, similarities
from cblue.utils import load_json, load_dict, write_dict, str_q2b


class EEDataProcessor(object):
    def __init__(self, root, is_lower=True, no_entity_label='O'):
        self.task_data_dir = os.path.join(root, 'CMeEE')
        self.train_path = os.path.join(self.task_data_dir, 'CMeEE_train.json')
        self.dev_path = os.path.join(self.task_data_dir, 'CMeEE_dev.json')
        self.test_path = os.path.join(self.task_data_dir, 'CMeEE_test.json')

        self.label_map_cache_path = os.path.join(self.task_data_dir, 'CMeEE_label_map.dict')
        self.label2id = None
        self.id2label = None
        self._get_labels()

        self.is_lower = is_lower
        self.no_entity_label = no_entity_label

    def get_train_samples(self):
        return self._pre_process(self.train_path, is_predict=False)

    def get_dev_samples(self):
        return self._pre_process(self.dev_path, is_predict=False)

    def get_test_samples(self):
        return self._pre_process(self.test_path, is_predict=True)

    def _get_labels(self):
        if os.path.exists(self.label_map_cache_path):
            label_map = load_dict(self.label_map_cache_path)
        else:
            label_set = set()
            samples = load_json(self.train_path)
            for sample in samples:
                for entity in sample["entities"]:
                    label_set.add(entity['type'])
            label_set = sorted(label_set)
            labels = [self.no_entity_label]
            for label in label_set:
                labels.extend(["B-{}".format(label), "I-{}".format(label)])
            label_map = {idx: label for idx, label in enumerate(labels)}
            write_dict(self.label_map_cache_path, label_map)
        self.id2label = label_map
        self.label2id = {val: key for key, val in self.id2label.items()}

    def _pre_process(self, path, is_predict):
        def label_data(data, start, end, _type):
            """label_data"""
            for i in range(start, end + 1):
                suffix = "B-" if i == start else "I-"
                data[i] = "{}{}".format(suffix, _type)
            return data

        outputs = {'text': [], 'label': []}
        samples = load_json(path)
        for data in samples:
            if self.is_lower:
                text_a = ["，" if t == " " or t == "\n" or t == "\t" else t
                          for t in list(data["text"].lower())]
            else:
                text_a = ["，" if t == " " or t == "\n" or t == "\t" else t
                          for t in list(data["text"])]
            text_a = "".join(text_a)
            outputs['text'].append(text_a)
            if not is_predict:
                labels = ["O"] * len(text_a)
                for entity in data['entities']:
                    start_idx, end_idx, type = entity['start_idx'], entity['end_idx'], entity['type']
                    labels = label_data(labels, start_idx, end_idx, type)
                outputs['label'].append('\002'.join(labels))
        return outputs


class REDataProcessor(object):
    def __init__(self, root):
        self.task_data_dir = os.path.join(root, 'CMeIE')
        self.train_path = os.path.join(self.task_data_dir, 'CMeIE_train.json')
        self.dev_path = os.path.join(self.task_data_dir, 'CMeIE_dev.json')
        self.test_path = os.path.join(self.task_data_dir, 'CMeIE_test.json')

        self.schema_path = os.path.join(self.task_data_dir, '53_schemas.json')
        self.pre_sub_obj = None
        self.predicate2id = None
        self.id2predicate = None
        self.s_entity_type = None
        self.o_entity_type = None
        self._load_schema()

    def get_train_samples(self):
        return self._pre_process(self.train_path)

    def get_dev_samples(self):
        return self._pre_process(self.dev_path)

    def get_test_samples(self, path):
        """ Need new test file generated from the result of ER prediction
        """
        return self._pre_process(path)

    def _pre_process(self, path):
        with open(path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            result = {'text': [], 'label': [], 'flag': []}
            for line in lines:
                data = json.loads(line)
                text = data['text']
                s_dict = {}  # sub_type : sub
                o_dict = {}  # obj_type : obj
                spo_dict = {}  # sub|obj : predicate|obj_type
                for spo in data['spo_list']:
                    sub = spo['subject']
                    # s_dict[spo['subject_type']] = spo['subject']
                    s_dict[spo['subject']] = spo['subject_type']
                    pre = spo['predicate']
                    p_o = pre + '|' + spo['object_type']['@value']
                    spo_dict[sub + '|' + spo['object']['@value']] = p_o
                    # o_dict[spo['object_type']['@value']] = spo['object']['@value']
                    o_dict[spo['object']['@value']] = spo['object_type']['@value']
                for sv, sk in s_dict.items():
                    for ov, ok in o_dict.items():
                        s_flag = self.s_entity_type[sk]  # '<s>, </s>'
                        o_flag = self.o_entity_type[ok]
                        s_start = self.search(text, sv)
                        s_end = s_start + len(sv)
                        text1 = text[:s_start] + s_flag[0] + sv + s_flag[1] + text[s_end:]
                        o_start = self.search(text1, ov)
                        o_end = o_start + len(ov)
                        text2 = text1[:o_start] + o_flag[0] + ov + o_flag[1] + text1[o_end:]
                        if sv + '|' + ov in spo_dict.keys():
                            labels = self.predicate2id[spo_dict[sv + '|' + ov]]
                        else:
                            labels = 0
                        result['text'].append(text2)
                        result['label'].append(labels)
                        result['flag'].append((s_flag[0], o_flag[0]))
            return result

    def _load_schema(self, ):
        with open(self.schema_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            predicate_list = ["无关系"]
            s_entity = []
            o_entity = []
            pre_sub_obj = {}
            for line in lines:
                data = json.loads(line)
                if data['subject_type'] not in s_entity:
                    s_entity.append(data['subject_type'])
                if data['object_type'] not in o_entity:
                    o_entity.append(data['object_type'])
                predicate_list.append(data['predicate'] + '|' + data['object_type'])
                pre_sub_obj[data['predicate'] + '|' + data['object_type']] = [data['subject_type'], data['object_type']]

            s_entity_type = {}
            for i, e in enumerate(s_entity):  # 主语
                s_entity_type[e] = ('<s>', '</s>')  # unused4 unused5

            o_entity_type = {}
            for i, e in enumerate(o_entity):
                o_entity_type[e] = ('<o>', '</o>')

            predicate2id = {v: i for i, v in enumerate(predicate_list)}
            id2predicate = {i: v for i, v in enumerate(predicate_list)}

            self.pre_sub_obj = pre_sub_obj
            self.predicate2id = predicate2id
            self.id2predicate = id2predicate
            self.s_entity_type = s_entity_type
            self.o_entity_type = o_entity_type

    def search(self, sequence, pattern):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回0。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return 0


class ERDataProcessor(object):
    def __init__(self, root):
        self.task_data_dir = os.path.join(root, 'CMeIE')
        self.train_path = os.path.join(self.task_data_dir, 'CMeIE_train.json')
        self.dev_path = os.path.join(self.task_data_dir, 'CMeIE_dev.json')
        self.test_path = os.path.join(self.task_data_dir, 'CMeIE_test.json')

    def get_train_samples(self):
        return self._pre_process(self.train_path, is_predict=False)

    def get_dev_samples(self):
        return self._pre_process(self.dev_path, is_predict=False)

    def get_test_samples(self):
        return self._pre_process(self.test_path, is_predict=True)

    def _pre_process(self, path, is_predict=False):
        with open(path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            result = {'text': [], 'spo_list': []}
            for line in lines:
                data = json.loads(line)
                text = data['text']

                if not is_predict:
                    one_spo_list = []
                    for spo in data['spo_list']:
                        s = spo['subject']
                        p = spo['predicate']
                        tmp_ob_type = [v for k, v in spo['object_type'].items()]
                        tmp_ob = [v for k, v in spo['object'].items()]
                        for i in range(len(tmp_ob)):
                            p_o = p + '|' + tmp_ob_type[i]
                            one_spo_list.append((s, p_o, tmp_ob[i]))
                else:
                    one_spo_list = None
                result['text'].append(text)
                result['spo_list'].append(one_spo_list)

            return result

    def search(self, sequence, pattern):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回0。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return 0


class CDNDataProcessor(object):
    def __init__(self, root, recall_k=200, negative_sample=3):
        self.task_data_dir = os.path.join(root, 'CHIP-CDN')
        self.train_path = os.path.join(self.task_data_dir, 'CHIP-CDN_train.json')
        self.dev_path = os.path.join(self.task_data_dir, 'CHIP-CDN_dev.json')
        self.test_path = os.path.join(self.task_data_dir, 'CHIP-CDN_test.json')

        self.label_path = os.path.join(self.task_data_dir, '国际疾病分类 ICD-10北京临床版v601.xlsx')
        self.label2id, self.id2label = self._get_labels()

        self.recall_k = recall_k
        self.negative_sample = negative_sample

    def get_train_samples(self, dtype='cls', do_augment=1):
        """
        do_augment: data augment
        """
        samples = self._pre_process(self.train_path, is_predict=False)
        if dtype == 'cls':
            outputs = self._get_cls_samples(orig_samples=samples, is_predict=False, do_augment=do_augment)
        else:
            outputs = self._get_num_samples(orig_sample=samples, is_predict=False)
        return outputs

    def get_dev_samples(self, dtype='cls'):
        samples = self._pre_process(self.dev_path, is_predict=False)
        if dtype == 'cls':
            outputs = self._get_cls_samples(orig_samples=samples, is_predict=False)
        else:
            outputs = self._get_num_samples(orig_sample=samples, is_predict=False)
        return outputs, samples

    def get_test_samples(self, dtype='cls'):
        samples = self._pre_process(self.test_path, is_predict=True)
        if dtype == 'cls':
            outputs = self._get_cls_samples(orig_samples=samples, is_predict=True)
        else:
            outputs = self._get_num_samples(orig_sample=samples, is_predict=True)

        return outputs, samples

    def _pre_process(self, path, is_predict=False):
        samples = load_json(path)
        outputs = {'text': [], 'label': []}

        for sample in samples:
            text = self._process_single_sentence(sample['text'], mode="text")
            if is_predict:
                outputs['text'].append(text)
            else:
                label = self._process_single_sentence(sample['normalized_result'], mode="label")
                outputs['label'].append([label_ for label_ in label.split("##")])
                outputs['text'].append(text)
        return outputs

    def _get_cls_samples(self, orig_samples, is_predict=False, do_augment=1):
        outputs = {'text1': [], 'text2': [], 'label': []}

        texts = orig_samples['text']
        recall_samples_idx, recall_samples_scores = self._recall(texts)

        if not is_predict:
            labels = orig_samples['label']
            for i in range(do_augment):
                for text, label in zip(texts, labels):
                    for label_ in label:
                        outputs['text1'].append(text)
                        outputs['text2'].append(label_)
                        outputs['label'].append(1)

            cnt_label = 0
            for text, orig_label, recall_label in zip(texts, labels, recall_samples_idx):
                orig_label_ids = [self.label2id[label] for label in orig_label]
                for label_ in recall_label:
                    if label_ not in orig_label_ids:
                        outputs['text1'].append(text)
                        outputs['text2'].append(self.id2label[label_])
                        outputs['label'].append(0)
                        cnt_label += 1
                    if cnt_label >= self.negative_sample:
                        break
        else:
            for text, recall_label in zip(texts, recall_samples_idx):
                for label_ in recall_label:
                    outputs['text1'].append(text)
                    outputs['text2'].append(self.id2label[label_])

        return outputs

    def _get_num_samples(self, orig_sample, is_predict=False):
        outputs = {'text1': [], 'text2': [], 'label': []}

        if not is_predict:
            texts = orig_sample['text']
            labels = orig_sample['label']

            for text, label in zip(texts, labels):
                outputs['text1'].append(text)
                outputs['label'].append(len(label))
        else:
            outputs['text1'] = orig_sample['text']

        return outputs

    def _recall(self, texts):
        all_label_list = []
        for idx in range(len(self.label2id.keys())):
            all_label_list.append(list(jieba.cut(self.id2label[idx])))

        dictionary = corpora.Dictionary(all_label_list)  # 词典
        corpus = [dictionary.doc2bow(doc) for doc in all_label_list]  # 语料库
        tfidf = models.TfidfModel(corpus)  # 建立模型
        index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))  # 相似度

        recall_scores_idx = np.zeros((len(texts), self.recall_k), dtype=np.int)

        recall_scores = np.zeros((len(texts), self.recall_k))
        for i, x in enumerate(texts):
            x_scores = np.zeros(len(self.label2id.keys()))
            # lambdaCooc: 字词为级别的相似 0-1
            x_split = list(jieba.cut(x))
            x_vec = dictionary.doc2bow(x_split)
            x_sim = index[tfidf[x_vec]]  # 相似度分数 (1, labels)

            # dice 字符级别的相似性 0-1
            # x : str
            x_dices = np.zeros(len(self.label2id.keys()))
            x_set = set(x)
            for j in range(len(self.label2id.keys())):
                y_set = set(self.id2label[j])
                x_dices[j] = len(x_set & y_set) / min(len(x_set), len(y_set))
            x_scores = x_sim + x_dices
            x_scores_idx = np.argsort(x_scores)[:len(x_scores) - self.recall_k - 1:-1]  # 由大到小排序,取前K个
            x_scores = np.sort(x_scores)[:len(x_scores) - self.recall_k - 1:-1]
            recall_scores[i] += x_scores
            recall_scores_idx[i] += x_scores_idx
        return recall_scores_idx, recall_scores

    def _get_labels(self):
        df = pd.read_excel(self.label_path, header=None)
        normalized_word = df[1].unique().tolist()
        label2id = {word: idx for idx, word in enumerate(normalized_word)}
        id2label = {idx: word for idx, word in enumerate(normalized_word)}

        num_label = len(label2id.keys())
        samples = self._pre_process(self.train_path)
        for labels in samples['label']:
            for label in labels:
                if label not in label2id:
                    label2id[label] = num_label
                    id2label[num_label] = label
                    num_label += 1

        samples = self._pre_process(self.dev_path)
        for labels in samples['label']:
            for label in labels:
                if label not in label2id:
                    label2id[label] = num_label
                    id2label[num_label] = label
                    num_label += 1
        return label2id, id2label

    def _process_single_sentence(self, sentence, mode='text'):
        sentence = str_q2b(sentence)
        sentence = sentence.strip('"')
        if mode == "text":
            sentence = sentence.replace("\\", ";")
            sentence = sentence.replace(",", ";")
            sentence = sentence.replace("、", ";")
            sentence = sentence.replace("?", ";")
            sentence = sentence.replace(":", ";")
            sentence = sentence.replace(".", ";")
            sentence = sentence.replace("/", ";")
            sentence = sentence.replace("~", "-")
        return sentence


class CTCDataProcessor(object):
    def __init__(self, root):
        self.task_data_dir = os.path.join(root, 'CHIP-CTC')
        self.train_path = os.path.join(self.task_data_dir, 'CHIP-CTC_train.json')
        self.dev_path = os.path.join(self.task_data_dir, 'CHIP-CTC_dev.json')
        self.test_path = os.path.join(self.task_data_dir, 'CHIP-CTC_test.json')

        self.label_path = os.path.join(self.task_data_dir, 'category.xlsx')
        self.label_list = self._get_labels()
        self.label2id = {label: idx for idx, label in enumerate(self.label_list)}
        self.id2label = {idx: label for idx, label in enumerate(self.label_list)}

    def get_train_samples(self):
        return self._pre_process(self.train_path, is_predict=False)

    def get_dev_samples(self):
        return self._pre_process(self.dev_path, is_predict=False)

    def get_test_samples(self):
        return self._pre_process(self.test_path, is_predict=True)

    def _pre_process(self, path, is_predict=False):
        samples = load_json(path)
        outputs = {'text': [], 'label': []}
        for sample in samples:
            outputs['text'].append(sample['text'])
            if not is_predict:
                outputs['label'].append(self.label2id[sample['label']])

        return outputs

    def _get_labels(self):
        data = pd.read_excel(self.label_path)
        labels = data['Label Name'].unique().tolist()
        return labels


class STSDataProcessor(object):
    def __init__(self, root):
        self.task_data_dir = os.path.join(root, 'CHIP-STS')
        self.train_path = os.path.join(self.task_data_dir, 'CHIP-STS_train.json')
        self.dev_path = os.path.join(self.task_data_dir, 'CHIP-STS_dev.json')
        self.test_path = os.path.join(self.task_data_dir, 'CHIP-STS_test.json')

        self.label2id = {'0': 0, '1': 1}
        self.id2label = {0: '0', 1: '1'}

    def get_train_sample(self):
        return self._pre_process(self.train_path, is_predict=False)

    def get_dev_sample(self):
        return self._pre_process(self.dev_path, is_predict=False)

    def get_test_sample(self):
        return self._pre_process(self.test_path, is_predict=True)

    def _pre_process(self, path, is_predict):
        samples = load_json(path)
        outputs = {'text1': [], 'text2': [], 'label': []}
        for sample in samples:
            outputs['text1'].append(sample['text1'])
            outputs['text2'].append(sample['text2'])
            if not is_predict:
                outputs['label'].append(self.label2id[sample['label']])
        return outputs


class QQRDataProcessor(object):
    def __init__(self, root):
        self.task_data_dir = os.path.join(root, 'KUAKE-QQR')
        self.train_path = os.path.join(self.task_data_dir, 'KUAKE-QQR_train.json')
        self.dev_path = os.path.join(self.task_data_dir, 'KUAKE-QQR_dev.json')
        self.test_path = os.path.join(self.task_data_dir, 'KUAKE-QQR_test.json')

        self.label2id = {'0': 0, '1': 1, '2': 2}
        self.id2label = {0: '0', 1: '1', 2: '2'}

    def get_train_sample(self):
        return self._pre_process(self.train_path, is_predict=False)

    def get_dev_sample(self):
        return self._pre_process(self.dev_path, is_predict=False)

    def get_test_sample(self):
        return self._pre_process(self.test_path, is_predict=True)

    def _pre_process(self, path, is_predict):
        samples = load_json(path)
        outputs = {'text1': [], 'text2': [], 'label': []}
        for sample in samples:
            outputs['text1'].append(sample['query1'])
            outputs['text2'].append(sample['query2'])
            if not is_predict:
                outputs['label'].append(self.label2id[sample['label']])
        return outputs


class QICDataProcessor(object):
    def __init__(self, root):
        self.task_data_dir = os.path.join(root, 'KUAKE-QIC')
        self.train_path = os.path.join(self.task_data_dir, 'KUAKE-QIC_train.json')
        self.dev_path = os.path.join(self.task_data_dir, 'KUAKE-QIC_dev.json')
        self.test_path = os.path.join(self.task_data_dir, 'KUAKE-QIC_test.json')

        self.label_list = ['疾病表述', '指标解读', '医疗费用', '治疗方案', '功效作用', '病情诊断',
                           '其他', '注意事项', '病因分析', '就医建议', '后果表述']
        self.label2id = {label: idx for idx, label in enumerate(self.label_list)}
        self.id2label = {idx: label for idx, label in enumerate(self.label_list)}

    def get_train_sample(self):
        return self._pre_process(self.train_path, is_predict=False)

    def get_dev_sample(self):
        return self._pre_process(self.dev_path, is_predict=False)

    def get_test_sample(self):
        return self._pre_process(self.test_path, is_predict=True)

    def _pre_process(self, path, is_predict):
        samples = load_json(path)
        outputs = {'text': [], 'label': []}
        for sample in samples:
            outputs['text'].append(sample['query'])
            if not is_predict:
                outputs['label'].append(self.label2id[sample['label']])
        return outputs


class QTRDataProcessor(object):
    def __init__(self, root):
        self.task_data_dir = os.path.join(root, 'KUAKE-QTR')
        self.train_path = os.path.join(self.task_data_dir, 'KUAKE-QTR_train.json')
        self.dev_path = os.path.join(self.task_data_dir, 'KUAKE-QTR_dev.json')
        self.test_path = os.path.join(self.task_data_dir, 'KUAKE-QTR_test.json')

        self.label2id = {'0': 0, '1': 1, '2': 2, '3': 3}
        self.id2label = {0: '0', 1: '1', 2: '2', 3: '3'}

    def get_train_sample(self):
        return self._pre_process(self.train_path, is_predict=False)

    def get_dev_sample(self):
        return self._pre_process(self.dev_path, is_predict=False)

    def get_test_sample(self):
        return self._pre_process(self.test_path, is_predict=True)

    def _pre_process(self, path, is_predict):
        samples = load_json(path)
        outputs = {'text1': [], 'text2': [], 'label': []}
        for sample in samples:
            outputs['text1'].append(sample['query'])
            outputs['text2'].append(sample['title'])
            if not is_predict:
                outputs['label'].append(self.label2id[sample['label']])
        return outputs




