from transformers import BertTokenizer
from torch.utils.data import DataLoader
from cblue.data import EEDataProcessor, EEDataset, REDataProcessor, REDataset, \
    ERDataProcessor, ERDataset, CTCDataProcessor, CTCDataset, CDNDataProcessor, \
    STSDataProcessor, STSDataset, QQRDataProcessor, QQRDataset, QICDataProcessor, \
    QICDataset, QTRDataset, QTRDataProcessor

DATA_ROOT = '/data/CBLUE'
MODEL_DATA_DIR = 'data/model_data/chinese-bert-wwm'


def test_ee_dataset():
    data_processor = EEDataProcessor(root=DATA_ROOT)
    train_samples = data_processor.get_train_samples()

    print(data_processor.label2id)
    print(data_processor.id2label)

    tokenizer = BertTokenizer.from_pretrained(MODEL_DATA_DIR)
    dataset = EEDataset(samples=train_samples, data_processor=data_processor,
                        tokenizer=tokenizer)

    print(dataset[0])

    data_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    for item in data_loader:
        print(item)
        break


def test_re_dataset():
    data_processor = REDataProcessor(root=DATA_ROOT)
    print(data_processor.predicate2id)
    print(data_processor.id2predicate)
    print(data_processor.s_entity_type)
    print(data_processor.o_entity_type)

    re_train_samples = data_processor.get_train_samples()
    print(re_train_samples['text'][0])
    print(re_train_samples['flag'][0])
    print(re_train_samples['label'][0])

    print(re_train_samples['text'][1])
    print(re_train_samples['flag'][1])
    print(re_train_samples['label'][1])

    tokenizer = BertTokenizer.from_pretrained(MODEL_DATA_DIR)
    dataset = REDataset(samples=re_train_samples, data_processor=data_processor,
                        tokenizer=tokenizer)
    print(dataset[0])

    data_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    for item in data_loader:
        print(item)
        break


def test_er_dataset():
    data_processor = ERDataProcessor(root=DATA_ROOT)

    train_samples = data_processor.get_train_samples()
    print(train_samples['text'][0])
    print(train_samples['spo_list'][0])

    tokenizer = BertTokenizer.from_pretrained(MODEL_DATA_DIR)
    dataset = ERDataset(samples=train_samples, data_processor=data_processor,
                        tokenizer=tokenizer)

    print(dataset[0])


def test_cdn_dataset():
    data_processor = CDNDataProcessor(root=DATA_ROOT)

    # train_cls_samples = data_processor.get_train_samples(dtype='cls', do_augment=1)
    # print(train_cls_samples['text1'])
    # print(train_cls_samples['text2'][0])
    # print(train_cls_samples['label'][0])

    train_num_samples = data_processor.get_train_samples(dtype='num', do_augment=1)
    print(train_num_samples['text1'][1])
    print(train_num_samples['label'][1])


def test_ctc_dataset():
    data_processor = CTCDataProcessor(root=DATA_ROOT)
    print(data_processor.label2id)
    print(data_processor.id2label)

    train_samples = data_processor.get_train_samples()
    print(train_samples['text'][0])
    print(train_samples['label'][1])

    dataset = CTCDataset(samples=train_samples, data_processor=data_processor)
    print(dataset[0])
    data_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    for item in data_loader:
        print(item)
        break


def test_sts_dataset():
    data_processor = STSDataProcessor(root=DATA_ROOT)

    train_samples = data_processor.get_train_sample()
    print(train_samples['text1'][0])
    print(train_samples['text2'][0])
    print(train_samples['label'][0])

    train_dataset = STSDataset(samples=train_samples, data_processor=data_processor)
    print(train_dataset[0])


def test_qqr_dataset():
    data_processor = QQRDataProcessor(root=DATA_ROOT)

    train_samples = data_processor.get_train_sample()
    print(train_samples['text1'][0])
    print(train_samples['text2'][0])
    print(train_samples['label'][0])

    train_dataset = QQRDataset(samples=train_samples, data_processor=data_processor)
    print(train_dataset[0])


def test_qic_dataset():
    data_processor = QICDataProcessor(root=DATA_ROOT)

    train_samples = data_processor.get_train_sample()
    print(train_samples['text'][0])
    print(train_samples['label'][0])

    train_dataset = QICDataset(samples=train_samples, data_processor=data_processor)
    print(train_dataset[0])


def test_qtr_dataset():
    data_processor = QTRDataProcessor(root=DATA_ROOT)

    train_samples = data_processor.get_train_sample()
    print(train_samples['text1'][0])
    print(train_samples['text2'][0])
    print(train_samples['label'][0])

    train_dataset = QTRDataset(samples=train_samples, data_processor=data_processor)
    print(train_dataset[0])


if __name__ == '__main__':
    print('---------- EE ----------')
    test_ee_dataset()
    print('---------- RE ----------')
    test_re_dataset()
    print('---------- ER ----------')
    test_er_dataset()
    print('---------- CTC ---------')
    test_ctc_dataset()
    print('---------- CDN ----------')
    test_cdn_dataset()
    print('---------- STS ---------')
    test_sts_dataset()
    print('---------- QQR ---------')
    test_qqr_dataset()
    print('---------- QIC ---------')
    test_qic_dataset()
    print('---------- QTR ---------')
    test_qtr_dataset()
