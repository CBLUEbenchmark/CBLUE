import os
import json


def sts_commit_prediction(dataset, preds, output_dir, id2label):
    text1 = dataset.text1
    text2 = dataset.text2
    label = preds
    ids = dataset.ids
    category = dataset.category

    pred_result = []
    for item in zip(ids, text1, text2, label, category):
        tmp_dict = {'id': item[0], 'text1': item[1], 'text2': item[2],
                    'label': id2label[item[3]], 'category': item[4]}
        pred_result.append(tmp_dict)
    with open(os.path.join(output_dir, 'CHIP-STS_test.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(pred_result, indent=2, ensure_ascii=False))


def qic_commit_prediction(dataset, preds, output_dir, id2label):
    text1 = dataset.text
    label = preds
    ids = dataset.ids

    pred_result = []
    for item in zip(ids, text1, label):
        tmp_dict = {'id': item[0], 'query': item[1],
                    'label': id2label[item[2]]}
        pred_result.append(tmp_dict)
    with open(os.path.join(output_dir, 'KUAKE-QIC_test.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(pred_result, indent=2, ensure_ascii=False))


def qtr_commit_prediction(dataset, preds, output_dir, id2label):
    text1 = dataset.text1
    text2 = dataset.text2
    label = preds
    ids = dataset.ids

    pred_result = []
    for item in zip(ids, text1, text2, label):
        tmp_dict = {'id': item[0], 'text1': item[1], 'text2': item[2],
                    'label': id2label[item[3]]}
        pred_result.append(tmp_dict)
    with open(os.path.join(output_dir, 'KUAKE-QTR_test.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(pred_result, indent=2, ensure_ascii=False))


def qqr_commit_prediction(dataset, preds, output_dir, id2label):
    text1 = dataset.text1
    text2 = dataset.text2
    label = preds
    ids = dataset.ids

    pred_result = []
    for item in zip(ids, text1, text2, label):
        tmp_dict = {'id': item[0], 'query1': item[1], 'query2': item[2],
                    'label': id2label[item[3]]}
        pred_result.append(tmp_dict)
    with open(os.path.join(output_dir, 'KUAKE-QQR_test.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(pred_result, indent=2, ensure_ascii=False))


def ctc_commit_prediction(dataset, preds, output_dir, id2label):
    text1 = dataset.texts
    label = preds
    ids = dataset.ids

    pred_result = []
    for item in zip(ids, text1, label):
        tmp_dict = {'id': item[0], 'text': item[1],
                    'label': id2label[item[2]]}
        pred_result.append(tmp_dict)
    with open(os.path.join(output_dir, 'CHIP-CTC_test.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(pred_result, indent=2, ensure_ascii=False))


def ee_commit_prediction(dataset, preds, output_dir):
    orig_text = dataset.orig_text

    pred_result = []
    for item in zip(orig_text, preds):
        tmp_dict = {'text': item[0], 'entities': item[1]}
        pred_result.append(tmp_dict)
    with open(os.path.join(output_dir, 'CMeEE_test.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(pred_result, indent=2, ensure_ascii=False))


def cdn_commit_prediction(text, preds, recall_labels, output_dir, id2label):
    text1 = text
    labels = preds

    print(text1)
    print(preds)
    print(preds.shape)
    print(len(recall_labels))

    pred_result = []
    for text, label, recall_label in zip(text1, labels, recall_labels):
        tmp_pred = [id2label[recall_label[idx]] for idx, p in enumerate(label) if p == 1]
        if len(tmp_pred) == 0:
            tmp_pred = [text]
        tmp_pred = "##".join(tmp_pred)

        tmp_dict = {'text': text, 'normalized_result': tmp_pred}
        pred_result.append(tmp_dict)
    with open(os.path.join(output_dir, 'CHIP-CDN_test.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(pred_result, indent=2, ensure_ascii=False))
