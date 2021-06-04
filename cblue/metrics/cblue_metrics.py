from sklearn.metrics import precision_recall_fscore_support


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def ee_metric(preds, labels):
    return precision_recall_fscore_support(y_pred=preds, y_true=labels, average='micro')


def ctc_metric(preds, labels):
    return precision_recall_fscore_support(y_pred=preds, y_true=labels, average='macro')


def sts_metric(preds, labels):
    return precision_recall_fscore_support(y_pred=preds, y_true=labels, average='macro')


def qic_metric(preds, labels):
    return simple_accuracy(preds, labels)


def qqr_metric(preds, labels):
    return simple_accuracy(preds, labels)


def qtr_metric(preds, labels):
    return simple_accuracy(preds, labels)


def er_metric(preds, labels):
    return precision_recall_fscore_support(y_pred=preds, y_true=labels, average='macro')


def re_metric(preds, labels):
    return precision_recall_fscore_support(y_pred=preds, y_true=labels, average='macro')


def cdn_cls_metric(preds_values, preds_ids, recall_labels, orig_labels):
    m = 0
    n = 0
    k = 0
    for pred_value, pred_id, recall_label, orig_label in zip(preds_values, preds_ids, recall_labels, orig_labels):
        tmp_pred = [recall_label[p_id] for p, p_id in zip(pred_value, pred_id) if p == 1]
        print(tmp_pred)
        k += len(set(tmp_pred) & set(orig_label))
        m += len(tmp_pred)
        n += len(orig_label)

    p = 1.0 * k / m
    r = 1.0 * k / n
    f1 = 2 * p * r / (p + r)
    return p, r, f1
