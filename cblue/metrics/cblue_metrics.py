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


def cdn_cls_metric(preds, labels):
    return precision_recall_fscore_support(y_pred=preds, y_true=labels, average='macro')


def cdn_num_metric(preds, labels):
    return precision_recall_fscore_support(y_pred=preds, y_true=labels, average='macro')
