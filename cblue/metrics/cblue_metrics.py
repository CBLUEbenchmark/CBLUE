
def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def sts_metric(preds, labels):
    return simple_accuracy(preds, labels)