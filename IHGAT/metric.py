import numpy as np

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    # print(preds)
    # print(labels)
    # exit()
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)