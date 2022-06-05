import numpy as np
import torch
import matplotlib.pyplot as plt
import itertools


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[t, p] += 1
    return conf_matrix


def calculate_all_precision(conf_matrix):
    total_sum = conf_matrix.sum()
    correct_sum = (np.diag(conf_matrix)).sum()
    prediction = round(100 * float(correct_sum) / float(total_sum), 2)
    return prediction


def calculate_label_precision(conf_matrix, labelidx):
    label_total_sum = conf_matrix.sum(axis=0)[labelidx]
    label_correct_sum = conf_matrix[labelidx][labelidx]
    precision = 0
    if label_total_sum != 0:
        precision = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
    return precision


def calculate_label_recall(conf_matrix, labelidx):
    label_total_sum = conf_matrix.sum(axis=1)[labelidx]
    label_correct_sum = conf_matrix[labelidx][labelidx]
    recall = 0
    if label_total_sum != 0:
        recall = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
    return recall


def calculate_f1(prediction, recall):
    if (prediction + recall) == 0:
        return 0
    return round(2 * prediction * recall / (prediction + recall), 2)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, index=1):
    cm = cm.numpy()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        cm = cm.astype('int')
        print('Confusion matrix, without normalization')
    #     print(cm)
    plt.figure(index)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '{:.2f}' if normalize else '{}'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(i, j, fmt.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == '__main__':
    cm = [[12109, 6, 130], [48, 7251, 96], [39, 225, 5754]]
    classes=['Normal', 'Pneumonia', 'COVID-19']
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix CovidNet')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '{}'
    thresh = 12109 / 2.
    for i, j in itertools.product(range(3), range(3)):
        plt.text(i, j, fmt.format(cm[i][j]), horizontalalignment="center",
             color="white" if cm[i][j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()