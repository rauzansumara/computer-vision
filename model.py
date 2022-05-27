import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
from torch.nn import Parameter
from collections.abc import Iterable
import timm
from pathlib import Path
import pickle
from efficientnet_pytorch import EfficientNet

from dataset import COVIDxCTDataset
from augmentations import CovidDatasetAugmentations

from utils \
    import calculate_label_prediction, calculate_all_prediction, calculate_label_recall, calculate_f1, confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NNCovidModel:
    def __init__(self, model_type, class_names=['Normal', 'Pneumonia', 'COVID-19'],
                 use_pretrained=True, unfreeze_num=4):
        self.model_type = model_type
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.use_pretrained = use_pretrained
        self.unfreeze_num = unfreeze_num
        self.model, _ = self.initialize_model()
        self.model = self.model.to(device)

    def initialize_model(self):
        def set_freeze_by_idxs(model, idxs, freeze=True):
            if not isinstance(idxs, Iterable):
                idxs = [idxs]
            num_child = len(list(model.children()))
            idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
            for idx, child in enumerate(model.children()):
                if idx not in idxs:
                    continue
                for param in child.parameters():
                    param.requires_grad = not freeze
            return model

        def freeze_by_idxs(model, idxs, freeze=False):
            return set_freeze_by_idxs(model, idxs, freeze)

        def set_parameter_requires_grad(model):
            for param in model.parameters():
                param.requires_grad = False
            return model

        if self.model_type == 'resnet50':
            model_pre = models.wide_resnet50_2(pretrained=self.use_pretrained)
            model_pre.conv1.in_channels = 1
            model_pre.conv1.weight = Parameter(model_pre.conv1.weight[:, 1:2, :, :])
            model_pre = set_parameter_requires_grad(model_pre)
            num_ftrs = model_pre.fc.in_features
            model_pre.fc = nn.Linear(num_ftrs, self.num_classes)

            for i in range(self.unfreeze_num):
                model_pre.layer4 = freeze_by_idxs(model_pre.layer4, -i, False)
            for param in model_pre.fc.parameters():
                param.requires_grad = True
            input_size = 224
        elif self.model_type == 'resnet101':
            model_pre = models.resnet101(pretrained=self.use_pretrained)
            model_pre.conv1.in_channels = 1
            model_pre.conv1.weight = Parameter(model_pre.conv1.weight[:, 1:2, :, :])
            model_pre = set_parameter_requires_grad(model_pre)
            num_ftrs = model_pre.fc.in_features
            model_pre.fc = nn.Linear(num_ftrs, self.num_classes)

            for i in range(self.unfreeze_num):
                model_pre.layer4 = freeze_by_idxs(model_pre.layer4, -i, False)
            for param in model_pre.fc.parameters():
                param.requires_grad = True
            input_size = 224
        elif self.model_type == 'resnet152':
            model_pre = models.resnet152(pretrained=self.use_pretrained)
            model_pre.conv1.in_channels = 1
            model_pre.conv1.weight = Parameter(model_pre.conv1.weight[:, 1:2, :, :])
            model_pre = set_parameter_requires_grad(model_pre)
            num_ftrs = model_pre.fc.in_features
            model_pre.fc = nn.Linear(num_ftrs, self.num_classes)

            for i in range(self.unfreeze_num):
                model_pre.layer4 = freeze_by_idxs(model_pre.layer4, -i, False)
            for param in model_pre.fc.parameters():
                param.requires_grad = True
            input_size = 224
        elif self.model_type == 'inception_resnet':
            model_pre = timm.create_model('inception_resnet_v2', pretrained=True)
            model_pre.conv2d_1a.conv.in_channels = 1
            model_pre.conv2d_1a.conv.weight = Parameter(model_pre.conv2d_1a.conv.weight[:, 1:2, :, :])
            model_pre = set_parameter_requires_grad(model_pre)
            num_ftrs = model_pre.num_features
            model_pre.classif = nn.Linear(num_ftrs, self.num_classes)

            for i in range(self.unfreeze_num):
                model_pre.mixed_7a = freeze_by_idxs(model_pre.mixed_7a, -i, False)
            for param in model_pre.classif.parameters():
                param.requires_grad = True
            input_size = 224

        elif self.model_type == 'inception_v4':
            model_pre = timm.create_model('inception_v4', pretrained=True)
            model_pre.features[0].conv.in_channels = 1
            model_pre.features[0].conv.weight = Parameter(model_pre.features[0].conv.weight[:, 1:2, :, :])
            model_pre = set_parameter_requires_grad(model_pre)
            num_ftrs = model_pre.num_features
            model_pre.last_linear = nn.Linear(num_ftrs, self.num_classes)

            for i in range(self.unfreeze_num):
                model_pre.features[21] = freeze_by_idxs(model_pre.features[21], -i, False)
            for param in model_pre.last_linear.parameters():
                param.requires_grad = True
            input_size = 224
        # elif self.model_type == 'efficientnetB0':
        #     model_pre = models.efficientnet_b0(pretrained=self.use_pretrained)
        #     # model_pre.conv1.in_channels = 1
        #     # model_pre.conv1.weight = Parameter(model_pre.conv1.weight[:, 1:2, :, :])
        #     model_pre = set_parameter_requires_grad(model_pre)
        #
        #     num_ftrs = model_pre.classifier._modules['1'].in_features
        #     model_pre.classifier._modules['1'] = nn.Linear(num_ftrs, self.num_classes)
        #
        #     # for i in range(self.unfreeze_num):
        #       #   model_pre.layer4 = freeze_by_idxs(model_pre.layer4, -i, False)
        #     # for param in model_pre.fc.parameters():
        #         # param.requires_grad = True
        #     input_size = 224
        # elif self.model_type == 'efficientnetB3':
        #     model_pre = models.efficientnet_b3(pretrained=self.use_pretrained)
        #     model_pre.conv1.in_channels = 1
        #     model_pre.conv1.weight = Parameter(model_pre.conv1.weight[:, 1:2, :, :])
        #     model_pre = set_parameter_requires_grad(model_pre)
        #     num_ftrs = model_pre.fc.in_features
        #     model_pre.fc = nn.Linear(num_ftrs, self.num_classes)
        #
        #     for i in range(self.unfreeze_num):
        #         model_pre.layer4 = freeze_by_idxs(model_pre.layer4, -i, False)
        #     for param in model_pre.fc.parameters():
        #         param.requires_grad = True
        #     input_size = 224
        else:
            print('model not implemented')
            return None, None
        return model_pre, input_size

    def save(self, model_name):
        model_dir = '.'
        my_path = Path(model_dir + '/{}'.format(self.model_type))
        if not my_path.is_dir():
            os.mkdir(my_path)
        torch.save(self.model, model_dir + '/{}/{}_{}.pth'.format(self.model_type, self.model_type, model_name))
        model_conf = (self.model_type, self.class_names, self.use_pretrained, self.unfreeze_num)
        pickle.dump(model_conf, open('./{}/{}_{}_conf.conf'.format(self.model_type, self.model_type, model_name), 'wb'))
        return self.model

    def load(self, path):
        self.model = torch.load(path)
        self.model = self.model.to(device)
        conf_file_path = path[:-4] + '_conf.conf'
        retrieved_model_conf = pickle.load(open(conf_file_path, 'rb'))
        self.model_type = retrieved_model_conf[0]
        self.class_names = retrieved_model_conf[1]
        self.num_classes = len(self.class_names)
        self.use_pretrained = retrieved_model_conf[2]
        self.unfreeze_num = retrieved_model_conf[3]
        return self.model

    def train(self, epoch, num_epochs, criterion, optimizer, data_loaders, dataset_sizes, batch_size):
        self.model.train()
        print('-' * 100)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        running_loss = 0.0
        running_corrects = 0
        for idx, (inputs, labels) in enumerate(data_loaders['train']):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 100 == 99:
                print('train iteration:{},loss:{},acc:{}%'.format(idx, loss.item(),
                                                                  torch.sum(preds == labels.data) / batch_size * 100))
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']
        print('train_total Loss: {:.4f} Acc: {:.4f}%'.format(epoch_loss, epoch_acc * 100))
        return epoch_acc, epoch_loss

    def test(self, criterion, best_acc, data_loaders, dataset_sizes):
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        best_acc = best_acc
        best_model_wts = copy.deepcopy(self.model.state_dict())
        conf_matrix = torch.zeros(self.num_classes, self.num_classes)
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(data_loaders['val']):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                conf_matrix = confusion_matrix(outputs, labels, conf_matrix)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # plot_confusion_matrix(conf_matrix, classes=class_names, normalize=False, title='confusion matrix')

        epoch_loss = running_loss / dataset_sizes['val']
        epoch_acc = running_corrects.double() / dataset_sizes['val']
        print('val_total Loss: {:.4f} Acc: {:.4f}%'.format(epoch_loss, epoch_acc * 100))

        all_prediction = calculate_all_prediction(conf_matrix)
        print('all_prediction:{}'.format(all_prediction))
        label_prediction = []
        label_recall = []
        for i in range(self.num_classes):
            label_prediction.append(calculate_label_prediction(conf_matrix, i))
            label_recall.append(calculate_label_recall(conf_matrix, i))

        keys = self.class_names
        values = list(range(self.num_classes))
        dictionary = dict(zip(keys, values))
        for ei, i in enumerate(dictionary):
            print(ei, '\t', i, '\t', 'prediction=', label_prediction[ei], '%,\trecall=', label_recall[ei], '%,\tf1=',
                  calculate_f1(label_prediction[ei], label_recall[ei]))  # 输出每个类的，精确率，召回率，F1
        p = round(np.array(label_prediction).sum() / len(label_prediction), 2)
        r = round(np.array(label_recall).sum() / len(label_prediction), 2)
        print('MACRO-averaged:\nprediction=', p, '%,recall=', r, '%,f1=', calculate_f1(p, r))

        if epoch_acc > best_acc:
            best_acc = epoch_acc.item()
            best_model_wts = copy.deepcopy(self.model.state_dict())

        return best_model_wts, best_acc, epoch_acc.item(), epoch_loss


if __name__ == '__main__':
    covid_dataset = COVIDxCTDataset()
    train_df = covid_dataset.train_dataset()
    val_df = covid_dataset.val_dataset()
    test_df = covid_dataset.test_dataset()

    batch_size = 64
    covid_augmentations = CovidDatasetAugmentations(batch_size)
    data_loaders, dataset_sizes = covid_augmentations.create_dataset_iterators(train_df, val_df, test_df)

    num_epochs = 10
    model_type = 'resnet50'
    covid_NN = NNCovidModel(model_type)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(covid_NN.model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    acc_train = []
    loss_train = []
    acc_val = []
    loss_val = []
    for epoch in range(num_epochs):
        epoch_acc_train, epoch_loss_train = covid_NN.train(epoch, num_epochs, criterion, optimizer, data_loaders,
                                                           dataset_sizes, batch_size)
        acc_train.append(epoch_acc_train)
        loss_train.append(epoch_loss_train)
        best_model_wts, best_acc, epoch_acc_val, epoch_loss_val = covid_NN.test(criterion, best_acc, data_loaders,
                                                                                  dataset_sizes)
        acc_val.append(epoch_acc_val)
        loss_val.append(epoch_loss_val)

    print('*' * 100)
    print('best_acc:{}'.format(best_acc))
    print('*' * 100)
    covid_NN.save('{}_model_best_acc'.format(model_type))
    # torch.save(best_model_wts, 'resnet50_3_model_best_acc.pth')

    acc_train = [x.item() for x in acc_train]
    print(acc_train)
    print(loss_train)
    print(acc_val)
    print(loss_val)
    x = range(len(acc_train))

    plt.figure(1)
    plt.title('{} model accuracy'.format(model_type))
    plt.plot(x, acc_train, 'r', label='train')
    plt.plot(x, acc_val, 'b', label='val')
    plt.legend(loc="upper left")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    plt.figure(2)
    plt.title('{} model loss'.format(model_type))
    plt.plot(x, loss_train, 'r', label='train')
    plt.plot(x, loss_val, 'b', label='val')
    plt.legend(loc="upper left")
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.show()