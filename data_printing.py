from tabulate import tabulate
import torch
import torch.nn as nn

from model import NNCovidModel
from split_dataset import CovidDataset
from image_transformations import CovidDatasetAugmentations

pytorch_model_dict = dict({'Resnet50': ['resnet50', 'resnet50/resnet50_resnet50_model_best_acc.pth'],
                            'Inception v4': ['inception_v4', 'inception_v4'
                                                            '/inception_v4_inception_v4_model_best_acc.pth'],
                            'Inception Resnet v2': ['inception_resnet', 'inception_resnet'
                                                                        '/inception_resnet_inception_resnet_model_best_acc.pth'],
                            'Xception': ['xception', 'xception'
                                                            '/xception_xception_model_best_acc.pth'],
                            'Vgg16': ['vgg16', 'vgg16/vgg16_vgg16_model_best_acc.pth']})


def print_number_of_parameters():
    column_names = ['Model', 'Total Params', 'Trainable Params', 'Non-trainable Params']
    parameter_table = [column_names]

    for model_name, [model_type, model_path] in pytorch_model_dict.items():
        covid_nn = NNCovidModel(model_type)
        model = covid_nn.load(model_path)
        pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        pytorch_nontrainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        pytorch_total_params = pytorch_trainable_params + pytorch_nontrainable_params
        model_data = [model_name, pytorch_total_params, pytorch_trainable_params, pytorch_nontrainable_params]
        parameter_table.append(model_data)

    print(tabulate(parameter_table, headers='firstrow', tablefmt='fancy_grid'))

    return


def print_metrics():
    covid_dataset = CovidDataset()
    train_df = covid_dataset.train_dataset()
    val_df = covid_dataset.val_dataset()
    test_df = covid_dataset.test_dataset()

    batch_size = 64
    covid_augmentations = CovidDatasetAugmentations(batch_size)
    data_loaders, dataset_sizes = covid_augmentations.create_dataset_iterators(train_df, val_df, test_df)
    criterion = nn.CrossEntropyLoss()
    index = 1
    for model_name, [model_type, model_path] in pytorch_model_dict.items():
        print('{} model metrics:'.format(model_name))
        best_acc = 0
        covid_nn = NNCovidModel(model_type)
        covid_nn.load(model_path)
        covid_nn.test(criterion, best_acc, data_loaders, dataset_sizes, plot_conf_matrix=True, index=index)
        index += 1
    return


if __name__ == '__main__':
    print_number_of_parameters()
    print_metrics()
