import pandas as pd
from sklearn.utils import resample
from sklearn.utils import shuffle


class CovidDataset:
    """COVIDx CT dataset class, which handles construction of train/validation/test datasets"""
    def __init__(self, train_df_path='../input/covidxct/train_COVIDx_CT-2A.txt',
                 val_df_path='../input/covidxct/val_COVIDx_CT-2A.txt',
                 test_df_path='../input/covidxct/test_COVIDx_CT-2A.txt',  image_path='../input/covidxct/2A_images/'):
        self.train_df_path = train_df_path
        self.val_df_path = val_df_path
        self.test_df_path = test_df_path
        self.image_path = image_path

    def train_dataset(self):
        """Returns training dataset"""
        train_df = self._make_dataset(self.train_df_path, self.image_path)
        train_df = self._balance_dataset(train_df)
        train_df = shuffle(train_df)
        return train_df.reset_index()

    def val_dataset(self):
        """Returns validation dataset"""
        val_df = self._make_dataset(self.val_df_path, self.image_path)
        val_df = self._balance_dataset(val_df)
        val_df = shuffle(val_df)
        return val_df.reset_index()

    def test_dataset(self):
        """Returns testing dataset"""
        test_df = self._make_dataset(self.test_df_path, self.image_path)
        test_df = shuffle(test_df)
        return test_df.reset_index()

    def _make_dataset(self, file_path, image_path):
        dataset = pd.read_csv(file_path, sep=" ", header=None)
        dataset.columns = ['filename', 'label', 'xmin', 'ymin', 'xmax', 'ymax']
        dataset = dataset.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)
        dataset['filename'] = image_path + dataset['filename']

        return dataset

    def _balance_dataset(self, dataset, n_samples=None):
        normal = dataset[dataset['label'] == 0]
        pneumonia = dataset[dataset['label'] == 1]
        covid = dataset[dataset['label'] == 2]
        if n_samples is None:
            n_samples = min(len(normal), len(pneumonia), len(covid))
        normal_resampled = resample(normal, replace=True, n_samples=n_samples, random_state=0)
        pneumonia_resampled = resample(pneumonia, replace=True, n_samples=n_samples, random_state=0)
        covid_resampled = resample(covid, replace=True, n_samples=n_samples, random_state=0)
        dataset = pd.concat([normal_resampled, pneumonia_resampled, covid_resampled])

        return dataset
