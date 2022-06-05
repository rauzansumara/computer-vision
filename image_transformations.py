from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CovidDataset(Dataset):
    def __init__(self, dataset_df, transform=None):
        self.dataset_df = dataset_df
        self.transform = transform

    def __len__(self):
        return self.dataset_df.shape[0]

    def __getitem__(self, idx):
        image_name = self.dataset_df['filename'][idx]
        img = Image.open(image_name)
        label = self.dataset_df['label'][idx]

        if self.transform:
            img = self.transform(img)
        return img, label


class CovidDatasetAugmentations:
    def __init__(self, batch_size=64):
        self.transform = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.6349431], [0.32605055])
            ]),
            'testOrVal': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.63507175], [0.3278614])
            ])
        }
        self.batch_size = batch_size

    def create_dataset_iterators(self, train_df, val_df, test_df):
        dataset_names = ['train', 'val', 'test']
        image_transforms = {'train': self.transform['train'], 'val': self.transform['testOrVal'],
                            'test': self.transform['testOrVal']}

        train_dataset = CovidDataset(train_df, transform=image_transforms['train'])
        val_dataset = CovidDataset(val_df, transform=image_transforms['val'])
        test_dataset = CovidDataset(test_df, transform=image_transforms['test'])

        image_dataset = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}

        data_loaders = {x: DataLoader(image_dataset[x], batch_size=self.batch_size, shuffle=True, num_workers=4)
                        for x in dataset_names}

        dataset_sizes = {x: len(image_dataset[x]) for x in dataset_names}
        return data_loaders, dataset_sizes
