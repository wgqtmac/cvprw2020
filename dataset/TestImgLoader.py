import os
import torch
import torch.utils.data as data
from PIL import Image


def default_loader(path):
    return Image.open(path).convert('RGB')


def YCbCr_loader(path):
    return Image.open(path).convert('YCbCr')


class ImgLoader(data.Dataset):
    def __init__(self, root_folder, list_file, transform=None, loader1=default_loader, loader2=YCbCr_loader,
                 stage='Train'):
        self.root_folder = root_folder
        self.loader1 = loader1
        self.loader2 = YCbCr_loader
        self.transform = transform

        items = []

        if stage == 'Train':
            fp_items = [line.rstrip('\n').split(' ') for line in open(list_file)]
            for file_name, label in fp_items:
                if os.path.isfile(os.path.join(root_folder, file_name)):
                    tup = (file_name, int(label))
                    items.append(tup)
        elif stage == 'Test':

            f = open(list_file)
            lines = f.readlines()

            for line in lines:
                line = line.strip().split(' ')
                test_dir = os.listdir(os.path.join(root_folder,line[0],'profile'))
                for i in range(len(test_dir)):
                    items.append(os.path.join(line[0], 'profile', test_dir[i]))
        else:

            f = open(list_file)
            lines = f.readlines()

            for line in lines:
                line = line.strip().split(' ')
                items.append(line[0])

            # fp_items = [line.rstrip('\n').split(' ') for line in open(list_file)]
            # for file_name in fp_items:
            #     if os.path.isfile(os.path.join(root_folder, file_name)):
            #         # tup = (file_name)
            #         items.append(file_name)

        self.items = items
        print('\nStage: ' + stage)
        print('The number of samples: {}'.format(len(items)))

    def __getitem__(self, index):
        image = self.items[index]
        img = self.loader1(os.path.join(self.root_folder, image))
        # img2 = self.loader2(os.path.join(self.root_folder, image))
        if self.transform is not None:
            img = self.transform(img)
            # img2 = self.transform(img2)

        return img, image

    def __len__(self):
        return len(self.items)
