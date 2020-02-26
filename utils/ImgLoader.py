import os
import torch
import torch.utils.data as data
from PIL import Image
import random
import torchvision.transforms as transforms

def default_loader(path):
    return Image.open(path).convert('RGB')


def random_sample(img_names, labels):
    anno_dict = {}
    img_list = []
    anno_list = []
    for img, anno in zip(img_names, labels):
        if not anno in anno_dict:
            anno_dict[anno] = [img]
        else:
            anno_dict[anno].append(img)

    for anno in anno_dict.keys():
        anno_len = len(anno_dict[anno])
        fetch_keys = random.sample(list(range(anno_len)), anno_len//10)
        img_list.extend([anno_dict[anno][x] for x in fetch_keys])
        anno_list.extend([anno for x in fetch_keys])
    return img_list, anno_list

class ImgLoader(data.Dataset):
    def __init__(self, root_folder, list_file, transform=None, loader=default_loader, stage='Train'):
        self.root_folder = root_folder
        self.loader = loader
        self.transform = transform

        items = []

        fp_items = [line.rstrip('\n').split(' ') for line in open(list_file)]
        for file_name, label in fp_items:
            if os.path.isfile(os.path.join(root_folder, file_name)):
                tup = (file_name, int(label))
                items.append(tup)
        self.items = items
        print('\nStage: ' + stage)
        print('The number of samples: {}'.format(len(items)))

    def __getitem__(self, index):
        image, label = self.items[index]
        img = self.loader(os.path.join(self.root_folder, image))
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label

    def __len__(self):
        return len(self.items)


if __name__ == '__main__':
    # src_dataset = ImgLoader(params.root_folder, os.path.join(params.root_folder, params.src_train_list),
    #                         transforms.Compose([
    #                             transforms.Resize(256),
    #                             transforms.RandomCrop(248),
    #                             transforms.RandomHorizontalFlip(),
    #                             transforms.ToTensor()
    #                         ]))
    src_dataset = ImgLoader('', '',
                            transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomCrop(248),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()
                            ]))
