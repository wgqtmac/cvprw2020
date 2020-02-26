#!/usr/bin/python
# -*- coding: utf-8 -*-
import os



def demo():
    txtfile_to_write = open('/home/gqwang/Spoof_Croped/CASIA_CeFA/4@3_dev_list.txt', 'a')
    list_file = '/home/gqwang/Spoof_Croped/CASIA_CeFA/4@3_dev_ref.txt'
    root_folder = '/home/gqwang/Spoof_Croped/CASIA_CeFA'
    items = []

    fp_items = [line.rstrip('\n').split(' ') for line in open(list_file)]
    for file_name, label in fp_items:
        for filename in os.listdir(os.path.join(root_folder, file_name, 'profile')):
            file_related_name = os.path.join(file_name, 'profile', filename)
            print(file_related_name)
            label = int(label)
            if label == 1:
                txtfile_to_write.write('{:s} {:d}\n'.format(file_related_name, 1))
            else:
                txtfile_to_write.write('{:s} {:d}\n'.format(file_related_name, 0))

        # if os.listdir(os.path.join(root_folder, file_name, 'profile')):
        #     tup = (file_name, int(label))
        #     items.append(tup)
        # label = int(label)
        # if label == 1:
        #     txtfile_to_write.write('{:s} {:d}\n'.format(file_name, 1))
        # else:
        #     txtfile_to_write.write('{:s} {:d}\n'.format(file_name, 0))

if __name__ == "__main__":
    demo()

