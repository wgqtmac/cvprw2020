# import pandas
import os

file_path = '4@all_test_res.txt'
result_path = 'result@all.txt'
root_folder = '/home/gqwang/Spoof_Croped/CASIA_CeFA'
f_dev = open(os.path.join(root_folder, file_path))
dev_items = [line.rstrip('\n').split(' ') for line in open(os.path.join(root_folder, file_path)).readlines()]
# dev_lines = f_dev.readlines()
f_res = open(os.path.join(root_folder, result_path))

res_items = [line.rstrip('\n').split(' ') for line in open(os.path.join(root_folder, result_path))]

# dev_item_total = 0.
# dev_item_cnt = 0

fp = open("./result_total_0227.txt", 'a+')


for i in range(len(dev_items)):
    print(dev_items[i][0])
    dev_item_total = 0.
    dev_item_cnt = 0
    for j in range(len(res_items)):
        if res_items[j][0].find(dev_items[i][0]) == 0:
            dev_item_cnt += 1
            dev_item_total += float(res_items[j][1])
    dev_item_avg = float(dev_item_total / dev_item_cnt)
    newline = dev_items[i][0] + ' ' + "{:.8f}".format(dev_item_avg) + "\n"
    fp.write(newline)
    print(newline)

    # print('\n')

fp.close()
