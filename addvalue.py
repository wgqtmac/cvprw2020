import pandas as pd
import os

train_data = pd.read_csv('./datasets/CUB_200_2011/anno/test.txt', sep=' ')
limit_len=len(train_data)
i = 0
while i<limit_len:
    item = train_data.loc[i]
    image = item[0]
    lable = item[1] - 1

    i+=1
    print('{0} {1}'.format(image, lable))


