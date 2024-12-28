from glob import glob
from os.path import exists
from pathlib import Path
import random
import os


random.seed(41)
#
# test = Path('/Users/xiulinyang/Desktop/all_data/comparative_merged_cleaned/de_en.txt').read_text().strip().split('\n')
# print(len([x for y in test for x in y.split()]))
# print(len(test))
# new_test = [x for x in test if len(x.split())>2]
# print(len(new_test))
# test_10k = random.sample(new_test, 10000)
# rest_test = [x for x in new_test if x not in test_10k]
# dev_10k = random.sample(rest_test, 5000)
#
# path_save_dev = Path('multilingual/multilingual/EN/dev')
# path_save_test = Path('multilingual/multilingual/EN/test')
# # Define the directory path
# path_save_test.mkdir(parents=True, exist_ok=True)
# path_save_dev.mkdir(parents=True, exist_ok=True)
# # Create the directory
# with open(f'{path_save_dev}/EN.dev', 'w') as dev, open(f'{path_save_test}/EN.test', 'w') as test:
#     test_to_write = '\n'.join(test_10k)
#     dev_to_write = '\n'.join(dev_10k)
#
#     dev.write(dev_to_write)
#     test.write(test_to_write)


split = 'test'
lang = 'zh'

if split in ['test', 'dev']:
    data_path = 'comparative_merged_cleaned'
# else:
#     data_path = 'parallel_merged'

train = Path(f'../../TODO/multilingual-LM/data/multilingual/multilingual/EN/{split}/EN.{split}' ).read_text().strip().split('\n')
test = Path(f'/Users/xiulinyang/Desktop/all_data/{data_path}/de_en.txt').read_text().strip().split('\n')

train = [x.strip() for x in train]
test = [x.strip() for x in test]
# print([x for x in train if x not in test])
overlap = set(train) & set(test)

print(len(overlap))

de_train = f'/Users/xiulinyang/Desktop/all_data/{data_path}/{lang.lower()}.txt'
en_train =  f'/Users/xiulinyang/Desktop/all_data/{data_path}/de_en.txt'



de_train = Path(de_train).read_text().strip().split('\n')
en_train = Path(en_train).read_text().strip().split('\n')
en_train = [line.strip() for line in en_train]
de_train = [line.strip() for line in de_train]
train = [line.strip() for line in train]
assert len(de_train)==len(en_train)

en_train_dic = {x:y for x, y in zip(en_train, de_train)}
print('dic len:', len(en_train_dic.keys()))
path_save_train = Path(f'../../TODO/multilingual-LM/data/multilingual/multilingual/{lang}/{split}' )
path_save_train.mkdir(parents=True, exist_ok=True)

to_write = []
for sent in train:
    to_write.append(en_train_dic[sent])
print('test_split', len(to_write))
to_write_train = '\n'.join(to_write)
with open(f'{path_save_train}/{lang}.{split}', 'w') as train:
    train.write(to_write_train)


