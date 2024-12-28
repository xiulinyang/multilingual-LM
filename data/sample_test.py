from glob import glob
from os.path import exists
from pathlib import Path
import random
import os


random.seed(41)

# test = Path('/Users/xiulinyang/Desktop/all_data/comparative_merged/de_en.txt').read_text().strip().split('\n')
# new_test = [x for x in test if len(x.split())>2]
# test_10k = random.sample(new_test, 10000)
# rest_test = [x for x in new_test if x not in test_10k]
# dev_10k = random.sample(rest_test, 5000)
#
# path_save_dev = Path('EN/dev')
# path_save_test = Path('EN/test')
# # Define the directory path
# path_save_test.mkdir(parents=True, exist_ok=True)
# path_save_dev.mkdir(parents=True, exist_ok=True)
# # Create the directory
# with open(f'{path_save_dev}/en.dev', 'w') as dev, open(f'{path_save_test}/en.test', 'w') as test:
#     test_to_write = '\n'.join(test_10k)
#     dev_to_write = '\n'.join(dev_10k)
#
#     dev.write(dev_to_write)
#     test.write(test_to_write)


train = Path('/Users/xiulinyang/Desktop/all_data/parallel_merged/de_en.txt').read_text().strip().split('\n')
new_train = [x for x in train if len(x.split())>2]

path_save_train = Path('EN/train')
path_save_train.mkdir(parents=True, exist_ok=True)

to_write_train = '\n'.join(new_train)
with open(f'{path_save_train}/en.train', 'w') as train:
    train.write(to_write_train)


