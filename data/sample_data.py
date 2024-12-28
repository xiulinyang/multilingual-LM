from pathlib import Path
import random
import argparse
from glob import glob

def sample_sents(sents, all_sents):
    percent = len(sents)/len(all_sents)*10000
    return random.sample(sents, int(percent))










if __name__ =='__main__':
    parser = argparse.ArgumentParser(
                        prog='sample dataset',
                        description='randomly sample sentences')
    parser.add_argument('-p', '--path', type = str,help="Path to file(s)")
    args = parser.parse_args()


    file_paths = args.path
    print(file_paths)
    sample_size = 0
    all_paths = glob(f'{file_paths}/*')
    text = [x for y in all_paths for x in Path(y).read_text().strip().split('\n')]
    print(len(text))
    toks = [len(x.split()) for x in text]
    print(sum(toks))
    for i, file_path in enumerate(glob(f'{file_paths}/*')):
        print(file_path)
        sents = Path(file_path).read_text().strip().split('\n')
        sampled_sents = sample_sents(sents, text)
        sample_size += len(sampled_sents)
        if i==len(all_paths)-1:
            if sample_size >10000:
                diff = sample_size-100000
                sampled_sents = sampled_sents[:-diff]
            else:
                diff = 10000-sample_size
                to_add = random.sample(sents, diff)
                sampled_sents.extend(to_add)
        print(len(sampled_sents))
        with open(f'{file_path}_part.test', 'w') as dev:
            to_write = '\n'.join(sampled_sents)
            dev.write(to_write)

