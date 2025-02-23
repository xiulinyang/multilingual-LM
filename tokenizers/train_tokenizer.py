from transformers import AutoTokenizer
from glob import glob
import json
import argparse

EXP_LANGS =['EN','DE','RO', 'TR','RU']
def collect_sents(lang,split):
    data_path = f'data/multilingual/{lang}/{split}/{lang}.json'
    all_text = []

    lang_parse = json.load(open(data_path))
    for sent in lang_parse:
        for s in sent['sent_annotations']:
            all_text.append(s['sent_text'])
    vocab = [x for y in all_text for x in y.split()]
    vocab = set(vocab)
    return all_text, vocab

if __name__ =='__main__':


    parser = argparse.ArgumentParser(
        prog='tokenization',
        description='train tokenizers')
    parser.add_argument('lang',
                        const='all',
                        nargs='?',
                        choices=EXP_LANGS,
                        help='language for the experiment')
    
    parser.add_argument('split',
                         const='all',
                        nargs='?',
                        choices=['train', 'dev', 'test'],
                        help='language for the experiment')
    
    parser.add_argument('size', 
                        type=int,
                        help='the size of vocab')
    # Get args
    args = parser.parse_args()
    language = args.lang
    split = args.split
    text, vocab = collect_sents(language, split)
    size = args.size
    #vocab_size = int(size*len(vocab))
    vocab_size = size
    old_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    new_tokenizer = old_tokenizer.train_new_from_iterator(text, vocab_size)
    new_tokenizer.save_pretrained(f'tokenizers/{language}')
    vocab_size = str(len(vocab))
    token_vocab_size = str(len(new_tokenizer.get_vocab()))
    print(f'The number of unique tokens in {language}: {vocab_size}')
    print(f'The number of vocabulary in the tokenizer: {token_vocab_size}')
    print(f'token id of end of text: ')
    print(new_tokenizer.encode('<|endoftext|>'))
