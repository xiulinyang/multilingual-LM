import sys
sys.path.append("..")

from transformers import GPT2LMHeadModel
from utils import FUNCTION_MAP, TOKENIZER, EXP_LANGS
from tqdm import tqdm
from glob import glob
import argparse
import pandas as pd
import math


MAX_TRAINING_STEPS = 1200
CHECKPOINTS = list(range(0, MAX_TRAINING_STEPS+1, 100))


def get_cross_entropy_from_ppl(ppl, tokenizer):
    sent_xentropy={}
    sent_text = ppl['Sentence']
    sent_xentropy['Sentences'] = sent_text
    for i in CHECKPOINTS:
        checkpoint = str(i)
        ppl_value = math.log(ppl[f'Perplexities (ckpt {checkpoint})'])
        num_tokens = len(tokenizer.encode(sent_text))-1
        sent_xentropy[f'Perplexities (ckpt {checkpoint})']= ppl_value*num_tokens
    return sent_xentropy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Edge probing',
        description='Edge probing experiments')
    parser.add_argument('perturbation_type',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=FUNCTION_MAP.keys(),
                        help='Perturbation function used to transform the multilingual dataset')
    parser.add_argument('test_perturbation_type',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=FUNCTION_MAP.keys(),
                        help='Perturbation function used to transform test the multilingual dataset')
    parser.add_argument('train_set',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=EXP_LANGS,
                        help='BabyLM train set')
    parser.add_argument('random_seed', type=int, help="Random seed")
    parser.add_argument('paren_model',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=["randinit"],
                        help='Parenthesis model')
    parser.add_argument('vs', help='Vocabulary size')

    args = parser.parse_args()
    vs = args.vs
    la = args.train_set
    lang_lower_case = args.train_set.lower()
    gpt2_tokenizer = TOKENIZER[la]['shuffle']

    language_ppl = pd.read_csv(f'perplexity_results/{args.perturbation_type}_{args.train_set}/{args.paren_model}_seed{args.random_seed}_test_{args.test_perturbation_type}_{lang_lower_case}_{vs}.csv').to_dict(orient='records')

    corpus_xentropy = []
    for sent in tqdm(language_ppl):
        sent_xentropy = get_cross_entropy_from_ppl(sent, gpt2_tokenizer)
        corpus_xentropy.append(sent_xentropy)

    corpus_xentropy = pd.DataFrame(corpus_xentropy)
    corpus_xentropy.to_csv(
        f'cross_entropy_results/{args.perturbation_type}_{args.train_set}_{args.test_perturbation_type}/{args.paren_model}_seed{args.random_seed}_test_{lang_lower_case}_{vs}.csv',
        mode='w', index=False)


