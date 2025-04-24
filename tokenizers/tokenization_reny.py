from torch.hub import ENV_GITHUB_TOKEN
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import math
import json
from tqdm import tqdm
import tokenization_scorer
from collections import Counter
TOKENIZER_DICT = {
    "EN": "gpt2",
    "DE": "malteos/gpt2-xl-wechsel-german",
    "RU": "sberbank-ai/rugpt3large_based_on_gpt2",
    "TR": "ytu-ce-cosmos/turkish-gpt2",
    "AR": "aubmindlab/aragpt2-base",

    "EN05k": "/scratch/xiulyang/multilingual-LM/tokenizers/EN/5000",
    "DE05k": "/scratch/xiulyang/multilingual-LM/tokenizers/DE/5000",
    "RU05k": "/scratch/xiulyang/multilingual-LM/tokenizers/RU/5000",
    "TR05k": "/scratch/xiulyang/multilingual-LM/tokenizers/TR/5000",
    "AR05k": "/scratch/xiulyang/multilingual-LM/tokenizers/AR/5000",

    "EN1k": "/scratch/xiulyang/multilingual-LM/tokenizers/EN/10000",
    "DE1k": "/scratch/xiulyang/multilingual-LM/tokenizers/DE/10000",
    "RU1k": "/scratch/xiulyang/multilingual-LM/tokenizers/RU/10000",
    "TR1k": "/scratch/xiulyang/multilingual-LM/tokenizers/TR/10000",
    "AR1k": "/scratch/xiulyang/multilingual-LM/tokenizers/AR/10000",

    "EN2k": "/scratch/xiulyang/multilingual-LM/tokenizers/EN/20000",
    "DE2k": "/scratch/xiulyang/multilingual-LM/tokenizers/DE/20000",
    "RU2k": "/scratch/xiulyang/multilingual-LM/tokenizers/RU/20000",
    "TR2k": "/scratch/xiulyang/multilingual-LM/tokenizers/TR/20000",
    "AR2k": "/scratch/xiulyang/multilingual-LM/tokenizers/AR/20000",

    "EN3k": "/scratch/xiulyang/multilingual-LM/tokenizers/EN/30000",
    "DE3k": "/scratch/xiulyang/multilingual-LM/tokenizers/DE/30000",
    "RU3k": "/scratch/xiulyang/multilingual-LM/tokenizers/RU/30000",
    "TR3k": "/scratch/xiulyang/multilingual-LM/tokenizers/TR/30000",
    "AR3k": "/scratch/xiulyang/multilingual-LM/tokenizers/AR/30000",

    "EN5k": "/scratch/xiulyang/multilingual-LM/tokenizers/EN/50000",
    "DE5k": "/scratch/xiulyang/multilingual-LM/tokenizers/DE/50000",
    "RU5k": "/scratch/xiulyang/multilingual-LM/tokenizers/RU/50000",
    "TR5k": "/scratch/xiulyang/multilingual-LM/tokenizers/TR/50000",
    "AR5k": "/scratch/xiulyang/multilingual-LM/tokenizers/AR/50000",
    }


for language in tqdm(TOKENIZER_DICT):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DICT[language])
    if language.startswith("EN"):
        lang='EN'
    elif language.startswith("DE"):
        lang='DE'
    elif language.startswith("RU"):
        lang='RU'
    elif language.startswith("TR"):
        lang='TR'
    elif language.startswith("AR"):
        lang='AR'
    else:
        raise ValueError ('This language is not supported yet!')
    sents = Path(f'/scratch/xiulyang/multilingual-LM/data/multilingual/{lang}/test/{lang}.test').read_text().strip().split('\n')
    token_num =0
    tokenized_text =[]
    for sent in tqdm(sents):
        tokens = tokenizer.tokenize(sent)
        tokenized_text.extend(tokens)
    with open(f'freq/{language}.json', 'w', encoding='utf-8') as f:
        freq = Counter(tokenized_text)
        json.dump(freq, f)
