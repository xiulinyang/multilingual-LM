from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import math
from tqdm import tqdm
TOKENIZER_DICT = {
    "EN": "gpt2",
   "DE": "malteos/gpt2-xl-wechsel-german",
   "RU": "sberbank-ai/rugpt3large_based_on_gpt2",
    "DE5k": "/scratch/xiulyang/multilingual-LM/tokenizers/DE/50000",
   "TR": "ytu-ce-cosmos/turkish-gpt2",

   "AR05k": "/scratch/xiulyang/multilingual-LM/tokenizers/AR/5000",
    "AR5k": "/scratch/xiulyang/multilingual-LM/tokenizers/AR/50000",
    "AR3k": "/scratch/xiulyang/multilingual-LM/tokenizers/AR/30000",
    "AR1k": "/scratch/xiulyang/multilingual-LM/tokenizers/AR/10000",
    "AR2k": "/scratch/xiulyang/multilingual-LM/tokenizers/AR/20000",

    "RU2k": "/scratch/xiulyang/multilingual-LM/tokenizers/RU/20000",
     "RU3k": "/scratch/xiulyang/multilingual-LM/tokenizers/RU/30000",
      "RU5k": "/scratch/xiulyang/multilingual-LM/tokenizers/RU/50000",
   "RU1k": "/scratch/xiulyang/multilingual-LM/tokenizers/RU/10000",
   "EN1k": "/scratch/xiulyang/multilingual-LM/tokenizers/EN/10000",
   "TR5k": "/scratch/xiulyang/multilingual-LM/tokenizers/TR/50000",
    "EN05k": "/scratch/xiulyang/multilingual-LM/tokenizers/EN/5000", 
   "TR05k": "/scratch/xiulyang/multilingual-LM/tokenizers/TR/5000",
     "RU05k": "/scratch/xiulyang/multilingual-LM/tokenizers/RU/5000",
   "TR1k": "/scratch/xiulyang/multilingual-LM/tokenizers/TR/10000",
   "DE05k": "/scratch/xiulyang/multilingual-LM/tokenizers/DE/5000",
   "DE3k": "/scratch/xiulyang/multilingual-LM/tokenizers/DE/30000",
    "DE1k": "/scratch/xiulyang/multilingual-LM/tokenizers/DE/10000",
     "DE2k": "/scratch/xiulyang/multilingual-LM/tokenizers/DE/20000",
   "TR3k": "/scratch/xiulyang/multilingual-LM/tokenizers/TR/30000",
     "EN5k": "/scratch/xiulyang/multilingual-LM/tokenizers/EN/50000",
   "EN3k": "/scratch/xiulyang/multilingual-LM/tokenizers/EN/30000",
   "EN2k": "/scratch/xiulyang/multilingual-LM/tokenizers/EN/20000",
   "TR2k": "/scratch/xiulyang/multilingual-LM/tokenizers/TR/20000",
    "AR": "aubmindlab/aragpt2-base"
    }

for lang in tqdm(TOKENIZER_DICT):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DICT[lang])
    sents = Path(f'/scratch/xiulyang/multilingual-LM/data/multilingual/{lang}/train/{lang}.train').read_text().strip().split('\n')
    token_num =0
    for sent in tqdm(sents):
        tokens = tokenizer.tokenize(sent)
        token_num += len(tokens)
    average = token_num/len(sents)
    #average = math.log(token_num)
    print(lang, average)
