from transformers import AutoTokenizer, AutoModel
from pathlib import Path

TOKENIZER_DICT = {
   "EN": "gpt2",
   "DE": "malteos/gpt2-xl-wechsel-german",
   "RU": "sberbank-ai/rugpt3large_based_on_gpt2",
   "RO": "dumitrescustefan/gpt-neo-romanian-780m",
   "TR": "ytu-ce-cosmos/turkish-gpt2",
   "FR": "lightonai/pagnol-xl",
   "NL": "yhavinga/gpt-neo-125M-dutch",
   "IT": "iGeniusAI/Italia-9B-Instruct-v0.1",
   "PL":"flax-community/papuGaPT2",
    "PT": "NOVA-vision-language/GlorIA-1.3B'",
    "ZH": "hfl/chinese-bert-wwm",
    "AR": "aubmindlab/aragpt2-base"}

for lang in TOKENIZER_DICT:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DICT[lang])
    sents = Path(f'data/multilingual/{lang}/train/{lang}.train','r').read_text().strip().split('\n')
    token_num =0
    for sent in sents:
        tokens = tokenizer.tokenize(sent)
        token_num += len(tokens)

    average = token_num/len(sents)
    print(lang, average)