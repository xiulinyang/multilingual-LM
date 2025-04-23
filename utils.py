# utils.py
# Author: Julie Kallini

from collections import deque
from logging import raiseExceptions
from string import punctuation
from warnings import WarningMessage
from nltk import Tree
from nltk.corpus.reader import ADJ_SAT
from transformers import AutoTokenizer, AddedToken,GPT2LMHeadModel, GPT2Tokenizer,AutoTokenizer, AutoModel
from functools import partial
from numpy.random import default_rng
from nltk.tree import ParentedTree
import torch


##############################################################################
# CONSTANTS
##############################################################################
ROOT_PATH = '/scratch/xiulyang'
EXP_LANGS = ['AR05k', 'AR1k', 'AR2k', 'AR3k', 'AR5k','EN1k', 'RU1k','RU2k','RU3k','RU05k','EN05k', 'DE05k', 'TR05k', 'RU5k','DE3k','DE5k','DE1k', 'DE2k', 'TR1k', 'TR2k', 'EN2k','TR3k','EN3k', 'EN5k', 'TR15k', 'TR20k', 'TR10k', 'TR5k', 'EN', 'DE', 'DENF', 'AR', 'ZH', 'RU', 'TR', 'RO','ES', 'FR', 'PL', 'PT', 'NL', 'IT', 'FR', 'ITRN', 'ENRN','PTRN','AANN','NNDA','ZHRN']
MULTILINGUAL_SPLITS = ["train", 'dev', 'test', 'unittest']
SEEDS = [21, 53, 84]
CHECKPOINTS = list(range(50, 501, 50))
GENRES = {
    "aochildes": "CHILDES",
    "bnc_spoken": "British National Corpus (BNC)",
    "cbt": "Children’s Book Test",
    "children_stories": "Children’s Stories Text Corpus",
    "gutenberg": "Standardized Project Gutenberg Corpus",
    "open_subtitles": "OpenSubtitles",
    "qed": "QCRI Educational Domain Corpus",
    "simple_wikipedia": "Simple Wikipedia",
    "switchboard": "Switchboard Dialog Act Corpus",
    "wikipedia": "Wikipedia"
}
CHECKPOINT_WRITE_PATH = f"/{ROOT_PATH}/multilingual_models"
CHECKPOINT_READ_PATH = f"/{ROOT_PATH}/multilingual_models"
TOKENIZER_PATH = f"/{ROOT_PATH}/multilingual-LM/tokenizers"
MULTILINGUAL_DATA_PATH = f"/{ROOT_PATH}/multilingual-LM/data/multilingual/"
MARKER_HOP_SING = "🅂"
MARKER_HOP_PLUR = "🄿"
MARKER_REV = "🅁"
BOS_TOKEN = "<BOS_TOKEN>"
PART_TOKENS = set(["n't", "'ll", "'s", "'re", "'ve", "'m"])
PUNCT_TOKENS = set(punctuation)



PAREN_MODEL_PATH = "/u/scr/isabelvp//tilt-stuff/tilt-finetuning/pretrained_checkpoints/"
PAREN_MODELS = {
    "CROSS": "flat-parens_vocab500-uniform_deplength-nesting-nolimit",
    "NEST": "nested-parens0.49_vocab500-uniform",
    "RAND": "random_vocab500-uniform",
}

##############################################################################
# HELPER FUNCTIONS
##############################################################################


def write_file(directory, filename, lines):
    f = open(directory + filename, "w")
    f.writelines(lines)
    f.close()


def get_gpt2_tokenizer_with_markers(marker_list, lang):
    if lang in EXP_LANGS:    
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DICT[lang],use_fast=False)
    else:
        WarningMessage("You didn't specify a language yet, "
                       "so we use the English gpt2 tokenizer by default!")
        tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # If no new markers to add, return normal tokenizer
    if len(marker_list) == 0:
        return tokenizer

    # Create tokens and return modified tokenizer
    new_tokens = []
    for marker in marker_list:
        new_tokens.append(AddedToken(marker, lstrip=True, rstrip=False))
    tokenizer.add_tokens(new_tokens)
    return tokenizer



# MARKER_TOKEN_IDS = [marker_sg_token, marker_pl_token, marker_rev_token]
MARKER_TOKEN_IDS =[]
def compute_surprisals(model, input_ids):
    # Get the log probabilities from the model
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, :-1]
        shifted_input_ids = input_ids[:, 1:]

        # Get the log probabilities for the actual next tokens
        log_probs = torch.log2(torch.nn.functional.softmax(logits, dim=-1))
        true_log_probs = log_probs.gather(
            2, shifted_input_ids.unsqueeze(-1)).squeeze(-1)

    # Get the negative log probabilities
    neg_log_probs = (-true_log_probs).tolist()
    surprisals = [[None] + probs for probs in neg_log_probs]
    return surprisals


def compute_token_probabilities(model, input_ids, token_id, pad_token_id):
    # Get the log probabilities from the model
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, :-1]
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Get the probabilities for the specified token at each position
        token_probs = probs[:, :, token_id]

    # Convert to list and add None at the beginning to align with input tokens
    # Put null probability for instances of pad token
    token_probs_list = []
    for batch_i, probs in enumerate(token_probs):
        input_ids_seq = input_ids[batch_i].tolist() + [pad_token_id]
        filtered = [p if input_ids_seq[pos_i+1] !=
                    pad_token_id else None for pos_i, p in enumerate(probs.tolist())]
        token_probs_list.append([None] + filtered)

    return token_probs_list


def merge_part_tokens(words):
    result = []
    for s in words:
        if result and s in PART_TOKENS and len(result) > 0:
            result[-1] += s
        else:
            result.append(s)
    return result


def __affect_hop_word(word):
    return word["feats"] and "Person=3" in word["feats"] \
        and "Tense=Pres" in word["feats"] \
        and "VerbForm=Fin" in word["feats"] \
        and "Number" in word["feats"]


def __perturb_hop_words(sent, num_hops, marker_sg, marker_pl,lang):
    perturbed_tokens, _ = __perturb_hop_words_complete_hops(
        sent, num_hops, marker_sg, marker_pl,lang)
    return perturbed_tokens

def reorder_np(lang, np_subtree, sequence):
    if lang=='EN':
        NPS = ['NN', 'NNS', 'NNP', 'NNPS']
        NUMP = ['QP', '$', 'CD']
        DP =['DT', 'PRP$', 'PDT','POS']
        ADJP = ['RB', 'ADJP', 'JJR', 'JJS', 'JJ']
    elif lang =='ZH':
        NPS = ['NN','NP', 'NR', 'NT', 'PRP', 'PN', 'FW']
        NUMP = ['CD','OD', 'QP']
        ADJP = ['JJ','ADJP', 'DNP', 'DEC','DEG']
        DP = ['DT','M', 'CLP', 'DP']
    elif lang =='IT':
        NPS =['NOUN', 'PRON', 'PROPN', 'SYM', 'X']
        NUMP = ['NUM', 'sq']
        DP = ['DET']
        ADJP = ['ADJ', 'sa']
    elif lang =='PT':
        NPS = ["N'", "NOUN", "PRON"]
        NUMP = ["QNT", "QNT'", "NUM", "PERCENTP", "PERCENTP'", "CARD", "CARD'"]
        ADJP = ["ADJ", "AP"]
        DP = ["DET", "D", "DEM", "POSS", "POSS'"]
    else:
        raise ValueError('The language is not supported!')

    desired = []
    others = []
    if sequence =='dnna':
        desired_order = DP+NUMP+NPS+ADJP
    elif sequence == 'dann':
        desired_order = DP+ADJP+NPS+NUMP
    elif sequence =='nnda':
        desired_order = NPS+NUMP+DP+ADJP
    elif sequence =='annd':
        desired_order = ADJP+NUMP+NPS+DP
    elif sequence =='andn':
        desired_order = ADJP+NUMP+DP+NPS
    elif sequence =='nand':
        desired_order = NPS+ADJP+NUMP+DP
    elif sequence =='dnan':
        desired_order = DP+NUMP+ADJP+NPS
    else:
        raise ValueError('The order is not available yet')
    for child in np_subtree:
        if hasattr(child, "label") and child.label() in desired_order:
            desired.append(child)
        else:
            others.append(child)

    sorted_desired = sorted(
        desired,
        key=lambda child: desired_order.index(child.label())
    )
    reordered_children = []
    desired_iter = iter(sorted_desired)
    for child in np_subtree:
        if child in desired:
            reordered_children.append(next(desired_iter))
        else:
            reordered_children.append(child)

    np_subtree[:] = reordered_children

def navigate_and_reorder_tree(lang, t, seq):
    if lang in ['EN', 'ZH', 'PT']:
        n = 'NP'
    elif lang =='IT':
        n = 'sn'
    else:
        raise ValueError('The language is not supported!')
    for subtree in t:
        try:
            if subtree.label() == n:
                reorder_np(lang, subtree, seq)
            navigate_and_reorder_tree(lang, subtree, seq)
        except AttributeError:
            continue
    return t


def check_word_hops_completed(sent, lang, num_hops=4, marker=MARKER_HOP_SING):
    _, hops_completed = __perturb_hop_words_complete_hops(
        sent, num_hops, marker, marker,lang)
    return hops_completed


def __perturb_hop_words_complete_hops(sent, num_hops, marker_sg, marker_pl, lang):

    word_annotations = sent["word_annotations"].copy()
    word_annotations.reverse()
    tokenizer = TOKENIZER[lang]['hop']
    hop_completed = []
    new_sent = []
    for word in word_annotations:

        # Identify 3.pres verbs
        if __affect_hop_word(word):

            # Lemmatize verb if possible
            new_sent.append(
                word["lemma"] if word["lemma"] is not None else word["text"])

            # Marker hopping logic
            insert_index = len(new_sent)-1
            skipped_words = 0
            while skipped_words < num_hops and insert_index > 0:

                # Handle edge case when punctuation (or sequence of
                # punctuation) begin the sentence
                if (not any([c.isalnum() for c in
                             "".join(new_sent[:insert_index])])):
                    break

                # Count word as skipped if it is not a special token
                if (new_sent[insert_index] not in PART_TOKENS) and \
                        (not set(new_sent[insert_index]).issubset(PUNCT_TOKENS)):
                    skipped_words += 1
                insert_index -= 1

            # Handle edge case when insert index is punctuation (and this is not
            # sentence-initial punctuation)
            if any([c.isalnum() for c in
                    "".join(new_sent[:insert_index])]):
                while insert_index != 0 and (new_sent[insert_index] in PART_TOKENS
                                             or set(new_sent[insert_index]).issubset(PUNCT_TOKENS)):
                    insert_index -= 1

            # Handle edge case when token before insert index is part/aux token
            if insert_index != 0 and new_sent[insert_index-1] in PART_TOKENS:
                insert_index -= 1

            # Log if this sentence had all full hops
            hop_completed.append(skipped_words == num_hops)

            # Use correct marker for singular vs. plural
            if "Number=Sing" in word["feats"]:
                new_sent.insert(insert_index, marker_sg)
            elif "Number=Plur" in word["feats"]:
                new_sent.insert(insert_index, marker_pl)
            else:
                raise Exception(
                    "Number not in verb features\n" + sent["sent_text"])

        else:
            new_sent.append(word["text"])

    new_sent.reverse()
    sent_string = " ".join(merge_part_tokens(new_sent))

    tokens = tokenizer.encode(sent_string)
    return tokens, all(hop_completed) and len(hop_completed) > 0


def __perturb_hop_tokens(sent, num_hops, lang):
    word_annotations = sent["word_annotations"].copy()
    word_annotations.reverse()
    tokenizer = TOKENIZER[lang]['hop']
    new_sent = deque()
    tokens = []
    for word in word_annotations:

        # Identify 3.pres verbs
        if __affect_hop_word(word):

            # Lemmatize verb if possible
            lemma = word["lemma"] if word["lemma"] is not None else word["text"]

            if len(new_sent) > 0 and new_sent[0] in PART_TOKENS:
                lemma = lemma + new_sent[0]
                new_sent.popleft()

            if len(new_sent) > 0:
                sent_string = " ".join(merge_part_tokens(new_sent))
                tokens = tokenizer.encode(
                    " " + sent_string) + tokens

            # Use correct marker for singular vs. plural
            if "Number=Sing" in word["feats"]:
                tokens.insert(num_hops, marker_sg_token)
            elif "Number=Plur" in word["feats"]:
                tokens.insert(num_hops, marker_pl_token)
            else:
                raise Exception(
                    "Number not in verb features\n" + sent["sent_text"])

            new_sent = deque()
            new_sent.append(lemma)

        else:
            new_sent.appendleft(word["text"])

    if len(new_sent) > 0:
        sent_string = " ".join(merge_part_tokens(new_sent))
        tokens = tokenizer.encode(sent_string) + tokens
    return tokens


def __perturb_reverse(sent, rng, reverse, full, lang):
    tokenizer = TOKENIZER[lang]['reverse']
    # Get sentence text and GPT-2 tokens
    tokens = tokenizer.encode(sent["sent_text"])

    # Pick random index to insert REV token
    i = rng.choice(len(tokens)+1)
    tokens.insert(i, marker_rev_token)

    # Extract tokens before/after the marker, and reverse tokens after
    tokens_before = tokens[:i+1]
    tokens_after = tokens[i+1:]
    if reverse:
        tokens_after.reverse()
    new_tokens = tokens_before + tokens_after
    if full:
        assert not reverse
        new_tokens.reverse()

    return new_tokens

def __perturb_reverse_full(sent, lang):
    tokenizer = TOKENIZER[lang]['shuffle']
    if lang=='ZH':
        sent_text = ''.join(sent["sent_text"].split())
        tokens = tokenizer.encode(sent_text)
    else:
        tokens = tokenizer.encode(sent["sent_text"])
    return tokens.reverse

def __perturb_np_num_det_adj(sent, lang, seq):
    tokenizer = TOKENIZER[lang]['shuffle']
    tree = sent['constituency_parse']
    t = Tree.fromstring(tree)
    t = navigate_and_reorder_tree(lang, t, seq)
    if lang=='ZH':
        return tokenizer.encode(''.join(t.leaves()))
    else:
        return tokenizer.encode(' '.join(t.leaves()))

def __perturb_reverse_full_word(sent, lang):
    tokenizer = TOKENIZER[lang]['shuffle']
    sent_words = sent["sent_text"].split()
    sent_words.reverse()
    if lang=='ZH':
        sent_text = ''.join(sent_words)
        tokens = tokenizer.encode(sent_text)
    else:
        tokens = tokenizer.encode(' '.join(sent_words))
    return tokens


def __perturb_shuffle_deterministic(sent, seed, shuffle, lang):
    # Get sentence text and GPT-2 tokens
    tokenizer = TOKENIZER[lang]['shuffle']
    if lang=='ZH':
        sent_text = ''.join(sent["sent_text"].split())
        tokens = tokenizer.encode(sent_text)
    else:
        tokens = tokenizer.encode(sent["sent_text"])
    if shuffle:
        default_rng(seed).shuffle(tokens)
    return tokens

def __perturb_shuffle_deterministic_full(sent, seed, shuffle, lang):
    # Get sentence text and GPT-2 tokens
    tokenizer = TOKENIZER[lang]['shuffle']
    if lang=='ZH':
        sent_text = ''.join(sent["sent_text"].split())
        tokens = tokenizer.encode(sent_text)
    else:
        tokens = tokenizer.encode(sent["sent_text"])
    if shuffle:
        default_rng(seed).shuffle(tokens)
    return tokens


def __perturb_shuffle_deterministic_word(sent, seed, shuffle, lang):
    tokenizer = TOKENIZER[lang]['shuffle']
    tokens = sent["sent_text"].split()
    if shuffle:
        default_rng(seed).shuffle(tokens)
        tokens = tokenizer.encode(' '.join(tokens))
    return tokens



def __perturb_shuffle_nondeterministic(sent, rng, lang):
    # Get sentence text and GPT-2 tokens
    tokenizer = TOKENIZER[lang]['shuffle']
    tokens = tokenizer.encode(sent["sent_text"])
    rng.shuffle(tokens)
    return tokens


def __perturb_shuffle_local(sent, seed, lang, window=5):
    # Get sentence text and GPT-2 tokens
    tokenizer = TOKENIZER[lang]['shuffle']
    if lang =='ZH':
        sent_text = ''.join(sent['sent_text'].split())
        tokens = tokenizer.encode(sent_text)
    else:
        tokens = tokenizer.encode(sent["sent_text"])

    # Shuffle tokens in batches of size window
    shuffled_tokens = []
    for i in range(0, len(tokens), window):
        batch = tokens[i:i+window].copy()
        if window==2:
            batch.reverse()
        else:
            default_rng(seed).shuffle(batch)
        shuffled_tokens += batch

    return shuffled_tokens


def __perturb_shuffle_local_word(sent, seed, lang, window=5):
    tokenizer = TOKENIZER[lang]['shuffle']
    tokens = sent['sent_text'].split()

    shuffled_tokens = []
    for i in range(0, len(tokens), window):
        batch = tokens[i:i+window].copy()  
        default_rng(seed).shuffle(batch) 
        shuffled_tokens += batch
    shuffled_words = tokenizer.encode(' '.join(shuffled_tokens))
    return shuffled_words

def __perturb_shuffle_even_odd(sent, lang):
    # Get sentence text and GPT-2 tokens
    tokenizer = TOKENIZER[lang]['shuffle']
    tokens = tokenizer.encode(sent["sent_text"])
    even = [tok for i, tok in enumerate(tokens) if i % 2 == 0]
    odd = [tok for i, tok in enumerate(tokens) if i % 2 != 0]
    return even + odd

def longest_continuous_span(nums, target):
    if target not in nums:
        return []
    nums = sorted(nums)
    current_span = []
    all_spans = []
    for num in nums:
        if not current_span or num == current_span[-1] + 1:
            current_span.append(num)
        else:
            current_span = [num]
        all_spans.append(current_span)
    return [x for x in all_spans if target==x[-1]][-1]

def __perturb_adj_num(sent, lang, post='ADJ'):
    tokenizer = TOKENIZER[lang]['shuffle']
    sent_pos = [x['upos'] for x in sent['word_annotations']]
    sent_toks = [x['text'] for x in sent['word_annotations']]
    sent_toks_copy = [x['text'] for x in sent['word_annotations']]
    adj_pos = [x for x, y in enumerate(sent_pos) if y==post]
    for i, pos in enumerate(sent_pos):
        if pos in ['NOUN', 'X', 'SYM', 'PROPN']:
            adj_indices = longest_continuous_span(adj_pos, i - 1)
            if len(adj_indices)>1:
                print(sent_toks)
            if i - 1 in adj_indices:
                len_adj = len(adj_indices)
                sent_toks_copy[i-len_adj] = sent_toks[i]
                sent_toks_copy[i-len_adj+1:i+1] = sent_toks[adj_indices[0]: adj_indices[-1]+1]
    assert len(sent_toks_copy) == len(sent_toks)
    tw = ' '.join(sent_toks_copy)
    tokens = tokenizer.encode(tw)
    return tokens

def __perturb_remove_fw(sent,lang):
    tokenizer = TOKENIZER[lang]['shuffle']
    sent_text = ' '.join([x['text'] for x in sent['word_annotations'] if x['upos'] not in ['DET', 'AUX', 'ADP', 'SCONJ', 'PART','CCONJ']])
    tokens = tokenizer.encode(sent_text)
    return tokens

##############################################################################
# AFFECT FUNCTIONS
# These functions define when a perturbation has been applied to a sentence
# not. This is used for identifying which test sentences have been
# altered to separate affected vs. unaffected senences. Affect functions are
# functions of the input sentence object and return a boolean.
##############################################################################


def affect_hop(sent,lang):
    return any([__affect_hop_word(word) for word in sent['word_annotations']]) \
        and sent["constituency_parse"] is not None


def affect_reverse(sent,lang):
    return True


def affect_shuffle(sent,lang):
    return True


def affect_none(sent,lang):
    return False


##############################################################################
# FILTER FUNCTIONS
# These functions define when an affected sentence should be included in the
# final dataset. For instance, hop perturbations where the marker is placed
# at the end of the sentence should be excluded. A filter function returns
# True if an affected sentence should be included in the dataset.
##############################################################################


def filter_hop(sent,lang):
    # Assertion needed since filter function is only defined for affected
    # sentences
    assert (affect_hop(sent,lang))
    return check_word_hops_completed(sent, 4)


def filter_reverse(sent,lang):
    return True


def filter_shuffle(sent, lang):
    tokenizer = TOKENIZER[lang]['shuffle']
    tokens = tokenizer.encode(sent["sent_text"])
    return len(tokens) > 1 and len(tokens) <= 350


def filter_none(sent,lang):
    return False


##############################################################################
# PERTURBATION FUNCTIONS
# These functions define how a perturbation will affect a sentence. They
# take in a sentence object and an optional marker
# for verb transformations. They return a string representing the transformed
# sentence.
##############################################################################


def perturb_hop_words4(sent, lang):
    return __perturb_hop_words(sent, 4, MARKER_HOP_SING, MARKER_HOP_PLUR, lang)


def perturb_hop_tokens4(sent,lang):
    return __perturb_hop_tokens(sent, 4,lang)


def perturb_hop_control(sent,lang):
    return __perturb_hop_tokens(sent, 0,lang)


def perturb_reverse(sent, rng, lang, reverse=True, full=False):
    return __perturb_reverse(sent, rng, reverse, full,lang)


def perturb_shuffle_deterministic(sent, lang, seed=None, shuffle=True):
    return __perturb_shuffle_deterministic(sent, seed, shuffle,lang)

def perturb_shuffle_deterministic_word(sent, lang, seed=None, shuffle=True):
    return __perturb_shuffle_deterministic_word(sent, seed, shuffle,lang)

def perturb_shuffle_nondeterministic(sent, rng, lang):
    return __perturb_shuffle_nondeterministic(sent, rng, lang)


def perturb_shuffle_local(sent, seed, window, lang):
    return __perturb_shuffle_local(sent, seed, lang, window)

def perturb_shuffle_local_word(sent, seed, window, lang):
    return __perturb_shuffle_local_word(sent, seed, lang, window)


def perturb_reverse_full(sent, lang):
    return __perturb_reverse_full(sent, lang)

def perturb_reverse_full_word(sent, lang):
    return __perturb_reverse_full_word(sent, lang)


def perturb_shuffle_even_odd(sent, lang):
    return __perturb_shuffle_even_odd(sent, lang)

def perturb_shuffle_remove_fw(sent, lang):
    return __perturb_remove_fw(sent, lang)

def perturb_adj_num(sent, lang, post):
    return __perturb_adj_num(sent, lang, post)

def perturb_np_num_det_adj(sent, lang, seq):
    return __perturb_np_num_det_adj(sent, lang, seq)


TOKENIZER_DICT = {
    'AANN':'gpt2',
    'NNDA':'gpt2',
    "ENRN": "gpt2",
   "EN": "gpt2",
   "DE": "malteos/gpt2-xl-wechsel-german",
   "RU": "sberbank-ai/rugpt3large_based_on_gpt2",
   "RO": "dumitrescustefan/gpt-neo-romanian-780m",
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
   "TR15k": "/scratch/xiulyang/multilingual-LM/tokenizers/TR/150000",
   "TR20k": "/scratch/xiulyang/multilingual-LM/tokenizers/TR/200000",
   "TR10k": "/scratch/xiulyang/multilingual-LM/tokenizers/TR/100000",
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
   "FR": "lightonai/pagnol-xl",
   "NL": "yhavinga/gpt-neo-125M-dutch",
    "IT":"flax-community/papuGaPT2",
     "ITRN":"flax-community/papuGaPT2",
   #"IT": "iGeniusAI/Italia-9B-Instruct-v0.1",
   #"ITRN": "iGeniusAI/Italia-9B-Instruct-v0.1",
   "PL":"flax-community/papuGaPT2",
   #"PT": "TucanoBR/Tucano-1b1",
   "PT": "NOVA-vision-language/GlorIA-1.3B",
    "PTRN": "TucanoBR/Tucano-1b1",
    "ZH": "google-bert/bert-base-chinese",
    "ZHRN":"google-bert/bert-base-chinese",
    "AR": "aubmindlab/aragpt2-base"}

gpt2_tokenizer_de = get_gpt2_tokenizer_with_markers([], 'DE')
gpt2_tokenizer_en = get_gpt2_tokenizer_with_markers([],'EN')
gpt2_tokenizer_tr = get_gpt2_tokenizer_with_markers([],'TR')
gpt2_tokenizer_ru = get_gpt2_tokenizer_with_markers([], 'RU')
gpt2_tokenizer_ro = get_gpt2_tokenizer_with_markers([], 'RO')
gpt2_tokenizer_fr = get_gpt2_tokenizer_with_markers([], 'FR') 
gpt2_tokenizer_nl = get_gpt2_tokenizer_with_markers([], 'NL')
gpt2_tokenizer_pl = get_gpt2_tokenizer_with_markers([], 'PL')
gpt2_tokenizer_pt = get_gpt2_tokenizer_with_markers([], 'PT')
gpt2_tokenizer_it = get_gpt2_tokenizer_with_markers([], 'IT')
gpt2_tokenizer_zh = get_gpt2_tokenizer_with_markers([], 'ZH')
gpt2_tokenizer_ar = get_gpt2_tokenizer_with_markers([], 'AR')



gpt2_tokenizer_ar5k = get_gpt2_tokenizer_with_markers([],'AR5k')
gpt2_tokenizer_ar3k = get_gpt2_tokenizer_with_markers([],'AR3k')
gpt2_tokenizer_ar2k = get_gpt2_tokenizer_with_markers([],'AR2k')
gpt2_tokenizer_ar1k = get_gpt2_tokenizer_with_markers([],'AR1k')
gpt2_tokenizer_ar05k = get_gpt2_tokenizer_with_markers([],'AR05k')

gpt2_tokenizer_ru5k = get_gpt2_tokenizer_with_markers([],'RU5k')
gpt2_tokenizer_ru3k = get_gpt2_tokenizer_with_markers([],'RU3k')
gpt2_tokenizer_ru2k = get_gpt2_tokenizer_with_markers([],'RU2k')
gpt2_tokenizer_ru1k = get_gpt2_tokenizer_with_markers([],'RU1k')
gpt2_tokenizer_tr1k = get_gpt2_tokenizer_with_markers([],'TR1k')
gpt2_tokenizer_tr15k = get_gpt2_tokenizer_with_markers([],'TR15k')
gpt2_tokenizer_tr20k = get_gpt2_tokenizer_with_markers([],'TR20k')
gpt2_tokenizer_tr5k = get_gpt2_tokenizer_with_markers([],'TR5k')
gpt2_tokenizer_tr10k = get_gpt2_tokenizer_with_markers([],'TR10k')
gpt2_tokenizer_de05k = get_gpt2_tokenizer_with_markers([],'DE05k')
gpt2_tokenizer_en1k = get_gpt2_tokenizer_with_markers([],'EN1k')
gpt2_tokenizer_tr05k = get_gpt2_tokenizer_with_markers([],'TR05k')
gpt2_tokenizer_ru05k = get_gpt2_tokenizer_with_markers([],'RU05k')
gpt2_tokenizer_en05k = get_gpt2_tokenizer_with_markers([],'EN05k')
gpt2_tokenizer_de5k = get_gpt2_tokenizer_with_markers([],'DE5k')
gpt2_tokenizer_de3k = get_gpt2_tokenizer_with_markers([],'DE3k')
gpt2_tokenizer_de2k = get_gpt2_tokenizer_with_markers([],'DE2k')
gpt2_tokenizer_de1k = get_gpt2_tokenizer_with_markers([],'DE1k')
gpt2_tokenizer_tr2k = get_gpt2_tokenizer_with_markers([],'TR2k')
gpt2_tokenizer_en2k = get_gpt2_tokenizer_with_markers([],'EN2k')
gpt2_tokenizer_tr3k = get_gpt2_tokenizer_with_markers([],'TR3k')
gpt2_tokenizer_en3k = get_gpt2_tokenizer_with_markers([],'EN3k')
gpt2_tokenizer_en5k = get_gpt2_tokenizer_with_markers([],'EN5k')
gpt2_hop_tokenizer_en = get_gpt2_tokenizer_with_markers(
           [MARKER_HOP_SING, MARKER_HOP_PLUR], 'EN')
gpt2_rev_tokenizer_en = get_gpt2_tokenizer_with_markers(
           [MARKER_REV], 'EN')
# Get ids of marker tokens
marker_rev_token = gpt2_rev_tokenizer_en.get_added_vocab()[
           MARKER_REV]
gpt2_det_tokenizer_en = get_gpt2_tokenizer_with_markers(
           [BOS_TOKEN], 'EN')
#GPT-2 reverse tokenization
#gpt2_rev_tokenizer_es = get_gpt2_tokenizer_with_markers([MARKER_REV], 'ES')  
# Get ids of marker tokens
bos_token_id = gpt2_det_tokenizer_en.get_added_vocab()[BOS_TOKEN]
marker_sg_token = gpt2_hop_tokenizer_en.get_added_vocab()[
           MARKER_HOP_SING]
marker_pl_token = gpt2_hop_tokenizer_en.get_added_vocab()[
           MARKER_HOP_PLUR]

TOKENIZER = {
'EN':{"shuffle": gpt2_tokenizer_en},
"DE":{"shuffle": gpt2_tokenizer_de},
"RU":{"shuffle": gpt2_tokenizer_ru},
"TR":{"shuffle": gpt2_tokenizer_tr},
"RO":{"shuffle": gpt2_tokenizer_ro},
"FR":{"shuffle": gpt2_tokenizer_fr},
"NL":{"shuffle": gpt2_tokenizer_nl},
"IT":{"shuffle": gpt2_tokenizer_it},
"ITRN":{"shuffle": gpt2_tokenizer_it},
"ZH":{"shuffle": gpt2_tokenizer_zh},
"PL":{"shuffle": gpt2_tokenizer_pl},
"PT":{"shuffle": gpt2_tokenizer_pt},
'AR':{'shuffle': gpt2_tokenizer_ar},
'ZHRN': {"shuffle": gpt2_tokenizer_zh},
'PTRN': {"shuffle": gpt2_tokenizer_pt},
'ENRN': {"shuffle": gpt2_tokenizer_en},
'AANN': {"shuffle": gpt2_tokenizer_en},
'NNDA': {"shuffle": gpt2_tokenizer_en},
"TR1k":{"shuffle": gpt2_tokenizer_tr1k},
"EN1k":{"shuffle": gpt2_tokenizer_en1k},
"EN2k":{"shuffle": gpt2_tokenizer_en2k},
"TR2k":{"shuffle": gpt2_tokenizer_tr2k},
"EN3k":{"shuffle": gpt2_tokenizer_en3k},
"EN5k":{"shuffle": gpt2_tokenizer_en5k},
"TR3k":{"shuffle": gpt2_tokenizer_tr3k},
"TR15k":{"shuffle": gpt2_tokenizer_tr15k},
"TR10k":{"shuffle": gpt2_tokenizer_tr10k},
"TR5k":{"shuffle": gpt2_tokenizer_tr5k},
"TR20k":{"shuffle": gpt2_tokenizer_tr20k},
"DE1k":{"shuffle": gpt2_tokenizer_de1k},
"DE2k":{"shuffle": gpt2_tokenizer_de2k},
"DE3k":{"shuffle": gpt2_tokenizer_de3k},
"DE5k":{"shuffle": gpt2_tokenizer_de5k},
"RU1k":{"shuffle": gpt2_tokenizer_ru1k},
"RU2k":{"shuffle": gpt2_tokenizer_ru2k},
"RU3k":{"shuffle": gpt2_tokenizer_ru3k},
"RU5k":{"shuffle": gpt2_tokenizer_ru5k},
"RU05k":{"shuffle": gpt2_tokenizer_ru05k},
"TR05k":{"shuffle": gpt2_tokenizer_tr05k},
"EN05k":{"shuffle": gpt2_tokenizer_en05k},
"DE05k":{"shuffle": gpt2_tokenizer_de05k},
"AR1k":{"shuffle": gpt2_tokenizer_ar1k},
"AR2k":{"shuffle": gpt2_tokenizer_ar2k},
"AR3k":{"shuffle": gpt2_tokenizer_ar3k},
"AR5k":{"shuffle": gpt2_tokenizer_ar5k},
"AR05k":{"shuffle": gpt2_tokenizer_ar05k},
}

FUNCTION_MAP = {
    'perturb_det_num_adj_np': {'function': perturb_np_num_det_adj, 'seq':'dnan'},
    # 'perturb_adj_num_det_np': {'function': perturb_np_num_det_adj, 'seq': 'andn'},
    # 'perturb_np_adj_num_det': {'function': perturb_np_num_det_adj, 'seq':'nand'},
    'perturb_adj_num_np_det': {'function': perturb_np_num_det_adj, 'seq':'annd'},
    'perturb_det_adj_np_num': {'function': perturb_np_num_det_adj, 'seq':'dann'},
    'perturb_det_num_np_adj': {'function': perturb_np_num_det_adj, 'seq':'dnna'},
    'perturb_np_num_det_adj': {'function': perturb_np_num_det_adj, 'seq':'nnda'},
    'perturb_reverse_full': {'function': perturb_reverse_full},
    'perturb_reverse_full_word': {'function': perturb_reverse_full_word},
    'perturb_adj_num':{'function': perturb_adj_num, 'seed': None, 'shuffle': False, 'post':'NUM'},
    'perturb_num_adj':{'function': perturb_adj_num, 'seed': None, 'shuffle': False, 'post':'ADJ'},
    'shuffle_remove_fw':{'function': perturb_shuffle_remove_fw, 'seed': None, 'shuffle': False},
    'shuffle_local_word3': {'function': perturb_shuffle_local_word, 'seed': None, 'window': 3},
    'shuffle_local_word5': {'function': perturb_shuffle_local_word, 'seed': None, 'window': 5},
    'shuffle_local_word10': {'function': perturb_shuffle_local_word, 'seed': None, 'window': 10},
    'shuffle_control': {'function': perturb_shuffle_deterministic,'seed': None, 'shuffle': False},
    'shuffle_local2': {'function':perturb_shuffle_local,'seed': None,  'window': 2},
    'shuffle_local3': {'function':perturb_shuffle_local,'seed': None,  'window': 3},
    'shuffle_local5': {'function':perturb_shuffle_local,'seed': None,  'window': 5},
    'shuffle_local10': {'function':perturb_shuffle_local,'seed': None, 'window': 10},
    'shuffle_deterministic21':{'function': perturb_shuffle_deterministic, 'seed':21, 'shuffle':True},
    'shuffle_deterministic57': {'function': perturb_shuffle_deterministic, 'seed':57, 'shuffle':True},
    'shuffle_deterministic84': {'function': perturb_shuffle_deterministic, 'seed':84, 'shuffle':True},
    'shuffle_nondeterministic':{'function': perturb_shuffle_nondeterministic, 'rng': default_rng(0)},
    'shuffle_even_odd':{'function':perturb_shuffle_even_odd},
    'shuffle_deterministic21_word': {'function': perturb_shuffle_deterministic_word,'seed':21, 'shuffle':True},
    'shuffle_local3_word':{'function':perturb_shuffle_local,'seed': None,'window': 3}
}

def get_perturbations(lang, function):
    lang_name = lang.lower()
    function_name = function+'_'+lang_name
    if 'shuffle_local_word' in function:
        return {function_name: {
            "perturbation_function": partial(FUNCTION_MAP[function]['function'], lang=lang, seed=0,
                                             window=FUNCTION_MAP[function]['window']),
            "lang": lang_name,
            "affect_function": affect_shuffle,
            "filter_function": filter_shuffle,
            "gpt2_tokenizer": TOKENIZER[lang]['shuffle'],
        }}
    elif 'perturb_reverse_full_word' in function:
        return {function_name: {
            "perturbation_function": partial(FUNCTION_MAP[function]['function'], lang=lang),
            "lang": lang_name,
            "affect_function": affect_shuffle,
            "filter_function": filter_shuffle,
            "gpt2_tokenizer": TOKENIZER[lang]['shuffle'],}
        }
    elif 'perturb_adj_num_det_np' in function:
        return {function_name: {
            "perturbation_function": partial(FUNCTION_MAP[function]['function'], lang=lang, seq=FUNCTION_MAP[function]['seq']),
            "lang": lang_name,
            "affect_function": affect_shuffle,
            "filter_function": filter_shuffle,
            "gpt2_tokenizer": TOKENIZER[lang]['shuffle'], }
        }
    elif 'perturb_det_num_adj_np' in function:
        return {function_name: {
            "perturbation_function": partial(FUNCTION_MAP[function]['function'], lang=lang, seq=FUNCTION_MAP[function]['seq']),
            "lang": lang_name,
            "affect_function": affect_shuffle,
            "filter_function": filter_shuffle,
            "gpt2_tokenizer": TOKENIZER[lang]['shuffle'], }
        }
    elif 'perturb_np_adj_num_det' in function:
        return {function_name: {
            "perturbation_function": partial(FUNCTION_MAP[function]['function'], lang=lang, seq=FUNCTION_MAP[function]['seq']),
            "lang": lang_name,
            "affect_function": affect_shuffle,
            "filter_function": filter_shuffle,
            "gpt2_tokenizer": TOKENIZER[lang]['shuffle'], }
        }
    elif 'perturb_np_num_det_adj' in function:
        return {function_name: {
            "perturbation_function": partial(FUNCTION_MAP[function]['function'], lang=lang, seq=FUNCTION_MAP[function]['seq']),
            "lang": lang_name,
            "affect_function": affect_shuffle,
            "filter_function": filter_shuffle,
            "gpt2_tokenizer": TOKENIZER[lang]['shuffle'], }
        }
    elif 'perturb_adj_num_np_det' in function:
        return {function_name: {
            "perturbation_function": partial(FUNCTION_MAP[function]['function'], lang=lang, seq=FUNCTION_MAP[function]['seq']),
            "lang": lang_name,
            "affect_function": affect_shuffle,
            "filter_function": filter_shuffle,
            "gpt2_tokenizer": TOKENIZER[lang]['shuffle'], }
        }
    elif 'perturb_det_num_np_adj' in function:
        return {function_name: {
            "perturbation_function": partial(FUNCTION_MAP[function]['function'], lang=lang, seq=FUNCTION_MAP[function]['seq']),
            "lang": lang_name,
            "affect_function": affect_shuffle,
            "filter_function": filter_shuffle,
            "gpt2_tokenizer": TOKENIZER[lang]['shuffle'], }
        }
    elif 'perturb_det_adj_np_num' in function:
        return {function_name: {
            "perturbation_function": partial(FUNCTION_MAP[function]['function'], lang=lang, seq=FUNCTION_MAP[function]['seq']),
            "lang": lang_name,
            "affect_function": affect_shuffle,
            "filter_function": filter_shuffle,
            "gpt2_tokenizer": TOKENIZER[lang]['shuffle'], }
        }
    elif 'perturb_reverse_full' in function:
        return {function_name: {
            "perturbation_function": partial(FUNCTION_MAP[function]['function'], lang=lang),
            "lang": lang_name,
            "affect_function": affect_shuffle,
            "filter_function": filter_shuffle,
            "gpt2_tokenizer": TOKENIZER[lang]['shuffle'],}
        }
    elif 'perturb_adj_num' in function:
        return {function_name: {
            "perturbation_function": partial(FUNCTION_MAP[function]['function'], lang=lang, post=FUNCTION_MAP[function]['post']),
            "lang": lang_name,
            "affect_function": affect_shuffle,
            "filter_function": filter_shuffle,
            "gpt2_tokenizer": TOKENIZER[lang]['shuffle'],
        }}
    elif 'perturb_num_adj' in function:
        return {function_name: {
            "perturbation_function": partial(FUNCTION_MAP[function]['function'], lang=lang, post=FUNCTION_MAP[function]['post']),
            "lang": lang_name,
            "affect_function": affect_shuffle,
            "filter_function": filter_shuffle,
            "gpt2_tokenizer": TOKENIZER[lang]['shuffle'],
        }}
    elif 'shuffle_remove_fw' in function:
        return {function_name: {
            "perturbation_function": partial(FUNCTION_MAP[function]['function'], lang=lang),
            "lang": lang_name,
            "affect_function": affect_shuffle,
            "filter_function": filter_shuffle,
            "gpt2_tokenizer": TOKENIZER[lang]['shuffle'],
        }}
    elif 'shuffle_local' in function:
        return {function_name: {
            "perturbation_function": partial(FUNCTION_MAP[function]['function'], lang=lang,seed=0,  window=FUNCTION_MAP[function]['window']),
            "lang": lang_name,
            "affect_function": affect_shuffle,
            "filter_function": filter_shuffle,
            "gpt2_tokenizer": TOKENIZER[lang]['shuffle'],
        }}
    elif 'shuffle_deterministic' in function:
        return {
            function_name: {
            "perturbation_function": partial(FUNCTION_MAP[function]['function'], lang=lang, seed=FUNCTION_MAP[function]['seed'], shuffle=FUNCTION_MAP[function]['shuffle']),
            "lang": lang_name,
            "affect_function": affect_shuffle,
            "filter_function": filter_shuffle,
            "gpt2_tokenizer": TOKENIZER[lang]['shuffle'],
        }}
    elif 'shuffle_control' in function:
        return {
            function_name: {
                "perturbation_function": partial(FUNCTION_MAP[function]['function'], lang=lang,
                                                 seed=None,
                                                 shuffle=False),
                "lang": lang_name,
                "affect_function": affect_shuffle,
                "filter_function": filter_shuffle,
                "gpt2_tokenizer": TOKENIZER[lang]['shuffle'],
            }}
    elif 'shuffle_even_odd' in function:
        return {
            function_name: {
                "perturbation_function": partial(FUNCTION_MAP[function]['function'], lang=lang),
                "lang": lang_name,
                "affect_function": affect_shuffle,
                "filter_function": filter_shuffle,
                "gpt2_tokenizer": TOKENIZER[lang]['shuffle'],
            }}

    elif 'shuffle_nondeterministic' in function:
        return {
            function_name: {
                "perturbation_function": partial(FUNCTION_MAP[function]['function'], lang=lang, rng=FUNCTION_MAP[function]['rng']),
                "lang": lang_name,
                "affect_function": affect_shuffle,
                "filter_function": filter_shuffle,
                "gpt2_tokenizer": TOKENIZER[lang]['shuffle'],
            }}

    else:
        raise WarningMessage('The pertubation is not available!')

