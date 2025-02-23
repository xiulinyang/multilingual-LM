from pathlib import Path
from tqdm import tqdm
import json
from collections import Counter
import nltk
import random
random.seed(41)
from nltk import Tree
import random


def shuffle_until_different(lst):
    if len(lst) < 2:
        return lst
    shuffled = lst[:]
    i =0
    while shuffled == lst and i <3:
        random.shuffle(shuffled)
        i += 1
    return shuffled

'''
IT: 'sn'
'''
def reorder_np(np_subtree):
    desired = []
    others = []
    # desired_order = ['NN', 'NNS', 'NNP', 'NNPS', 'QP', '$', 'CD', 'DT','PRP$', 'PDT', 'POS', 'RB', 'ADJP', 'JJR', 'JJS', 'JJ', ]
    # desired_order = ['NOUN', 'X', 'SYM', 'PROPN', 'PRON', 'sa', 'ADJ', 'DET', 'sq']
    # desired_order = ['FW', 'NP', 'NN', 'NR', 'NT', 'PRP', 'PN', 'DEG', 'JJ', 'ADJP', 'DNP', 'DEC', 'DT', 'M', 'CLP', 'DP', 'CD', 'OD','QP']
    desired_order = ["DET", "D", "DEM", "POSS", "POSS''", "N'","NOUN", "PRON", "QNT", "QNT'", "NUM","PERCENTP","PERCENTP'" "CARD","CARD'","ADJ", "AP"]
    for child in np_subtree:
        if hasattr(child, "label") and child.label() in desired_order:
            desired.append(child)
        else:
            others.append(child)
    desired = shuffle_until_different(desired)
    # random.shuffle(desired)
    reordered_children = []
    desired_iter = iter(desired)
    for child in np_subtree:
        if child in desired:
            reordered_children.append(next(desired_iter))
        else:
            reordered_children.append(child)

    np_subtree[:] = reordered_children

def navigate_and_reorder_tree(t):
    for subtree in t:
        try:
            if subtree.label() in ["NP", "N'"]:
                print(subtree)
                reorder_np(subtree)
                print(subtree)
            navigate_and_reorder_tree(subtree)
        except AttributeError:
            continue
    return t

en_file = open('/Users/xiulinyang/Desktop/PT_parsed.json', 'r')
new_en_pos = open('PTRN.json', 'w')
en = json.load(en_file)


updated_json = []
for sent_annotates in tqdm(en):
    for sent_text in sent_annotates['sent_annotations']:
        constituent = sent_text['constituency_parse']
        word_annotation = [{'upos':x['upos'],'xpos': x['xpos'], 'text': x['text']} for x in sent_text['word_annotations']]
        t = Tree.fromstring(constituent)
        t = navigate_and_reorder_tree(t)
        leaves = t.leaves()
        text = ' '.join(leaves)
        sent_meta = {'sent_annotations': [{'sent_text': text}]}
        updated_json.append(sent_meta)


json.dump(updated_json, new_en_pos, ensure_ascii=False, indent=4)
#
# en = Path('en_pos_parsed.txt').read_text().strip().split('\n')
#
# en = [x for x in en if x]
# freq_en = Counter(en)
# print(freq_en.most_common())
# en_pos = Path('en_pos.txt').read_text().strip().split('\n')
# all_bigrams = []
# for de_line in en_pos:
#     bigrams = [' '.join([x, y]) for x, y in zip(de_line.split()[:-1], de_line.split()[1:])]
#     all_bigrams.extend(bigrams)
# bigram100 = Counter(all_bigrams).most_common(100)
#
# print(bigram100)
# bi_rep = []
# bi_filtered = []
# for bi, _ in bigram100:
#     if len(set(bi.split()))==1:
#         continue
#     elif Counter(bi.split()) not in bi_rep:
#         bi_rep.append(Counter(bi.split()))
#         bi_filtered.append(bi)
#     else:
#         continue
#
# s = 0
# with open('test.txt', 'w') as t:
#     for x in tqdm(bi_filtered):
#         t.write(f'{x}\n')
#         freq = Counter(all_bigrams)[x]
#         s+=freq
#
#
# def longest_continuous_span(nums, target):
#     if target not in nums:
#         return []
#     nums = sorted(nums)
#     current_span = []
#     all_spans = []
#     for num in nums:
#         if not current_span or num == current_span[-1] + 1:
#             current_span.append(num)
#         else:
#             current_span = [num]
#         all_spans.append(current_span)
#     final = [x for x in all_spans if target==x[-1]]
#     if final:
#         return final[-1]
#     else:
#         return []
#
#
#
#
# def convert_sent_order(sent, prep, post):
#     sent_meta = {}
#     sent_pos = [x['upos'] for x in sent['word_annotations']]
#     sent_pos_copy = [x['upos'] for x in sent['word_annotations']]
#     sent_toks = [x['text'] for x in sent['word_annotations']]
#     sent_toks_copy = [x['text'] for x in sent['word_annotations']]
#     adj_pos = [x for x, y in enumerate(sent_pos) if y==post]
#     for i, pos in enumerate(sent_pos):
#         if pos == prep:
#             adj_indices = longest_continuous_span(adj_pos, i - 1)
#             if i - 1 in adj_indices:
#                 len_adj = len(adj_indices)
#                 sent_toks_copy[i-len_adj] = sent_toks[i]
#                 sent_pos_copy[i-len_adj] = sent_pos[i]
#                 sent_toks_copy[i-len_adj+1:i+1] = sent_toks[adj_indices[0]: adj_indices[-1]+1]
#                 sent_pos_copy[i-len_adj+1:i+1] = sent_pos[adj_indices[0]: adj_indices[-1]+1]
#     assert len(sent_toks_copy) == len(sent_toks)
#     assert len(sent_pos_copy) == len(sent_pos)
#
#     sent_meta['word_annotations'] = [{'upos': x, 'text': y} for x,y in zip(sent_pos_copy, sent_toks_copy)]
#
#     return sent_meta
#
# de_file = open('/Users/xiulinyang/Desktop/EN.json')
# de = json.load(de_file)
# reverse = []
# with open('en_pos.json', 'w') as de_pos:
#     for exp in tqdm(de):
#         for sent in exp['sent_annotations']:
#             sent_annotate = copy.deepcopy(sent)
#             for collo, _ in bigram100:
#                 post, prep = collo.split()
#                 sent_annotate = convert_sent_order(sent_annotate, prep, post)
#                 # print(' '.join([x['text'] for x in sent_annotate['word_annotations']]), ' '.join([x['upos'] for x in sent_annotate['word_annotations']]))
#             tw = ' '.join([x['text'] for x in sent_annotate['word_annotations']])
#             pos_tw = ' '.join([x['upos'] for x in sent_annotate['word_annotations']])
#             reverse.append({"sent_annotations": [{"sent_text":tw, "pos": pos_tw}]})
#
#     json.dump(reverse, de_pos,ensure_ascii=False, indent=4)
#






# for b, f in Counter(all_bigrams):
#     two_pos = b.split()
#     if len(set(two_pos))!=1:
#         s+=f
#     else:
#         print(two_pos)

# print(s)
# print(len(all_bigrams))

#
# with open('adj_noun.json', 'w') as adjn:
#     json.dump(Counter(all_bigrams), adjn)
# de_file = open('/Users/xiulinyang/Downloads/EN.json')
# de = json.load(de_file)
#
# de_no_function=[]
# with open('/Users/xiulinyang/Desktop/DENF.json', 'w') as denf:
#     for sent_annotate in tqdm(de):
#         for word_annotate in sent_annotate['sent_annotations']:
#             sent = ' '.join([x["text"] for x in word_annotate['word_annotations'] if x['upos'] not in ['DET', 'AUX', 'ADP', 'SCONJ', 'PART','CCONJ']])
#             de_no_function.append({"sent_annotations":[{"sent_text": sent}]})
#
#     json.dump(de_no_function, denf, ensure_ascii=False, indent=4)


# import json
# from tqdm import tqdm
# de_file = open('/Users/xiulinyang/Desktop/EN.json')
# de = json.load(de_file)
# with open('en_pos.txt', 'w') as de_pos:
#     for sent_annotate in tqdm(de):
#         for word_annotate in sent_annotate['sent_annotations']:
#             pos_info = ' '.join([x['upos'] for x in word_annotate['word_annotations']])
#             de_pos.write(f'{pos_info}\n')

