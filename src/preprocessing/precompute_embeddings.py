'''
precompute_word_level_ver2
For all the word embeddings
references:
https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/

'''


import os
import torch
import json
from collections import OrderedDict
import pickle
from tqdm import tqdm
import argparse
import transformers as ppb # pytorch transformers
from transformers import BertModel, BertTokenizer
from transformers import RobertaTokenizer, RobertaModel
from transformers import GPT2Tokenizer, GPT2Model
from transformers import ElectraTokenizer, ElectraModel
from transformers import TransfoXLTokenizer, TransfoXLModel


def mkdir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)


def compute_embeddings(idir, ofile, tokenizer, model, add_pref_suf=True):
    '''
    :param idir:
    :param ofile: .pkl file
    :param tokenizer:
    :param model:
    :param add_pref_suf:
    :return:
    '''
    if os.path.isfile(ofile):
        print("File exist! Skip!", ofile)
        return
    print('Running for ', idir)
    token_ofile = "{}.json".format(ofile[:-4])

    fname2feats = {}  # {filename: {id: feature}}  # features for all the tokens
    fname2tokens = {}  # {filename: {id: tokens}}

    for fname in tqdm(os.listdir(idir)):
        if fname.startswith('.'):
            continue
        data = json.load(open(os.path.join(idir, fname), 'r'))
        id2feats = OrderedDict()
        id2tokens = OrderedDict()
        for event in data:
            marked_text = ' '.join(event['tokens'])
            if add_pref_suf:
                marked_text = PREFIX + marked_text + SUFFIX
            tokenized_text = tokenizer.tokenize(marked_text)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [1] * len(tokenized_text)
            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens]).to(device)
            segments_tensors = torch.tensor([segments_ids]).to(device)

            with torch.no_grad():
                if model_type != 'transformerxl':
                    outputs = model(input_ids=tokens_tensor, attention_mask=segments_tensors)
                else:
                    outputs = model(input_ids=tokens_tensor)
                last_hidden = outputs[0].squeeze(0)
                id2feats[event['event_id']] = last_hidden.to('cpu')
                id2tokens[event['event_id']] = tokenized_text
        fname2feats[fname] = id2feats
        fname2tokens[fname] = id2tokens
    pickle.dump(fname2feats, open(ofile, 'wb'))
    json.dump(fname2tokens, open(token_ofile, 'w'))
    print("Saved features to ", ofile)
    print("Saved tokens to ", token_ofile)


ap = argparse.ArgumentParser()
ap.add_argument('--model', choices=['roberta', 'gpt2', 'electra', 'bert', 'distill', 'transformerxl'], default='roberta')
ap.add_argument('--gpu', choices=[0, 1, 2, 3], default=2, type=int)
# ap.add_argument('--output', default='wb5-word_level')
ap.add_argument('--input', default='data/preprocessed/window-based_5s', help='path to preprocessed window-based data')
ap.add_argument('--output', default='data/preprocessed/embeddings/', help='path to embeddings dir')

args = ap.parse_args()

'''
precompute for window-based
output: pickle files
'''

# odirname = 'words_window-based_5_seconds'
# idir = 'data/lowercased/window_based_data/words_window-based_5_seconds'

# idir = 'data/EWE_sentence_level_standardized'
# odirname = 'words_sentence-based'

# odirname = args.output
if args.input == 'sentence':
    idir = 'data/EWE_sentence_level_standardized'
    odirname = 'sent-word_level'
else:
    idir = 'data/lowercased/window_based_data/words_window-based_5_seconds'
    odirname = 'wb5-word_level'
sfile = 'data/_TrainSetAssignments/splits.json'

model_type = args.model
gpu = args.gpu
idir = args.input
odir = os.path.join(args.output, "{}_{}".format(os.path.basename(idir), model_type))

add_pref_suf = True
PREFIX = None
SUFFIX = None
# model_type = 'bert'
print("Using {}".format(model_type))
mkdir(odir)

if model_type == 'roberta':
    model_class = 'roberta-base'
    add_pref_suf = True
    PREFIX = '<s>'
    SUFFIX = '</s>'
    tokenizer = RobertaTokenizer.from_pretrained(model_class)
    # model = RobertaModel.from_pretrained(model_class, return_dict=True)
    model = RobertaModel.from_pretrained(model_class)
elif model_type == 'gpt2':
    add_pref_suf = False
    model_class = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_class)
    # model = GPT2Model.from_pretrained(model_class, return_dict=True)
    model = GPT2Model.from_pretrained(model_class)
elif model_type == 'electra':
    model_class = 'google/electra-small-discriminator'
    add_pref_suf = True
    PREFIX = '[CLS]'
    SUFFIX = '[SEP]'
    tokenizer = ElectraTokenizer.from_pretrained(model_class)
    # model = ElectraModel.from_pretrained('google/electra-small-discriminator', return_dict=True)
    model = ElectraModel.from_pretrained(model_class)
elif model_type == 'bert':
    add_pref_suf = True
    PREFIX = '[CLS]'
    SUFFIX = '[SEP]'
    model_class = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_class)
    model = BertModel.from_pretrained(model_class)
elif model_type == 'distill':
    add_pref_suf = True
    PREFIX = '[CLS]'
    SUFFIX = '[SEP]'
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
elif model_type == 'transformerxl':  # transformer XL
    add_pref_suf = False
    model_class = 'transfo-xl-wt103'
    tokenizer = TransfoXLTokenizer.from_pretrained(model_class)
    model = TransfoXLModel.from_pretrained(model_class)
else:
    exit(0)

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.set_device(gpu)

sets = ['test', 'train', 'valid']

model.to(device)
model.eval()

for setname in sets:
    compute_embeddings(idir=os.path.join(idir, setname), ofile=os.path.join(odir, '{}.pkl'.format(setname)), tokenizer=tokenizer,
                       model=model, add_pref_suf=add_pref_suf)

