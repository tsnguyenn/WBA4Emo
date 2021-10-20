'''
not yet support glove
'''

import torch
from torch import nn
import sys
sys.path.append("/home/son/WBA4Emo")
import torch.nn.functional as F
from src.utils import scoring
import os
import json
import re
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import pandas
import time
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
import argparse
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
import numpy as np
import pickle
from tqdm import tqdm
import math


class WBA(nn.Module):
    def __init__(self, input_dim, hid1_dim, hid1_bi, hid1_nlayers, max_num_words,
                 hid2_dim, hid2_bi, hid2_nlayers, batch_first, device, dropout_p, use_attention):
        super(WBA, self).__init__()
        self.lstm1_factor = (2 if hid1_bi else 1) * hid1_nlayers
        self.hid1_dim = hid1_dim
        self.lstm2_factor = (2 if hid2_bi else 1) * hid2_nlayers
        self.hid2_dim = hid2_dim
        self.device = device
        self.batch_first = batch_first
        self.event_num_words = max_num_words
        self.train_mode = True
        self.use_attention = use_attention

        self.dropout = nn.Dropout2d(p=dropout_p)

        # 1st-layer LSTM: input: word embedding; output: sentence representation
        self.lstm1 = nn.LSTM(input_dim, hid1_dim, bidirectional=hid1_bi, num_layers=hid1_nlayers, batch_first=batch_first)
        tmp_dim_1 = self.lstm1_factor * hid1_dim
        self.attention = nn.Sequential(nn.Linear(tmp_dim_1, tmp_dim_1),
                                       nn.Tanh(),
                                       nn.Linear(tmp_dim_1, 1))
        self.att_11 = nn.Sequential(nn.Linear(self.lstm1_factor*hid1_dim, 1),
                                    nn.Tanh())
        self.att_12 = nn.Linear(max_num_words, max_num_words)

        # 2nd-layer LSTM: input: sentence representation, output: prediction
        self.lstm2 = nn.LSTM(self.lstm1_factor * hid1_dim, hid2_dim, bidirectional=hid2_bi, num_layers=hid2_nlayers, batch_first=batch_first)

        self.valence = nn.Sequential(nn.Linear(self.lstm2_factor * hid2_dim, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 1))


    def forward(self, x, sentence_lens, word_lens):
        '''
        :param x: [batch, num_sentences, num_words, word embedding]
        :param sentence_lens:
        :param word_lens: [batch, num_sentences]
        :return:
        '''
        if self.train_mode:
            x = self.dropout(x)

        x1, att1 = self.get_sentence_representation(x, word_lens)         # [batch, num_sentences, (2 if bi else 1) * hidden1_size]

        # level 2 LSTM
        # x1_packed = pack_padded_sequence(x1, sentence_lens, batch_first=self.batch_first)
        x1_packed = pack_padded_sequence(x1, sentence_lens.cpu(), batch_first=self.batch_first)
        hidden = self.init_hidden_level2(x1.shape[0])
        hiddens, _ = self.lstm2(x1_packed, hidden)
        hiddens, _ = pad_packed_sequence(hiddens, batch_first=self.batch_first)

        # Prediction
        # output = self.reg(hiddens)
        output = self.valence(hiddens)
        # print_tensor('output', output, False)
        return output, att1

    def get_sentence_representation(self, x, word_lens):
        '''
        1st-layer LSTM: sentence level  (checked for 1-layer 1-directional LSTM)
        :param x: [batch, num_sentences, num_words, word embedding]
        :param word_lens: [batch, num_sentences]
        :return: [batch, num_sentences, (2 if bi else 1) * hidden1_size]  (sentence representations)
        '''
        # print_tensor('x', x, False)
        local_batch_size, local_num_sentences = x.shape[0], x.shape[1]

        # convert 4-dim input to 3-dim input. Assuming batch * num_sentences = batch size for the 1st layer lstm
        x = x.view(-1, x.shape[2], x.shape[3])  # [batch * num_sentences, num_words, word_embedding]
        # print_tensor('x2', x, False)
        word_lens = word_lens.view(-1)  # [batch * num_sentences]

        # sort the new input by number of words (to feed to the 1st lstm). Keep the unsort indices info for unsort the output
        lens_sort, idx_sort = torch.sort(word_lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)  # for later unsorting
        x_sort = torch.index_select(x, dim=0, index=idx_sort)

        # remove the 0-len "sentences"
        tmp = (lens_sort != 0).nonzero().squeeze()
        x_sort = x_sort[tmp]
        lens_sort_non_zeros = lens_sort[tmp]

        # run lstm 1
        # print(lens_sort_non_zeros)
        # x_sort_pad = pack_padded_sequence(x_sort, lens_sort_non_zeros, batch_first=self.batch_first)
        x_sort_pad = pack_padded_sequence(x_sort, lens_sort_non_zeros.cpu(), batch_first=self.batch_first)
        hidden1 = self.init_hidden_level1(x_sort.shape[0])
        h1, (h1n, c1n) = self.lstm1(x_sort_pad, hidden1)

        if self.use_attention:
            h1, lens_tmp = pad_packed_sequence(h1, batch_first=self.batch_first)
            att_12 = self.attention(h1).squeeze(2)
            # masked softmax
            att1 = mask_softmax_full(att_12, lens_tmp, self.event_num_words)
            x_att = torch.sum(att1.unsqueeze(-1) * h1, dim=1)  # [batch * num_sentences - #empty_sentences, (2 if bi else 1) * hidden1_size]

            # add zero back
            num_zeros = lens_sort.shape[0] - lens_sort_non_zeros.shape[0]
            zeros = torch.zeros(num_zeros, x_att.shape[1]).to(self.device)
            x_att = torch.cat((x_att, zeros), dim=0)  # [batch * num_sentences, (2 if bi else 1) * hidden1_size]
            zeros_att = torch.zeros(num_zeros, att1.shape[1]).to(self.device)
            att1 = torch.cat((att1, zeros_att), dim=0)
            # unsort the sentences (putting back to the stories)
            x_att = torch.index_select(x_att, dim=0, index=idx_unsort)  # [batch * num_sentences, (2 if bi else 1) * hidden1_size] (checked!)
            att1 = torch.index_select(att1, dim=0, index=idx_unsort)  # [batch * num_sentences, (2 if bi else 1) * hidden1_size] (checked!)
            sent_rep = x_att.view(local_batch_size, local_num_sentences, -1)  # [batch, num_sentences, (2 if bi else 1) * hidden1_size]
            att1 = att1.view(local_batch_size, local_num_sentences, -1)
        else:
            # do not use the attention
            h1n = torch.transpose(h1n, 0, 1)
            h1n = h1n.reshape(h1n.shape[0], h1n.shape[1] * h1n.shape[2])
            # add zero back
            num_zeros = lens_sort.shape[0] - lens_sort_non_zeros.shape[0]
            zeros = torch.zeros(num_zeros, h1n.shape[1]).to(self.device)
            # append zeros
            h1n = torch.cat((h1n, zeros), dim=0)  # [batch * num_sentences, (2 if bi else 1) * hidden1_size]
            # unsort the sentences (putting back to the stories)
            h1n = torch.index_select(h1n, dim=0, index=idx_unsort)
            sent_rep = h1n.view(local_batch_size, local_num_sentences, -1)  # [batch, num_sentences, (2 if bi else 1) * hidden1_size]
            att1 = None
        return sent_rep, att1

    def init_hidden_level1(self, batch_size):
        return (torch.zeros(self.lstm1_factor, batch_size, self.hid1_dim).to(self.device),
                torch.zeros(self.lstm1_factor, batch_size, self.hid1_dim).to(self.device))

    def init_hidden_level2(self, batch_size):
        return (torch.zeros(self.lstm2_factor, batch_size, self.hid2_dim).to(self.device),
                torch.zeros(self.lstm2_factor, batch_size, self.hid2_dim).to(self.device))


class StoryDataset(Dataset):
    def __init__(self, set_name, embedding_dir, data_dir, min_len, max_len,
                 actual, scaling, start_score, num_words, event_score_type, emb_size=768, has_pre_suf=True):
        print("Loading data for ", set_name)
        self.actual = actual
        self.num_event_words = num_words
        self.scaling = scaling
        self.start_score = start_score
        self.emb_size = emb_size
        self.ids = []   # stories ids
        self.embds = {} # {filename: Tensor(num_events, num_words, 300)}
        self.event_ids = {}    # {video_id: [window_ids]}
        self.valence_scores = {}
        self.salient_scores = {}
        self.min_len = min_len
        self.max_len = max_len
        self.event_score_type = event_score_type
        self.other_info = {}    # e.g., len of the events (#tokens, position of the event in the story)
        self.event_info = {}  # {video_id: {window_id: (start_time, end_time, #values)}}
        self.event_details = {}  # {video_id: {event_id: "{start_time}\t{end_time}\t{tokens}\t{real_value_groundtruth}\t{groundtruth}"}}
        self.event_lens = {}
        self.vid_sent_mapping = {}  # {video_id: [sentence_ids]}
        self.vid_sent_content = {}  # {video_id: {event_id: [tokens]}
        self.load_data(os.path.join(data_dir, set_name),
                       os.path.join(embedding_dir, '{}.pkl'.format(set_name)), os.path.join(embedding_dir, '{}.json'.format(set_name)),
                       has_pre_suf=has_pre_suf)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        name = self.ids[item]
        # return self.X[video_id], self.y[video_id], video_id, self.vid_sent_words_lens[video_id]
        # return self.embds[name], self.valence_scores[name], self.event_ids[name], name, self.salient_scores[name], self.other_info[name]
        return self.embds[name], self.valence_scores[name], name, self.event_lens[name]

    def load_data(self, data_dir, embedding_file, info_file, has_pre_suf=True):
        word_embeddings = pickle.load(open(embedding_file, 'rb'))
        tokens_info = json.load(open(info_file, 'r'))

        for name in tqdm(os.listdir(data_dir)):
            if name.startswith('.'):
                continue
            event_embds = word_embeddings[name]
            event_tokens = tokens_info[name]

            events_embeddings = []  # [events in file] (n, #words, 300)

            valid_valence_scores = []
            valid_salient_scores = []
            valid_event_ids = []
            event_content = {}
            sent_ids = []
            event_info = {}
            details = {}
            event_lens = []
            prev_score = self.start_score
            other_info = []
            if self.scaling:
                prev_score = std_single_score(start_score)
            ifile = os.path.join(data_dir, name)
            events = json.load(open(ifile, 'r', encoding='utf8'))

            for event in events:
                eid = event['event_id']

                # extract embedding
                tmp, event_len, valid_tokens = self.get_features(event_tokens[str(eid)], event_embds[eid], has_pre_suf=has_pre_suf)  # x_tmp shape: [#words, embeddings]

                # events_embeddings.append(torch.mean(torch.Tensor(tmp), dim=0))      # event embeddings = average words' embeddings
                events_embeddings.append(tmp)      # event embeddings = average words' embeddings
                # score
                if self.scaling:
                    event_scores = scale_scores(event['scores'])
                else:
                    event_scores = event['scores']

                if self.event_score_type == "last":
                    real_score = event_scores[-1]  # predict the mean/last score
                else:
                    real_score = np.mean(event_scores)  # predict the mean/last score

                if not self.actual:
                    score = real_score - prev_score
                    prev_score = real_score
                else:
                    score = real_score
                event_lens.append(event_len)
                event_content[eid] = valid_tokens
                sent_ids.append(eid)
                valid_valence_scores.append(score)
                vtmp = np.std(event['scores'])
                valid_salient_scores.append(vtmp)
                valid_event_ids.append(eid)
                other_info.append([event_len, len(valid_event_ids)])     # event len, event order
                event_info[eid] = (event['event_start'], event['event_end'])
                details[eid] = "{}\t{}\t{}\t{}\t{}\t{}".format(eid, event['event_start'], event['event_end'], event['tokens'], real_score, vtmp)
            self.event_details[name] = details
            self.ids.append(name)
            self.embds[name] = torch.stack(events_embeddings)  # Tensor(#windows, #events, embeddings)
            self.valence_scores[name] = torch.Tensor(valid_valence_scores)
            self.salient_scores[name] = torch.Tensor(valid_salient_scores)
            self.event_ids[name] = valid_event_ids
            self.event_info[name] = event_info
            self.other_info[name] = torch.Tensor(other_info)
            self.event_lens[name] = event_lens
            self.vid_sent_mapping[name] = sent_ids
            self.vid_sent_content[name] = event_content

    def get_features(self, words, embeddings, has_pre_suf=True):
        if has_pre_suf:
            valid_words = words[1:-1]  # ignore [CLS] and [SEP]
            valid_embds = embeddings[1:-1]
        else:
            valid_words = words
            valid_embds = embeddings

        if len(valid_words) > self.num_event_words:
            valid_words = valid_words[:self.num_event_words]
            valid_embds = valid_embds[:self.num_event_words]
            actual_len = self.num_event_words
        else:
            tmp = self.num_event_words - len(valid_words)
            actual_len = len(valid_words)
            if tmp > 0:
                for _ in range(tmp):
                    valid_embds = torch.cat((valid_embds, torch.zeros(1, self.emb_size)), dim=0)
        return valid_embds, actual_len, valid_words


def mkdir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)


def print_tensor(t_label, t_content, details=False):
    if details:
        print("{}: {}\n{}".format(t_label, t_content.shape, t_content))
    else:
        print("{}: {}".format(t_label, t_content.shape))


def my_padding(batch):
    '''
    Apply padding for each batch (different length of sentences)
    self.embds[name], self.valence_scores[name], name, self.event_lens[name]
    :param batch: x [#sentences, n_words, embeddings], y [#sentences, scores], video_id, [sentences' actual lengths]
    :return:
    '''
    xs = []
    ys = []
    video_ids = []
    sentences_count = []
    words_count = []
    for data_item in batch:
        video_ids.append(data_item[2])
        # print("Data X: {}\n{}".format(data_item[0].shape, data_item[0]))
        # print("Data Y: {}\n{}".format(data_item[1].shape, data_item[1]))
        # print("#words: {}".format(data_item[3]))
        xs.append(data_item[0])
        ys.append(data_item[1])
        sentences_count.append(data_item[0].shape[0])
        words_count.append(torch.IntTensor(data_item[3]))
        # words_count.append(data_item[3])

    xs_padded = pad_sequence(xs, batch_first=batch_first, padding_value=0)
    ys_padded = pad_sequence(ys, batch_first=batch_first)
    words_count = pad_sequence(words_count, batch_first=batch_first, padding_value=0)

    # print("Padded: ")
    # print("X: {}\n{}".format(xs_padded.shape, xs_padded))
    # print("Y: {}\n{}".format(ys_padded.shape, ys_padded))
    # print("#words: {}".format(words_count))

    # sort lens
    sorted_sentence_lens, sorted_idx = torch.sort(torch.IntTensor(sentences_count), dim=0, descending=True)
    sorted_video_ids = [video_ids[i.item()] for i in sorted_idx]

    iselect_dim = 1
    if batch_first:
        iselect_dim = 0
    sorted_xs_padded = torch.index_select(xs_padded, dim=iselect_dim, index=sorted_idx)
    sorted_ys_padded = torch.index_select(ys_padded, dim=iselect_dim, index=sorted_idx)
    sorted_words_count = torch.index_select(words_count, dim=0, index=sorted_idx)

    # print("Sorted: ")
    # print("X: {}\n{}".format(sorted_xs_padded.shape, sorted_xs_padded))
    # print("Y: {}\n{}".format(sorted_ys_padded.shape, sorted_ys_padded))
    # print("#words: {}\n{}".format(sorted_words_count.shape, sorted_words_count))
    # checked done.
    return sorted_xs_padded, sorted_ys_padded, sorted_video_ids, sorted_sentence_lens, sorted_words_count


def init_model():
    return WBA(input_dim=args.emb_dim, hid1_dim=hid1_dim, hid1_bi=hid1_bi, hid1_nlayers=hid1_nlayers, max_num_words=event_num_words,
               hid2_dim=hid2_dim, hid2_bi=hid2_bi, hid2_nlayers=hid2_nlayers, batch_first=batch_first,
               device=device, dropout_p=dropout_p, use_attention=use_attention)

def train(run_num):
    model = init_model()
    if not os.path.isfile(setting_file):
        with open(setting_file, 'w') as writer:
            writer.write(get_setting(model.__str__()))
    model.to(device)
    log_writer = SummaryWriter()
    loss_func = nn.MSELoss()
    if optimizer_opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    start = time.time()
    best_ccc = 0
    best_output = None
    global_iter = 0
    best_epoch = 0
    # a1 = 0.0
    for epoch in range(n_epoch):
        # print("att_12: {}".format(model.att_12.weight.sum()))
        # with torch.no_grad():
        #     a2 = model.att1.weight.sum()
        #     if a2 != a1:
        #         print("Diff: {}".format(a2 * 100000))
        #     a1 = a2
        scheduler.step()
        total_loss = 0
        counter = 0
        model.train()
        model.train_mode = True
        for x, y, ids, sentence_lens, words_lens in trainloader:
            # print("Word lens: {}".format(words_lens))
            counter += 1
            optimizer.zero_grad()
            x, y, sentence_lens, words_lens = x.to(device), y.to(device), sentence_lens.to(device), words_lens.to(device)
            pred, _ = model(x, sentence_lens, words_lens)
            pred = mask_prediction(pred, sentence_lens)
            # print_tensor("pred", pred, True)
            # print_tensor("y", y, True)
            loss = loss_func(pred.squeeze(-1), y)
            # print("loss: {}".format(loss))
            # input("Enter")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            log_writer.add_scalar("MSELoss", loss.item(), global_iter)
            global_iter += 1

            if counter % print_interval == 0:
                print("[{:10}]. Epoch: {:3}. Iter: {:4}. "
                      "LR: {:7.6f}. Loss: {:>10.6f}".format(time_since(start), epoch, counter, scheduler.get_lr()[0],
                                                            total_loss/counter))
        # evaluate ccc
        ccc_train, rho_train = evaluate(model, trainloader, train_gt)
        ccc_dev, rho_dev = evaluate(model, devloader, dev_gt)
        ccc_test, rho_test = evaluate(model, testloader, test_gt)

        log_writer.add_scalar("CCC_Train", ccc_train, epoch)
        log_writer.add_scalar("CCC_Dev", ccc_dev, epoch)
        log_writer.add_scalar("CCC_Test", ccc_test, epoch)

        score_writer.write("{}\t{}\t{}\t{}\t\t{}\t{}\t{}\t{}\n".format(epoch, ccc_train, ccc_dev, ccc_test,
                                                                       epoch, rho_train, rho_dev, rho_test))
        ccc_to_compare = ccc_dev
        if not earlyStop_dev:
            ccc_to_compare = ccc_test
        if ccc_to_compare > best_ccc:
            best_output = save_model(model, epoch, best=True, run=run_num)
            best_ccc = ccc_to_compare
            best_epoch = epoch
        print("{:10}. Epoch {:3}. Total loss: {:7.4f}. LR: {:7.6f}. "
              "ccc_train: {:4.3f}. ccc_dev: {:4.3f}. ccc_test: {:4.3f}".format(time_since(start), epoch, total_loss, scheduler.get_lr()[0],
                                                                               ccc_train, ccc_dev, ccc_test))
    # save the last model
    last_output = save_model(model, n_epoch-1, run=run_num)
    log_writer.close()
    return last_output, best_output, best_epoch


def load_model(model_path):
    model = init_model()
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    return model


def save_model(model, epoch, best=False, run=None):
    if run is not None:
        model_output = os.path.join(model_odir, "run_{}_".format(run))
    else:
        model_output = os.path.join(model_odir, '')

    if best:
        model_output = "{}best.pt".format(model_output)
    else:
        model_output = "{}epoch_{}.pt".format(model_output, epoch)
    torch.save(model.state_dict(), model_output)
    return model_output

def mask_prediction(pred, lens):
    bsize, max_len = pred.shape[0], pred.shape[1]
    # print("pred: {}\t{}".format(pred.shape, pred))
    # print("lens: {}\n{}".format(lens.shape, lens))
    # create an index matrix size (b, max_sentences, 1): [[[0 1 2 3 ... num_event], <repeat>]; .to(x) is just to make the output have the same type as x
    idx = torch.arange(max_len).repeat(bsize, 1).to(lens).unsqueeze(-1)
    # .repeat(bsize, max_len, 1).to(pred)
    # print("idx: {}\t{}".format(idx.shape, idx))
    # the mask is a matrix having the same size as x, has value 0 or 1, by comparing the index value with the value of lens
    mask = torch.gt(lens[:, None, None], idx).to(pred)
    # print("mask: {}\t{}".format(mask.shape, mask))
    # items that we don't want to use when computing softmax with have very very low value (negative infinity), so that the softmax of those = 0
    masked_pred = pred * mask
    # print("Lens: {}".format(lens))
    # print("masked pred: {}\t{}".format(masked_pred.shape, masked_pred))
    return masked_pred


# def map_prediction_sentID(pred_scores, sent_ids):
#     scores = {}
#     prev_score = start_score
#     for i, sid in enumerate(sent_ids):
#         score = pred_scores[i]
#         if not train_on_actual_value:
#             score = prev_score + pred_scores[i]
#             prev_score = score
#         scores[sid] = score
#     return scores


def map_prediction_sentID(pred_scores, sent_ids, scores_salient=None):
    scores = {}
    salient = None
    if scores_salient is not None:
        salient = {}
    if scaling:
        pred_scores = rescale_scores(pred_scores)
    prev_score = start_score
    for i, sid in enumerate(sent_ids):
        score = pred_scores[i]
        if not train_on_actual_value:
            score = prev_score + pred_scores[i]
            prev_score = score
        scores[sid] = score
        if scores_salient is not None:
            salient[sid] = scores_salient[i]
    return scores, salient


def get_valid_sentence_index(current_index, scores, sent_list):
    next_index = current_index + 1
    if next_index >= len(sent_list):
        return next_index
    while next_index < len(sent_list) and sent_list[next_index] not in scores:
        next_index += 1
    return next_index


def construct_pred_sequences(all_scores, vid_sentences_mapping, sent_info, gt_sequences, scores_salient=None):
    '''
    Construct sequences from the predicted scores and groundtruth info
    :param scores: {video: [list of scores]}
    :param vid_sentences_mapping:
    :param sent_info:
    :return:
    '''
    pred_sequences = {}
    for vid, sent_list in vid_sentences_mapping.items():
        # print("Generating pred seq for {}\t{}".format(vid, len(sent_list)))
        current_sent_info = sent_info[vid]
        seq = []
        gt_len = len(gt_sequences[vid])
        scores, scores_variances = map_prediction_sentID(all_scores[vid], sent_list, scores_salient[vid] if scores_salient is not None else None)
        current_sent_index = get_valid_sentence_index(-1, scores, sent_list)
        current_sent = sent_list[current_sent_index]
        current_info = current_sent_info[current_sent]  # [stime, etime]
        default_score = start_score

        while len(seq) < gt_len:
            current_time = len(seq) * 0.5
            # print("current_time: {}\t{}".format(current_time, vid))
            if current_info[0] <= current_time <= current_info[1]:
                # if current_sent == "ID165_vid4_4":
                #     print(current_info[0], current_time, current_info[1])
                if current_sent in scores:
                    default_score = scores[current_sent]
                    # if scores_salient is not None:
                        # default_score = np.random.normal(scores[current_sent], scores_variances[current_sent])
                    # print("0")
                    # if scaling:
                    #     default_score = utils.rescale_single_score(default_score)
                seq.append(default_score)
                # print("1")
                continue
            if current_time < current_info[0]:
                seq.append(default_score)
                # print("2")
                continue
            if current_time > current_info[1]:
                current_sent_index = get_valid_sentence_index(current_sent_index, scores, sent_list)
                if current_sent_index >= len(sent_list):
                    seq.append(default_score)
                    # print("3")
                else:
                    current_sent = sent_list[current_sent_index]
                    current_info = current_sent_info[current_sent]
                    # print("4: {}\t{}\t{}".format(current_sent_index, current_sent, len(sent_list)))
        pred_sequences[vid] = seq
    return pred_sequences


def write_seq(seq, ofile):
    with open(ofile, 'w') as writer:
        for i, value in enumerate(seq):
            writer.write("{}\t{}\n".format(i*0.5, value))

def get_att_str(attention):
    tmp = attention.cpu().tolist()
    c = ''
    for i in tmp:
        if i > 0.0:
            c += "{:.2f}, ".format(i)
        else:
            break
    return c.strip()[:-1]

def write_att_details(vid_sents_mapping, sent_content_mapping, event_details, init_predictions, attentions, odir):
    for video, sents in vid_sents_mapping.items():
        ofile = os.path.join(odir, "{}.txt".format(video))
        tmp_details = event_details[video]
        tmp_content = sent_content_mapping[video]
        tmp_preds = init_predictions[video]
        att = attentions[video]
        with open(ofile, 'w', encoding='utf-8') as writer:
            writer.write("@EventID\tStartTime\tEndTime\tAllTokens\tGT\tStdDev\tValidTokens\tAttentions\tPred_Score\n")
            for i, sent in enumerate(sents):
                writer.write("{}\t{}\t{}\t{}\n".format(tmp_details[sent], tmp_content[sent], get_att_str(att[i]), tmp_preds[i].cpu().item()))


def run_test(model_path, scoring_dir=None, train_output_dir=None, dev_output_dir=None,
             test_output_dir=None, att_dir=None, return_scores=False, return_att=False):
    model = load_model(model_path)

    tmp = os.path.basename(model_path)[:-3]

    ccctrain, rhotrain = evaluate(model, trainloader, train_gt, scoring_dir=scoring_dir, output_dir=train_output_dir, return_att=return_att,
                        att_output_dir=att_dir, source="train_{}".format(tmp))
    # ccctrain, rhotrain = 0, 0
    cccdev, rhodev = evaluate(model, devloader, dev_gt, scoring_dir=scoring_dir, output_dir=dev_output_dir, return_att=return_att,
                        att_output_dir=att_dir, source="dev_{}".format(tmp))

    ccctest, rhotest = evaluate(model, testloader, test_gt, scoring_dir=scoring_dir, output_dir=test_output_dir, return_att=return_att,
                        att_output_dir=att_dir, source="test_{}".format(tmp))
    if return_scores:
        return ccctrain, cccdev, ccctest
    else:
        print("Train. rho: {}. ccc: {}. ".format(rhotrain, ccctrain))
        print("Dev. rho: {}. ccc: {}. ".format(rhodev, cccdev))
        print("Test. rho: {}. ccc: {}. ".format(rhotest, ccctest))


def evaluate(model, dataloader, gt_sequences, scoring_dir=None, output_dir=None, return_att=False, att_output_dir=None, source=None):
    model.eval()
    model.train_mode = False
    scores = {}  # {videoid: [score]}
    init_predictions = {}  # {sent_id: [init prediction of each event]}
    attentions = {}  # {sent_id: [attentions]}
    dataset = dataloader.dataset
    for x, y, ids, sentence_lens, words_lens in dataloader:
        x, sentence_lens, words_lens = x.to(device), sentence_lens.to(device), words_lens.to(device)
        with torch.no_grad():
            pred, att = model(x, sentence_lens, words_lens)
            if return_att:
                for i in range(len(ids)):
                    init_predictions[ids[i]] = pred[i]
                    if att is not None:
                        attentions[ids[i]] = att[i]
                # print_tensor("Att", att, True)
                # print("att as dict")
                # print(attentions)
                # input()
            # print_tensor("pred 1", pred, True)
            pred = pred.squeeze(2).cpu()
            # print_tensor("pred 2", pred, True)
        for i in range(len(ids)):
            scores[ids[i]] = pred[i, :].tolist()
    # print(scores.keys())
    # pred_sequences = construct_pred_sequences(scores, dataset.vid_sent_mapping,
    #                                           dataset.sent_info, gt_sequences)
    pred_sequences = construct_pred_sequences(scores, dataset.event_ids,
                                              dataset.event_info, gt_sequences, None)
    tmp_output_file = None
    if source and scoring_dir:
        tmp_output_file = os.path.join(scoring_dir, "{}.txt".format(source))
    ccc, rho = scoring.ccc_set(gt_sequences, pred_sequences, output_file=tmp_output_file)
    if output_dir:
        # write sequences to file
        for vid, seq in pred_sequences.items():
            ofile = os.path.join(output_dir, "{}.txt".format(vid))
            write_seq(seq, ofile)
    if return_att and att_output_dir:
        write_att_details(dataset.vid_sent_mapping, dataset.vid_sent_content, dataset.event_details, init_predictions, attentions, att_output_dir)

    return ccc, rho


def load_groundtruth_file(ifile, rating_column='evaluatorWeightedEstimate'):
    data = pandas.read_csv(ifile)
    return list(data[rating_column])


def get_ids(filename):
    pattern = "ID(\d*)_vid(\d*)"
    result = re.search(pattern, filename)
    if result:
        return result.group(1), result.group(2)
    else:
        return None, None


def load_groundtruth(gt_dir):
    gts = {}
    for fname in os.listdir(gt_dir):
        if fname.startswith("."):
            continue
        vid, sent = get_ids_from_rating(fname)
        if vid is None:
            continue
        gt = load_groundtruth_file(os.path.join(gt_dir, fname))
        gts["ID{}_vid{}.json".format(vid, sent)] = gt
    return gts


def get_setting(model_setting):
    setting = "model: {}\n\n" \
              "args:{}".format(model_setting, args.__dict__)
    return setting


def get_epoch_number(input_name):
    patt = '_(\d+).pt'
    res = re.search(patt, input_name)
    if res:
        return int(res.group(1))
    return -1


def time_since(start):
    s = time.time() - start
    h = math.floor(s / 3600)
    s = s - h * 3600
    m = math.floor(s / 60)
    s = s - m * 60
    if h > 0:
        return "{}h:{}m:{:.0f}s".format(h, m, s)
    return "{}m:{:.0f}s".format(m, s)


def print_time(start):
    print("Time: {}".format(time_since(start)))


def std_single_score(score):
    return (score - 50)/50


def scale_scores(scores):
    tmp = [std_single_score(i) for i in scores]
    return tmp


def rescale_single_score(score):
    return score * 50 + 50


def rescale_scores(scores):
    tmp = [rescale_single_score(i) for i in scores]
    return tmp


def get_ids_from_rating(filename):
    pattern = "results_(\d*)_(\d*)"
    result = re.search(pattern, filename)
    if result:
        return result.group(1), result.group(2)
    else:
        return None, None


def seq_mask(seq_len, max_len):
    """
    # Source: https://www.kaggle.com/robertke94/pytorch-bi-lstm-attention
    Create sequence mask.

    Parameters
    ----------
    seq_len: torch.long, shape [batch_size],
        Lengths of sequences in a batch.
    max_len: int
        The maximum sequence length in a batch.

    Returns
    -------
    mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.
    """
    idx = torch.arange(max_len).to(seq_len).repeat(seq_len.size(0), 1)
    mask = torch.gt(seq_len.unsqueeze(1), idx).to(seq_len)
    return mask


def mask_softmax(matrix, mask=None):
    """Perform softmax on length dimension with masking.
    # Source: https://www.kaggle.com/robertke94/pytorch-bi-lstm-attention
    Parameters
    ----------
    matrix: torch.float, shape [batch_size, .., max_len]
    mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.

    Returns
    -------
    output: torch.float, shape [batch_size, .., max_len]
        Normalized output in length dimension.
    """
    if mask is None:
        result = F.softmax(matrix, dim=-1)
    else:
        mask_norm = ((1 - mask) * NEG_INF).to(matrix)
        for i in range(matrix.dim() - mask_norm.dim()):
            mask_norm = mask_norm.unsqueeze(1)
        result = F.softmax(matrix + mask_norm, dim=-1)
    return result


def mask_softmax_full(matrix, seq_len, max_len):
    mask = seq_mask(seq_len, max_len)
    # print("Mask: {}\t{}".format(mask.shape, mask))
    return mask_softmax(matrix, mask)

global_start = time.time()

'''
Arguments
'''
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mode", help="{train, test}", choices=['train', 'test'], default='test')
ap.add_argument("--model", help="path to the model")
ap.add_argument("--output", help="path to the output dir")
ap.add_argument("--gpu", help="gpu", default=0, choices=[0, 1, 2, 3], type=int)
ap.add_argument('--value', help="actual: predict actual value; diff: predict differences", choices=['actual', 'diff'], default='actual')
ap.add_argument("--bi", help="{no: LSTM; yes: BiLSTM}", default='yes', choices=["no", "yes"])
ap.add_argument("--event_score", help="type of event score to use {mean, last}", default="mean", choices=["mean", "last"])
ap.add_argument("--scaling", help="whether to do scaling", default='yes', choices=['yes', 'no'])
ap.add_argument("--batch", help="batch size", default=117, type=int)
ap.add_argument("--epoch", help="number of epoches", default=100, type=int)
ap.add_argument("--runs", help="number of runs", default=1, type=int)
ap.add_argument("--h1", help="LSTM-1 hidden dim", default=128, type=int)
ap.add_argument("--h2", help="LSTM-2 hidden dim", default=128, type=int)
ap.add_argument("--lr", help="learning rate", default=0.01, type=float)
ap.add_argument("--dropout", help="dropout", default=0.0, type=float)
ap.add_argument('--optimizer', help="Optimizer: SGD or Adam", choices=['sgd', 'adam'])
ap.add_argument("--event_min", help="Event min length", default=2, type=int)
ap.add_argument("--event_max", help="Event max length", default=25, type=int)
ap.add_argument("--event_len", help="number of words in an event (padding)", default=25, type=int)
ap.add_argument('--embeddings', default='data/with_pretrained_embds/bert/words_window-based_5_seconds')
ap.add_argument('--data', default='data/lowercased/EWE_allen_origin_tokens_inclStopwords_withScores/')
ap.add_argument('--splits', default='data/_TrainSetAssignments/splits.json')
ap.add_argument('--suffix', help='suffix to logdir', default='')
ap.add_argument("--attention", help="whether to use attention (If no, the last hidden will be used)", default='yes', choices=['yes', 'no'])
ap.add_argument("--emb_dim", help="Word embedding dim", default=768, type=int)
ap.add_argument("--has_pre_suf", help="whether to trim prefix and suffix. No for GPT2", default='yes', choices=['yes', 'no'])
ap.add_argument('--ratings', required=True, help='path to ratings dir', default='data/ratings/')

args = ap.parse_args()

'''
Parameters
'''
train_mode = False
if args.mode == 'train':
    train_mode = True
print("Mode: ", args.mode)

sgd = False
lr = args.lr
lr_step = 50
lr_gamma = 0.9
num_runs = args.runs
has_pre_suf = True
if args.has_pre_suf == 'no':
    has_pre_suf = False

bidirectional = False
if args.bi == "yes":
    bidirectional = True

hid1_dim = args.h1
hid1_bi = bidirectional
hid1_nlayers = 1

optimizer_opt = args.optimizer

hid2_dim = args.h2
hid2_bi = bidirectional
hid2_nlayers = 1

n_epoch = args.epoch
batch_size = args.batch
dropout_p = args.dropout

gpu = args.gpu
batch_first = True

use_attention = True
if args.attention == 'no':
    use_attention = False

event_score_type = args.event_score
event_num_words = args.event_len
event_words_min = args.event_min
event_words_max = args.event_max

earlyStop_dev = True  # true: early stop based on dev; false: early stop based on test
train_on_actual_value = True
if args.value == 'diff':
    train_on_actual_value = False

if earlyStop_dev:
    earlyStop = "dev"
else:
    earlyStop = "test"

model_save_freq = 30  # save the trained model in every n epochs
input_dim = 300
scaling = False
if args.scaling == 'yes':
    scaling = True

'''
Constants
'''
NEG_INF = -10000
print_interval = 30
start_score = 50
max_num_sentences = 37

'''
Directories
'''
prog_id = time.strftime("%Y%m%d_%H%M%S", time.localtime(global_start))  # program id
if args.mode == 'train':
    if args.suffix != '':
        prog_id += "_{}".format(args.suffix)
    log_dir = "data/log/{}/".format(prog_id)
    print("log_dir: {}".format(log_dir))
    model_odir = os.path.join(log_dir, "models/")
    scoring_dir = os.path.join(log_dir, 'scores')
    setting_file = os.path.join(log_dir, "setting.txt")
else:
    output_root = args.output

score_writer = None

'''
Initialization
'''
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.set_device(gpu)

'''
Data loading
'''
print("Loading data")
start = time.time()
data_dir = args.data

print("Load data")
train_dataset = StoryDataset('train', args.embeddings, data_dir, event_words_min, event_words_max,
                             train_on_actual_value, scaling, start_score, event_num_words, event_score_type,
                             emb_size=args.emb_dim, has_pre_suf=has_pre_suf)
dev_dataset = StoryDataset('valid', args.embeddings, data_dir, event_words_min, event_words_max,
                           train_on_actual_value, scaling, start_score, event_num_words, event_score_type,
                           emb_size=args.emb_dim, has_pre_suf=has_pre_suf)
test_dataset = StoryDataset('test', args.embeddings, data_dir, event_words_min, event_words_max,
                            train_on_actual_value, scaling, start_score, event_num_words, event_score_type,
                            emb_size=args.emb_dim, has_pre_suf=has_pre_suf)

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_padding)
devloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_padding)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_padding)

# load groundtruth
gt_dir = args.ratings  # "data/ratings
rating_fname = 'observer_EWE'
train_gt = load_groundtruth(os.path.join(gt_dir, 'train', rating_fname))
dev_gt = load_groundtruth(os.path.join(gt_dir, 'valid', rating_fname))
test_gt = load_groundtruth(os.path.join(gt_dir, 'test', rating_fname))

ccc_trains = []
ccc_devs = []
ccc_tests = []

if train_mode:
    mkdir(model_odir)
    mkdir(scoring_dir)
    score_writer = open(os.path.join(log_dir, "train_scores.txt"), 'w')
    # score_writer.write("Epoch\tccc_train\tccc_dev\tccc_test\t\tEpoch\trho_train\trho_dev\trho_test\n")

    n_batch = len(trainloader)
    # print("#batches: {}".format(n_batch))
    # write setting
    ccc_trains = []
    ccc_devs = []
    ccc_tests = []
    best_epoches = []
    runs_writer = open(os.path.join(log_dir, "runs_scores.txt"), 'w')
    runs_writer.write("@Run\tBestEpoch\tTrain(Best)\tDev\tTest\n")
    for run in range(num_runs):
        print("=================================")
        print("| Run #{}".format(run))
        print("=================================")
        score_writer.write("\n\n--------------------\nRUN {}\n\n".format(run))
        score_writer.write("Epoch\tccc_train\tccc_dev\tccc_test\t\tEpoch\trho_train\trho_dev\trho_test\n")
        last_output, best_output, best_epoch = train(run)
        ccc_train, ccc_dev, ccc_test = run_test(best_output, return_scores=True)
        ccc_trains.append(ccc_train)
        ccc_devs.append(ccc_dev)
        ccc_tests.append(ccc_test)
        best_epoches.append(best_epoch)
        print("Best ({})".format(best_output))
        print("ccc_train: {:.4f}\tccc_dev: {:.4f}\tccc_test: {:.4f}".format(ccc_train, ccc_dev, ccc_test))
        runs_writer.write("{}\t{}\t{}\t{}\t{}\n".format(run, best_epoch, ccc_train, ccc_dev, ccc_test))

    score_writer.flush()
    score_writer.close()
    runs_writer.flush()
    runs_writer.close()
    print("\n============================================================")
    print("Average for BEST")
    print("train: {:.4f}; dev: {:.4f}; test: {:.4f}\n".format(np.mean(ccc_trains), np.mean(ccc_devs), np.mean(ccc_tests)))
    print("StdDev")
    print("train: {:.4f}; dev: {:.4f}; test: {:.4f}\n".format(np.std(ccc_trains), np.std(ccc_devs), np.std(ccc_tests)))
    print("Average best epoch: {:.0f}".format(np.mean(best_epoches)))
    print("Total time: {}".format(time_since(global_start)))
else:
    # output_root = os.path.join(output_root, args.name)
    model_path = args.model
    if not os.path.isfile(model_path):
        print("Model not found ", model_path)
        sys.exit(0)
    mkdir(output_root)
    att_output_dir = os.path.join(output_root, "attention_output/")
    seq_odir = os.path.join(output_root, "pred_seqs/")
    train_output_dir = os.path.join(seq_odir, "train/")
    dev_output_dir = os.path.join(seq_odir, "dev/")
    test_output_dir = os.path.join(seq_odir, "test/")
    scoring_dir = os.path.join(output_root, "scores/")

    mkdir(att_output_dir)
    mkdir(seq_odir)
    mkdir(train_output_dir)
    mkdir(dev_output_dir)
    mkdir(test_output_dir)
    mkdir(scoring_dir)

    print("Test model {}".format(model_path))
    run_test(model_path, scoring_dir=scoring_dir, train_output_dir=train_output_dir, dev_output_dir=dev_output_dir,
             test_output_dir=test_output_dir, return_att=True, att_dir=att_output_dir, return_scores=False)
    print("Output is at", output_root)
    print("Total time: {}".format(time_since(global_start)))


