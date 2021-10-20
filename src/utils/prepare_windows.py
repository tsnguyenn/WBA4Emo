'''
change the window-based files to have the same format as the event-based's
'''


import os
import json
import re
import pandas
from tqdm import tqdm
import argparse


def mkdir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)


def get_ratings_for_time_range(time_ratings, start_time, end_time, start_index=0, first_line=False):
    ratings = []
    timings = []
    if start_index < 0:
        start_index = 0
    max_index = start_index
    if end_time - start_time < 0.5:
        start_time = int(start_time)
    # print("! Start from {}, starting time: {}".format(start_index, time_ratings[start_index][0]))
    for i in range(start_index, len(time_ratings)):
        tr = time_ratings[i]
        if tr[0] >= start_time or first_line:
            if tr[0] <= end_time:
                timings.append(tr[0])
                ratings.append(tr[1])
                max_index = i
            else:
                break
    return timings, ratings, max_index


def get_tokens(word, to_lower=False):
    # process for the words like "gonna"
    word = word.strip()
    if word == 'yknow':
        return ["y'know"]
    if word == "'til":
        return ["'", "til"]
    if word == "gonna":
        return ["gon", "na"]
    if word == "gotta":
        return ["got", "ta"]
    if word == "cannot":
        return ["can", "not"]
    special_tokens = ["n't", "...", ",", ".", "'", "-", "?", "!", '"', ":", ";"]
    if word in special_tokens:
        if to_lower:
            return [word.strip().lower()]
        else:
            return [word.strip()]
    for special_token in special_tokens:
        pos = get_position(word, special_token)
        if pos == 0 and special_token in ["-", '"']:
            return [special_token] + get_tokens(word[len(special_token):])
        if pos > 0:
            return get_tokens(word[:pos]) + get_tokens(word[pos:])
    if to_lower:
        return [word.strip().lower()]
    else:
        return [word.strip()]


def get_position(word, special_token):
    if word.__contains__(special_token):
        return word.index(special_token)
    return -1


def load_scores(csv_rating_file, time_col='time', rating_col=' rating'):
    '''
    :param csv_rating_file:
    :return: [(time, rating)] list of (time, rating) tuples
    '''
    ratings = []
    data = pandas.read_csv(csv_rating_file)
    for i, row in data.iterrows():
        ratings.append((float(row[time_col]), float(row[rating_col])))
    return ratings


def get_window(words, start_time, end_time, start_index, first_line, time_scores, window_id):
    if words is None or len(words) == 0:
        return None, start_index
    timings, scores, start_index = get_ratings_for_time_range(time_scores, start_time, end_time, start_index, first_line)
    window = {'event_start': start_time, 'event_end': end_time, 'tokens': words, 'scores': scores, 'event_id': window_id}
    return window, start_index


def get_ids(file_name):
    pattern = 'ID(\d+)_vid(\d+)'
    results = re.search(pattern, file_name)
    if results:
        return results.group(1), results.group(2)
    return None, None


def group_words_by_time_window_for_file(time_window, ifile, ofile, score_dir, to_lower=True, rating='evaluatorWeightedEstimate'):
    windows = []
    start_index = 0
    first_line = True
    id1, id2 = get_ids(os.path.basename(ifile))
    time_scores = load_scores(os.path.join(score_dir, "results_{}_{}.csv".format(id1, id2)), rating_col=rating)  # load annotations
    reader = open(ifile, 'r')
    sent_start = 0
    sent_end = sent_start + time_window
    current_string = ""
    prev_end = 0
    words = []
    for line in reader.readlines()[1:]:
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue
        start = float(parts[0])
        end = float(parts[1])
        tokens = get_tokens(parts[2], to_lower=to_lower)
        if start > sent_end:
            window, start_index = get_window(words, sent_start, sent_end-0.001, start_index, first_line, time_scores, len(windows))
            if window is not None:
                windows.append(window)
            first_line = False
            words = []
            sent_start = sent_end
            sent_end = sent_start + time_window
        words += tokens
    if len(words) > 0:
        window, start_index = get_window(words, sent_start, sent_end - 0.001, start_index, first_line, time_scores, len(windows))
        if window is not None:
            windows.append(window)
    reader.close()
    json.dump(windows, open(ofile, 'w'), indent=2)

def group_words_by_time_window_for_dir(time_window, input_dir, output_dir, score_dir, to_lower=True, rating='evaluatorWeightedEstimate'):
    mkdir(output_dir)
    for fname in tqdm(os.listdir(input_dir)):
        output_file = os.path.join(output_dir, "{}.json".format(fname[:-12]))
        ifile = os.path.join(input_dir, fname)
        group_words_by_time_window_for_file(time_window, ifile, output_file, score_dir=score_dir, to_lower=to_lower, rating=rating)


def is_valid(idir, score_dir):
    if not os.path.isdir(idir):
        print("Cannot find ", idir)
        return False
    if not os.path.isdir(score_dir):
        print("Cannot find ", score_dir)
        return False
    return True


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='path to features dir')
    ap.add_argument('--ratings', required=True, help='path to ratings dir')
    ap.add_argument('--output', default='data/preprocessed/window-based_5s/', help='path to output dir')
    ap.add_argument('--window', default=5, help='window-time length in seconds', type=int)
    ap.add_argument('--key', default='evaluatorWeightedEstimate', help='key of ratings')

    args = ap.parse_args()
    input_path = args.input  # e.g., SENDv1_featuresRatings_pw/features/
    gt_dir = args.ratings  # e.g., SENDv1_featuresRatings_pw/ratings
    output_path = args.output  # e.g., preprocessed_5s/
    time_window = args.window  # e.g., 5
    rating_key = args.key

    sets = ['test', 'train', 'valid']
    for setname in sets:
        idir = os.path.join(input_path, setname, 'linguistic')
        score_dir = os.path.join(gt_dir, setname, 'observer_EWE')
        out_dir = os.path.join(output_path, setname)
        if not is_valid(idir, score_dir):
            continue
        print("Input: {}\nScore dir: {}\nOutput dir: {}".format(idir, score_dir, out_dir))
        group_words_by_time_window_for_dir(time_window, input_dir=idir, output_dir=out_dir, score_dir=score_dir, to_lower=True, rating=rating_key)
