# ccc from OMG Empathy

from __future__ import print_function
import argparse
import os

from scipy.stats import pearsonr
import numpy
import pandas


def ccc(y_true, y_pred, verbose=False):
    true_mean = numpy.mean(y_true)
    true_variance = numpy.var(y_true)

    pred_mean = numpy.mean(y_pred)
    pred_variance = numpy.var(y_pred)

    std_predictions = numpy.std(y_pred)
    std_gt = numpy.std(y_true)

    rho,_ = pearsonr(y_pred,y_true)
    if verbose:
        print("True mean: {}".format(true_mean))
        print("true_variance: {}".format(true_variance))
        print("pred_mean: {}".format(pred_mean))
        print("pred_variance: {}".format(pred_variance))
        print("std_predictions: {}".format(std_predictions))
        print("std_gt: {}".format(std_gt))
        print("rho: {}".format(rho))

    ccc_score = 2 * rho * std_gt * std_predictions / (
        std_predictions ** 2 + std_gt ** 2 +
        (pred_mean - true_mean) ** 2)

    return ccc_score, rho


def ccc_set(y_trues, y_preds, output_file=None, allow_missing_video=False):
    '''
    :param y_trues: dictionary
    :param y_preds:
    :return:
    '''
    # print(y_trues)
    # print(y_preds)
    if len(y_trues) != len(y_preds):
        if not allow_missing_video:
            print("#videos not matched!\tTrue: {}\tPred: {}".format(len(y_trues), len(y_preds)))
            return -1, -1
    ccc_scores = []
    rho_scores = []

    writer = None
    if output_file:
        writer = open(output_file, 'w')
        writer.write("vID\trho\tccc\n")

    # for vid, y_true in y_trues.items():
    #     if vid not in y_preds:
    #         print("Cannot find {} in y_preds".format(vid))
    #         return -1, -1
    #     ccc_score, rho = ccc(y_true, y_preds[vid])
    #     ccc_scores.append(ccc_score)
    #     rho_scores.append(rho)
    #     if writer:
    #         writer.write("{}\t{}\t{}\n".format(vid, rho, ccc_score))
        # print(ccc_score, "\t", rho)
    for vid, y_pred in y_preds.items():
        if vid not in y_trues:
            print("Cannot find {} in y_trues".format(vid))
            return -1, -1
        y_true = y_trues[vid]
        ccc_score, rho = ccc(y_true, y_pred)
        ccc_scores.append(ccc_score)
        rho_scores.append(rho)
        if writer:
            writer.write("{}\t{}\t{}\n".format(vid, rho, ccc_score))
    mean_ccc, mean_rho = numpy.mean(ccc_scores), numpy.mean(rho_scores)
    if writer:
        writer.write("\nAverage\t{}\t{}".format(mean_rho, mean_ccc))
        writer.flush()
        writer.close()

    return mean_ccc, mean_rho

