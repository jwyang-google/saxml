from typing import List, Literal, Optional, Tuple, TypedDict

from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np
import random


def print_stats(np_array):
  min = np.percentile(np_array, 0)
  percentile10 = np.percentile(np_array, 10)
  percentile25 = np.percentile(np_array, 25)
  percentile50 = np.percentile(np_array, 50)
  percentile75 = np.percentile(np_array, 75)
  percentile90 = np.percentile(np_array, 90)
  max = np.percentile(np_array, 100)
  avg = np.average(np_array)
  print("min: {:.3f}, percentile 10: {:.3f}, percentile 25: {:.3f}, percentile 50: {:.3f}, percentile 75: {:.3f}, percentile 90: {:.3f}, max: {:.3f}, avg: {:.3f}".format(
    min, percentile10, percentile25, percentile50, percentile75, percentile90, max, avg))


def eval_rouge(gt_outputs, predicted_outputs):
  def print_results():
    print("r1_precision: ")
    print_stats(rouge_scores['r1_prec'])

    print("r1_recall: ")
    print_stats(rouge_scores['r1_recall'])

    print("r1_fmeasure: ")
    print_stats(rouge_scores["r1_fm"])

    print("r2_precision: ")
    print_stats(rouge_scores['r2_prec'])

    print("r2_recall: ")
    print_stats(rouge_scores['r2_recall'])

    print("r2_fmeasure: ")
    print_stats(rouge_scores["r2_fm"])

    print("rl_precision: ")
    print_stats(rouge_scores['rl_prec'])

    print("rl_recall: ")
    print_stats(rouge_scores['rl_recall'])

    print("r1_fmeasure: ")
    print_stats(rouge_scores["rl_fm"])


    print("rl_sum_precision: ")
    print_stats(rouge_scores['rl_sum_prec'])

    print("rl_sum_recall: ")
    print_stats(rouge_scores['rl_sum_recall'])

    print("r1_sum_fmeasure: ")
    print_stats(rouge_scores["rl_sum_fm"])

    print_stats(num_gt_output_tokens)
    print_stats(num_pred_output_tokens)

  scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', "rougeLsum"], use_stemmer=True)
  rouge_scores = {"r1_prec": [], "r1_recall": [], "r1_fm": [],
                  "r2_prec": [], "r2_recall": [], "r2_fm": [],
                  "rl_prec": [], "rl_recall": [], "rl_fm": [],
                  "rl_sum_prec": [], "rl_sum_recall": [],"rl_sum_fm": []}
  num_gt_output_tokens = []
  num_pred_output_tokens = []
  truncation = 0
  for i, (predicted_output, gt_output) in tqdm(enumerate(zip(predicted_outputs, gt_outputs))):
    if truncation > 0 and len(predicted_output.split(" ")) > truncation:
      predicted_output = " ".join(predicted_output.split(" ")[:truncation])
    scores = scorer.score(gt_output, predicted_output)
    rouge_scores['r1_prec'].append(scores['rouge1'].precision)
    rouge_scores['r1_recall'].append(scores['rouge1'].recall)
    rouge_scores['r1_fm'].append(scores['rouge1'].fmeasure)

    rouge_scores['r2_prec'].append(scores['rouge2'].precision)
    rouge_scores['r2_recall'].append(scores['rouge2'].recall)
    rouge_scores['r2_fm'].append(scores['rouge2'].fmeasure)

    rouge_scores['rl_prec'].append(scores['rougeL'].precision)
    rouge_scores['rl_recall'].append(scores['rougeL'].recall)
    rouge_scores['rl_fm'].append(scores['rougeL'].fmeasure)

    rouge_scores['rl_sum_prec'].append(scores['rougeLsum'].precision)
    rouge_scores['rl_sum_recall'].append(scores['rougeLsum'].recall)
    rouge_scores['rl_sum_fm'].append(scores['rougeLsum'].fmeasure)

    num_gt_output_tokens.append(len(gt_output.split(" ")))
    num_pred_output_tokens.append(len(predicted_output.split(" ")))

  print_results()


def load_predicted_output(data_pkl_dir):
  import pickle
  import pandas
  import glob

  data_pkls = glob.glob(data_pkl_dir + "/*.pkl")

  predicted_outputs = []
  gt_outputs = []
  latency = []

  for data_pkl in tqdm(data_pkls):
    samples = pickle.load(open(data_pkl, "rb"))
    predicted_outputs.extend(samples['llama2_chat_outputs'])
    gt_outputs.extend(samples['output'])
    latency.extend(samples['latency'])

  print(len(predicted_outputs))
  return gt_outputs, predicted_outputs


def draw_results(num_gt_output_tokens, num_pred_output_tokens):
  import matplotlib.pyplot as plt
  plt.style.use('seaborn-deep')
  bins = np.linspace(0, 2000, 25)
  plt.hist([num_gt_output_tokens, num_pred_output_tokens], bins, label=['GT', 'Prediction'])
  plt.legend(loc='upper right')
  plt.savefig("num_tokens_comp.png")


def draw_distribution(num_tokens, plt_name):
  import matplotlib.pyplot as plt
  plt.style.use('seaborn-deep')
  bins = np.linspace(0, 2048, 8)
  plt.hist(num_tokens, bins, label=['length'])
  plt.legend(loc='upper right')
  plt.savefig("{}.png".format(plt_name))


def input_output_len_analysis(predicted_outputs, batch_size=36):
  num_output_tokens = []
  for predicted_output in predicted_outputs:
    num_output_tokens.append(len(predicted_output.split(" ")))

  num_batches = int(len(num_output_tokens) / batch_size)
  batched_outputs = []
  for i in range(num_batches):
    batched_outputs.append(num_output_tokens[i*batch_size: (i+1)*batch_size])

  batch_data = []
  for batched_output in batched_outputs:
    median_len = np.median(batched_output)
    max_len = np.max(batched_output)
    min_len = np.min(batched_output)
    max_min_ratio = max_len/min_len
    max_avg_ratio = max_len/median_len

    waiting_ratios = max_len / np.array(batched_output)
    avg_waiting_ratio = np.sum(waiting_ratios)/batch_size
    print("cur_median: {:.3f}, cur_min: {:.3f}, cur_max: {:.3f}, avg_waiting_ratio: {:.3f}".format(median_len, min_len, max_len, avg_waiting_ratio))
    batch_data.append([min_len, median_len, max_len, max_min_ratio, max_avg_ratio, avg_waiting_ratio])

  batch_data = np.array(batch_data)
  print("min_len: {:.3f}, median_len: {:.3f}, max_len: {:.3f}, max_min_ratio: {:.3f}, max_median_ratio: {:.3f}, avg_waiting_ratio: {}".format(
    np.average(batch_data[:, 0]), 
    np.average(batch_data[:, 1]), 
    np.average(batch_data[:, 2]), 
    np.average(batch_data[:, 3]), 
    np.average(batch_data[:, 4]),
    np.average(batch_data[:, 5])))

  # draw_distribution(num_output_tokens, "output_len_distribution")


gt_outputs, predicted_outputs = load_predicted_output("/home/jwyang/llama/greedy/batch32_input2048_output2048/")
# eval_rouge(gt_outputs, predicted_outputs)
input_output_len_analysis(predicted_outputs)

