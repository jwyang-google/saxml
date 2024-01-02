import os, sys
import threading
import time
import numpy as np

sys.path.append('/home/jwyang/saxml/bazel-bin/saxml/client/python/')
import sax

 
def load_prompts(pkl_file, num_samples):
  import pandas
  samples = pandas.read_pickle(pkl_file)
  prompts = []
  for index, row in samples.iterrows():
    prompts.append(row['input'])
  samples = samples[:num_samples]
  prompts = prompts[:num_samples]
  assert len(prompts) == samples.shape[0]
  return samples, prompts


def lm_predict(prompt):
  res = lm_model.Generate(prompt)
  return res


def register_sax_model(model_id):
  model = sax.Model(model_id)
  global lm_model
  lm_model = model.LM()


def process_data(thread_idx, samples, prompts, result_dir, batch_size, seq_len, output_len, results):
    latency = []
    predictions = []
    num_tokens = []
    for i in range(len(prompts)):
      # print("Processing batch {}: {}th example".format(thread_idx, i))
      cur_prompt = prompts[i]
      start_time = time.time()
      predicted = lm_predict(cur_prompt)
      end_time = time.time()
      
      # latency.append(end_time - start_time)
      # predictions.append(predicted[-1][0])
      cur_num_tokens = len(predicted[-1][0].split())
      num_tokens.append(cur_num_tokens)
      print("Generated thread {}: {} prediction, {} tokens, in {}s".format(thread_idx, i, cur_num_tokens, (end_time-start_time)))

    #   if i % 16 == 0:
    #     import pickle
    #     with open(result_dir + "/batch{}_input{}_output{}_thread{}_iter{}.pkl".format(
    #         batch_size, seq_len, output_len, thread_idx, i), "wb") as file:
    #         outputs = {'results': predictions, 'latency': latency, "num_tokens": num_tokens}
    #         pickle.dump(outputs, file)

    # print("Generated result length: {}".format(len(predictions)))
    # samples['llama2_chat_outputs'] = predictions
    # samples['latency'] = latency
    # samples['num_tokens'] = num_tokens
    # samples.to_pickle(result_dir + "/batch{}_input{}_output{}_thread{}.pkl".format(
    #   batch_size, seq_len, output_len, thread_idx))

    results[thread_idx] = num_tokens


def main(args):
  register_sax_model(args.model_id)
  samples, prompts = load_prompts(args.data_pkl, args.num_samples)
  # import random, string
  # prompt = " ".join(random.choices(string.ascii_uppercase, k=args.input_seq_len))
  # prompts = [prompt] * args.num_samples

  # Multi-thread
  num_threads = args.batch_size * 4
  per_thread_samples = int(len(prompts)/num_threads)

  batch_samples = []
  batch_prompts = []
  for i in range(num_threads):
    cur_thread_prompts = prompts[i*per_thread_samples: min(len(prompts), (i+1)*per_thread_samples)]
    batch_prompts.append(cur_thread_prompts)
    cur_thread_samples = samples[i*per_thread_samples: min(len(prompts), (i+1)*per_thread_samples)]
    batch_samples.append(cur_thread_samples)

  threads = []
  results = [None] * num_threads
  start_time = time.time()
  for i in range(num_threads):
    t = threading.Thread(target=process_data, 
                         args=(i, 
                               batch_samples[i], 
                              # batch_samples,
                               batch_prompts[i], 
                               args.results_dir, 
                               args.batch_size,
                               args.input_seq_len,
                               args.max_decode_steps,
                               results))
    t.start()
    threads.append(t)

  print("Wating for threads to join...")
  for t in threads:
    t.join()
  end_time = time.time()

  # compute metrics
  total_tokens = 0
  for result in results:
    total_tokens += np.sum(result)

  num_samples = int(num_threads * per_thread_samples)
  num_batches = int(num_samples / args.batch_size)
  total_latency = end_time - start_time
  print("################### METRICS ##################")
  print("processing {} requests in {}s, query throughput: {}".format(num_samples, total_latency, num_samples/total_latency))
  print("processing {} tokens in {}s, token throughput: {}".format(total_tokens, total_latency, total_tokens/total_latency))
  print("processing {} requests in {}s, average query latency: {}".format(num_samples, total_latency, total_latency/num_samples))


import argparse
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_id', type=str, required=True)
  parser.add_argument('--data_pkl', type=str, required=True)
  parser.add_argument('--num_samples', type=int, required=True)
  parser.add_argument('--results_dir', type=str, required=True)
  parser.add_argument('--batch_size', type=int, required=True)
  parser.add_argument('--input_seq_len', type=int, required=True)
  parser.add_argument('--max_decode_steps', type=int, required=True)
  args = parser.parse_args()

  main(args)