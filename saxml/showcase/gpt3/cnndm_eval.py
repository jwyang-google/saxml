import tensorflow as tf
import os, sys
import numpy as np
import json
import threading
from rouge_score import rouge_scorer

sys.path.append('/home/jwyang/saxml/bazel-bin/saxml/client/python/')
import sax

 
N = None
 
original_filenames = [
  "gs://test-example-123/datasets/cnn_dailymail/3.4.0/cnn_dailymail-test.tfrecord-00000-of-00001"
]
 
def get_dataset(file_name):
  dataset = tf.data.TFRecordDataset(file_name)
  count = sum([1 for _ in dataset])
  return dataset, count


def lm_predict(prompt):
  # print("*****************")
  # print("Generate result for article prompt ---- {}".format(prompt))
  res = lm_model.Generate(prompt)
  return res


def register_sax_model(model_id):
  model = sax.Model(model_id)
  global lm_model
  lm_model = model.LM()


def compute_rogue(targets, predictions):
  assert len(targets) == len(predictions)
  scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
  print("Compute rouge for {} samples".format(len(targets)))

  r1, r2, rl, rlsum = [], [], [], []
  for target, prediction in zip(targets, predictions):
    scores = scorer.score(target, prediction)
    r1.append(scores['rouge1'])
    r2.append(scores['rouge2'])
    rl.append(scores['rougeL'])
    rlsum.append(scores['rougeLsum'])
  r1_mean = np.mean(r1)
  r2_mean = np.mean(r2)
  rl_mean = np.mean(rl)
  rlsum_mean = np.mean(rlsum)
  return r1_mean, r2_mean, rl_mean, rlsum_mean


def dump_output(res, targets, predictions, output_file):
    output = {}
    # compute rogue scores
    r1, r2, r, rlsum = compute_rogue(targets, predictions)
    output['rouge1'] = r1
    output['rouge2'] = r2
    output['rouge'] = r
    output['rougeLsum'] = rlsum
    output['num_samples'] = len(targets)

    # dump output
    output['predictions'] = res
    json.dump(output, open(output_file, 'w'), indent=4)


def process_data(batch_idx, dataset):
    targets = []
    predictions = []
    all_res = []
    for i in range(len(dataset)):
      original_record = dataset[i]
      if i > 0 and i % 5 == 0:
        output_file = os.path.join("cnndm_inference", "inference_batch_{}_step_{}.json".format(batch_idx, i))
        print("Dump output at step {}, saving to file {}".format(i, output_file))
        dump_output(all_res, targets=targets, predictions=predictions, output_file=output_file)

      # print("-----------------------------------")
      print("Processing batch {}: {}th example".format(batch_idx, i))

      original_example_proto = tf.train.Example()
      original_example_proto.ParseFromString(original_record)

      res = {}
      for key, feature in original_example_proto.features.feature.items():
        if key not in ["article", "highlights"]:
          continue
        kind = feature.WhichOneof('kind')
        feature_value = getattr(feature, kind).value[0]
        if key == "article":
          strs_to_join = ['summarize:', feature_value.decode("utf-8")]
          prompt = " ".join(strs_to_join)
          res['article'] = prompt

          predicted = lm_predict(prompt)
          # print("*****************")
          # print("Predicted summary ---- {}".format(predicted))
          predictions.append(predicted[-1][0])
          res['prediction'] = predicted

        if key == "highlights":
          # print("*****************")
          # print("Label summary ---- {}".format(feature_value.decode("utf-8")))
          targets.append(feature_value.decode("utf-8"))
          res['label'] = feature_value.decode("utf-8")
      all_res.append(res)

    final_output = os.path.join("cnndm_inference", "inference_batch_{}.json".format(batch_idx))
    dump_output(all_res, targets, predictions, final_output)
 

def main():
  register_sax_model("/sax/test/gpt175b")

  count = 0
  for original_filename in original_filenames:
    count += 1
    original_dataset, num_original_examples = get_dataset(original_filename)
    print("Processing {} examples in the file: {}".format(num_original_examples, original_filename))
    original_dataset = original_dataset if not N else original_dataset.take(N)

    # Multi-thread
    num_threads = 64
    per_batch_samples = int(num_original_examples/num_threads) + 1
    batch_datasets = list(original_dataset.batch(per_batch_samples).as_numpy_iterator())
    threads = []
    for i in range(len(batch_datasets)):
      t = threading.Thread(target=process_data, args=(i, batch_datasets[i]))
      t.start()
      threads.append(t)

    print("Wating for threads to join...")
    for t in threads:
      t.join()

    # # Single-thread
    # targets = []
    # predictions = []
    # all_res = []
    # for i, original_record in enumerate(original_dataset):
    #   if i > 5:
    #     break

    #   if i % 10 == 0:
    #     output_file = os.path.join("cnndm_inference", "inference_step_{}.json".format(i))
    #     print("Dump output at step {}, saving to file {}".format(i, output_file))
    #     dump_output(all_res, targets=targets, predictions=predictions, output_file=output_file)

    #   print("-----------------------------------")
    #   logging.info("Processing {}th example".format(i))

    #   original_example_proto = tf.train.Example()
    #   original_example_proto.ParseFromString(original_record.numpy())

    #   res = {}
    #   for key, feature in original_example_proto.features.feature.items():
    #     if key not in ["article", "highlights"]:
    #       continue
    #     kind = feature.WhichOneof('kind')
    #     feature_value = getattr(feature, kind).value[0]
    #     if key == "article":
    #       strs_to_join = ['summarize:', feature_value.decode("utf-8")]
    #       prompt = " ".join(strs_to_join)
    #       res['article'] = prompt

    #       predicted = lm_predict(prompt)
    #       print("*****************")
    #       print("Predicted summary ---- {}".format(predicted))
    #       predictions.append(predicted[0][0])
    #       res['prediction'] = predicted

    #     if key == "highlights":
    #       print("*****************")
    #       print("Label summary ---- {}".format(feature_value.decode("utf-8")))
    #       targets.append(feature_value.decode("utf-8"))
    #       res['label'] = feature_value.decode("utf-8")
    #   all_res.append(res)

    # final_output = os.path.join("cnndm_inference", "inference_final.json")
    # dump_output(all_res, targets, predictions, final_output)
    
if __name__ == '__main__':
  main()