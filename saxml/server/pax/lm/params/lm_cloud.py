# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Serving model parameters for lm_cloud."""

import os
from typing import List, cast

from jax import numpy as jnp
from paxml import base_experiment
from paxml import tasks_lib
from paxml.tasks.lm.params import lm_cloud
from praxis import base_input
from praxis import layers
from praxis import optimizers
from praxis import pax_fiddle
from praxis import schedules
from praxis.layers import activations
from praxis.layers import multi_query_attention
from saxml.server import servable_model_registry
from saxml.server.pax import quantization
# from saxml.server.pax.lm.layers import LLaMARotaryEmbedding
# from saxml.server.pax.lm.layers import ParallelTransformer
from saxml.server.pax.lm.params import template

from saxml.server.pax.lm import layers as sax_layers
LLaMARotaryEmbedding = sax_layers.LLaMARotaryEmbedding
ParallelTransformer = sax_layers.ParallelTransformer


@template.make_servable()
class BaseLLaMA(base_experiment.BaseExperiment):
  """Base LLaMA Transformer LM configuration."""

  SPM_MODEL = 'gs://cloud-tpu-inference-public/sax-tokenizers/llama/llama-tokenizer.model'
  SOS_ID = 1
  EOS_ID = 2

  # architecture related
  NUM_LAYERS = 32
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = 32
  MODEL_DIMS = 4096
  HIDDEN_DIMS = MODEL_DIMS * 4
  FPROP_DTYPE = jnp.bfloat16
  MODEL_DTYPE = jnp.bfloat16
  USE_MQA = False

  ACTIVATION_CLS = activations.SiLU
  USE_GATED_ACTIVATION = True
  RMS_NORM_EPSILON = 1.0e-05

  # Sub-class has to specify a mesh.
  ICI_MESH_SHAPE = [1, 1, 1]
  DCN_MESH_SHAPE = None
  DECODE_MESH_TRANSPOSE = None

  BATCH_SIZE = 1
  NUM_SAMPLES = 1
  ENABLE_GENERATE_STREAM = True
  STREAM_INTERVAL_STEPS = 16
  FPROP_FOR_PREFIX = True
  INPUT_SEQ_LEN = 4096
  BUCKET_KEYS = [128, 1024, 4096]
  MAX_DECODE_STEPS = [128, 512, 1024]
  EXTRA_INPUTS = {
      'temperature': 0.5,
      'per_example_max_decode_steps': 128,
      'per_example_top_k': 200,
      'per_example_top_p': 0.95,
  }

  def model(self):
    model_p = pax_fiddle.Config(layers.LanguageModel, name='xformer_lm')
    model_p.lm_tpl.packed_input = False
    model_p.lm_tpl.model_dims = self.MODEL_DIMS
    model_p.lm_tpl.vocab_size = self.VOCAB_SIZE
    model_p.lm_tpl.position_emb_tpl = None
    model_p.lm_tpl.softmax_tpl = pax_fiddle.Config(
        layers.FullSoftmax,
        name='output',
        input_dims=self.MODEL_DIMS,
        num_classes=self.VOCAB_SIZE,
    )
    model_p.lm_tpl.softmax_tpl.feed_forward_tpl.has_bias = False
    model_p.lm_tpl.separate_embedding_tpl = pax_fiddle.Config(
        layers.Embedding,
        name='tok_embeddings',
        input_dims=self.MODEL_DIMS,
        num_classes=self.VOCAB_SIZE,
    )
    ln_tpl = pax_fiddle.Config(
        layers.RmsNorm,
        name='norm',
        direct_scale=True,
        epsilon=self.RMS_NORM_EPSILON,
    )
    model_p.lm_tpl.final_ln_tpl = ln_tpl.clone()

    stacked_transformer_tpl = pax_fiddle.Config(layers.StackedTransformer)
    stacked_transformer_tpl.model_dims = self.MODEL_DIMS
    stacked_transformer_tpl.hidden_dims = self.HIDDEN_DIMS
    stacked_transformer_tpl.num_layers = self.NUM_LAYERS
    stacked_transformer_tpl.num_heads = self.NUM_HEADS
    stacked_transformer_tpl.dim_per_head = self.DIMS_PER_HEAD

    transformer_layer_p = cast(
        pax_fiddle.Config[layers.Transformer],
        stacked_transformer_tpl.transformer_layer_params_tpl,
    )
    transformer_layer_p.norm_policy = 'pre'
    transformer_layer_p.ln_tpl = ln_tpl.clone()

    if self.USE_MQA:    
      transformer_layer_p.tr_atten_tpl = pax_fiddle.Config(
          multi_query_attention.MultiQueryDotProductAttention,
          num_kv_heads=self.NUM_KV_HEADS,
      )
      transformer_layer_p.tr_atten_tpl.combine_qkv = False
    else:
      transformer_layer_p.tr_atten_tpl.internal_enable_per_dim_scale = False
      transformer_layer_p.tr_atten_tpl.combine_qkv = True
    
    transformer_layer_p.tr_atten_tpl.internal_enable_query_scale = True
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_atten_tpl.rotary_position_emb_tpl = (
        pax_fiddle.Config(LLaMARotaryEmbedding)
    )
    transformer_layer_p.tr_atten_tpl.use_rotary_position_emb = True

    transformer_layer_p.tr_fflayer_tpl.has_bias = False
    transformer_layer_p.tr_fflayer_tpl.ln_tpl = ln_tpl.clone()
    transformer_layer_p.tr_fflayer_tpl.activation_tpl = pax_fiddle.Config(
        self.ACTIVATION_CLS
    )
    transformer_layer_p.tr_fflayer_tpl.use_gated_activation = (
        self.USE_GATED_ACTIVATION
    )

    model_p.lm_tpl.stacked_transformer_tpl = stacked_transformer_tpl

    model_p.fprop_dtype = self.FPROP_DTYPE
    model_p.dtype = self.MODEL_DTYPE
    return model_p

  def datasets(self) -> List[pax_fiddle.Config[base_input.BaseInput]]:
    return []

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='xformer_task')
    model_p = self.model()
    task_p.model = model_p

    # Set sharding
    task_p = template.set_decoding_sharding_hparams(
        task_p,
        mesh_shape=self.ICI_MESH_SHAPE,
        decode_mesh_transpose=self.DECODE_MESH_TRANSPOSE,
    )
    # Unused.
    lp = task_p.train.learner
    lp.loss_name = 'total_loss'
    lp.optimizer = pax_fiddle.Config(
        optimizers.ShardedSgd,
        learning_rate=1e-3,
        lr_schedule=pax_fiddle.Config(schedules.Constant)
    )
    return task_p


# llama-7b models
# @quantization.for_transformer(quantize_on_the_fly=True)
@servable_model_registry.register
class LLaMA7BFP16TPUv5e1(BaseLLaMA):
  NUM_LAYERS = 32
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = 32
  MODEL_DIMS = 4096
  HIDDEN_DIMS = 11008

  ICI_MESH_SHAPE = [1, 1, 1]

  SPM_MODEL = 'gs://cloud-tpu-inference-public/sax-tokenizers/llama/llama-tokenizer.model'
  INPUT_SEQ_LEN = 128
  MAX_DECODE_STEPS = 32
  ENABLE_GENERATE_STREAM = False

  @property
  def test_mode(self) -> bool:
    return True

# @quantization.for_transformer(quantize_on_the_fly=True)
@servable_model_registry.register
class LLaMA7BFP16TPUv5e4(LLaMA7BFP16TPUv5e1):
  BATCH_SIZE = 1
  INPUT_SEQ_LEN = 64
  MAX_DECODE_STEPS = 64
  NUM_SAMPLES = 1
  TOP_K = 1

  ICI_MESH_SHAPE = [1, 1, 4]

  @property
  def test_mode(self) -> bool:
    return False


@servable_model_registry.register
class LLaMA7BFP16TPUv5e8(LLaMA7BFP16TPUv5e1):
  ICI_MESH_SHAPE = [1, 1, 8]


# llama-13b models
@servable_model_registry.register
class LLaMA13BFP16TPUv5e4(BaseLLaMA):
  """13B model on a A100-40GB.

  April 12, 2023
  Latency = 5.06s with 128 decoded tokens. 38ms per output token.
  """

  NUM_LAYERS = 40
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = 40
  MODEL_DIMS = 5120
  HIDDEN_DIMS = 13824

  ICI_MESH_SHAPE = [1, 1, 4]

  SPM_MODEL = 'gs://cloud-tpu-inference-public/sax-tokenizers/llama/llama-tokenizer.model'
  INPUT_SEQ_LEN = 128
  MAX_DECODE_STEPS = 32
  ENABLE_GENERATE_STREAM = False

  @property
  def test_mode(self) -> bool:
    return False
  

@servable_model_registry.register
class LLaMA13BFP16TPUv5e8(LLaMA13BFP16TPUv5e4):
    ICI_MESH_SHAPE = [1, 1, 8]


@servable_model_registry.register
class LLaMA13BFP16TPUv5e16(LLaMA13BFP16TPUv5e4):
    ICI_MESH_SHAPE = [1, 1, 16]


# llama-33b models
@servable_model_registry.register
class LLaMA33BFP16TPUv5e8(BaseLLaMA):
  """33B model on TPU v4-8.

  April 12, 2023
  Latency = 3.35s with 128 decoded tokens. 25ms per output token.
  """

  NUM_LAYERS = 60
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = 52
  MODEL_DIMS = 6656
  HIDDEN_DIMS = 17920

  ICI_MESH_SHAPE = [1, 1, 8]

  SPM_MODEL = 'gs://cloud-tpu-inference-public/sax-tokenizers/llama/llama-tokenizer.model'
  INPUT_SEQ_LEN = 128
  MAX_DECODE_STEPS = 32
  ENABLE_GENERATE_STREAM = False

  @property
  def test_mode(self) -> bool:
    return False


@servable_model_registry.register
class LLaMA33BFP16TPUv5e16(LLaMA33BFP16TPUv5e8):
    ICI_MESH_SHAPE = [1, 1, 16]


@servable_model_registry.register
class LLaMA33BFP16TPUv5e32(LLaMA33BFP16TPUv5e8):
    ICI_MESH_SHAPE = [1, 1, 32]
  

# llama-65b models
@servable_model_registry.register
class LLaMA65BFP16TPUv5e64(BaseLLaMA):
  NUM_LAYERS = 80
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = 64
  MODEL_DIMS = 8192
  HIDDEN_DIMS = 22016

  ICI_MESH_SHAPE = [1, 1, 64]
  
  SPM_MODEL = 'gs://cloud-tpu-inference-public/sax-tokenizers/llama/llama-tokenizer.model'
  INPUT_SEQ_LEN = 512
  MAX_DECODE_STEPS = 128
  ENABLE_GENERATE_STREAM = False

  @property
  def test_mode(self) -> bool:
    return False

  
@servable_model_registry.register
class LLaMA65BFP16TPUv5e32(LLaMA65BFP16TPUv5e64):
  ICI_MESH_SHAPE = [1, 1, 32]


@servable_model_registry.register
class LLaMA65BFP16TPUv5e16(LLaMA65BFP16TPUv5e64):
  ICI_MESH_SHAPE = [1, 1, 16]


# llama2-70b models
@servable_model_registry.register
class LLaMA70BFP16TPUv5e64(BaseLLaMA):
  NUM_LAYERS = 80
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = 64
  MODEL_DIMS = 8192
  HIDDEN_DIMS = 28672
  USE_MQA = True
  NUM_KV_HEADS = 8

  ICI_MESH_SHAPE = [1, 1, 64]

  SPM_MODEL = 'gs://cloud-tpu-inference-public/sax-tokenizers/llama/llama-tokenizer.model'
  ENABLE_GENERATE_STREAM = False

  @property
  def test_mode(self) -> bool:
    return False


@servable_model_registry.register
# @quantization.for_transformer(quantize_on_the_fly=False)
class LLaMA70BFP16TPUv5e16(LLaMA70BFP16TPUv5e64):
  # Decoding params
  BATCH_SIZE = 36
  INPUT_SEQ_LEN = 1024
  NUM_SAMPLES = 1
  TOP_K = 1

  MAX_DECODE_STEPS = 1024
  BUCKET_KEYS = [32, 128, 1024]

  ICI_MESH_SHAPE = [1, 1, 16]

  @property
  def test_mode(self) -> bool:
    return True
  

@servable_model_registry.register
class LLaMA70BFP16TPUv5e32(LLaMA70BFP16TPUv5e16):
  BATCH_SIZE = 72
  INPUT_SEQ_LEN = 1024
  MAX_DECODE_STEPS = 1024
  BUCKET_KEYS = [2048]
  ICI_MESH_SHAPE = [1, 1, 32]
  

@servable_model_registry.register
@quantization.for_transformer(use_symmetric=False, linear_only=True, num_bits=4)
class LLaMA70BFP16TPUv5e8(LLaMA70BFP16TPUv5e16):
  # Decoding params
  BATCH_SIZE = 1
  INPUT_SEQ_LEN = 1024
  MAX_DECODE_STEPS = 1024
  NUM_SAMPLES = 1
  TOP_K = 1

  ICI_MESH_SHAPE = [1, 1, 8]
  


@servable_model_registry.register
@quantization.for_transformer()
class LLaMA13BFP16TPUv5e8(BaseLLaMA):
  NUM_LAYERS = 40
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = 40
  MODEL_DIMS = 5120
  HIDDEN_DIMS = 13824

  ICI_MESH_SHAPE = [1, 8, 1]

  SPM_MODEL = 'gs://cloud-tpu-inference-public/sax-tokenizers/llama/llama-tokenizer.model'
  
  INPUT_SEQ_LEN = 2048
  MAX_DECODE_STEPS = 256
  ENABLE_GENERATE_STREAM = False
  BATCH_SIZE = 28

  NUM_SAMPLES = 1
  TOP_K = 1

  # BATCH_WAIT_SECS=0.2

  @property
  def test_mode(self) -> bool:
    return True
  

@servable_model_registry.register
# @quantization.for_transformer(quantize_on_the_fly=False)
class LLaMA13BFP16TPUv5e4(BaseLLaMA):
  NUM_LAYERS = 40
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = 40
  MODEL_DIMS = 5120
  HIDDEN_DIMS = 13824

  ICI_MESH_SHAPE = [1, 2, 2]

  SPM_MODEL = 'gs://cloud-tpu-inference-public/sax-tokenizers/llama/llama-tokenizer.model'
  
  INPUT_SEQ_LEN = 2048
  MAX_DECODE_STEPS = 256
  ENABLE_GENERATE_STREAM = False
  BATCH_SIZE = 12

  NUM_SAMPLES = 1
  TOP_K = 1

  # def compiler_options(self) -> jax.stages.CompilerOptions:
  #   return {'xla_jf_auto_cross_replica_sharding': 'False', 'xla_tpu_nd_short_transfer_max_chunks': '2048', 'xla_tpu_perform_spmd_cse_prevention': 'True', 'xla_tpu_rwb_fusion': 'False', }

  @property
  def test_mode(self) -> bool:
    return True