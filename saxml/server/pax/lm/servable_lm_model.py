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
"""Wraps a model with LMService APIs."""

import abc
import dataclasses
import functools
import inspect
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from absl import logging
import jax
from jax import numpy as jnp
from jax.experimental import host_callback as hcb
import numpy as np
from praxis import base_layer
from praxis import base_model
from praxis import decoder_hparams
from praxis import decoder_utils
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from saxml.server.jax import np_tf_sess_wrapper
from saxml.server.jax import servable_model as jax_servable_model
from saxml.server.pax import servable_model
from saxml.server.pax import servable_model_params
from saxml.server.pax.lm import lm_tokenizer
from saxml.server.pax.lm import servable_lm_common
from saxml.server.services import lm_service
import tensorflow as tf


JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedNpTensor = pytypes.NestedNpTensor
PRNGKey = pytypes.PRNGKey
NestedMap = py_utils.NestedMap
NestedPartitionSpec = pytypes.NestedPartitionSpec
NestedTfTensor = pytypes.Nested[tf.Tensor]
NestedNpOrTfTensor = Union[NestedNpTensor, NestedTfTensor]
LMMethodName = lm_service.LMMethodName
HostTensors = servable_model.HostTensors
ShapesAndDtypes = servable_model.ShapesAndDtypes
InputShapeInfo = servable_lm_common.InputShapeInfo
TensorSpec = servable_lm_common.TensorSpec

decode_tf_post_processing = servable_lm_common.decode_tf_post_processing


@dataclasses.dataclass
class ScoreHParams(servable_model_params.ServableMethodParams):
  """HParameters for LM score method.

  Attributes:
    max_input_seq_len: static prefix sequence length dimension size.
    max_suffix_seq_len: static suffix sequence length dimension size. Defaults
      to be equal to `max_input_seq_len` if not set. Inputs are padded or
      truncated to (max_input_seq_len + max_suffix_seq_len) size.
    include_eos_score: whether to add EOS score to the result.
  """

  max_input_seq_len: int = 0
  max_suffix_seq_len: int = 0
  include_eos_score: bool = False
  fetch_prefix_lengths_from_inputs: bool = False
  output_geometric_mean_prob_score: bool = False


@dataclasses.dataclass
class DecodeHParams(servable_model_params.ServableMethodParams):
  """HParameters for LM sample decode method.

  Attributes:
    max_input_seq_len: static sequence length dimension size. Inputs are padded
      or truncated to this size.
    decoder: decoder params.
    include_prefix_in_result: whether to include the input prefix in the result.
    encoder_decoder_model: whether this is an encoder decoder model.
    t5_model: whether this is a T5 flaxformer based model.
    output_geometric_mean_prob_score: Whether to return geometric mean of prob
      score instead of sum of log prob as the score.
    output_avg_entropy_score: Whether to return avg entropy score instead of sum
      of log prob as the score.
  """

  max_input_seq_len: int = 0
  decoder: decoder_hparams.DecoderHParams = dataclasses.field(
      default_factory=decoder_hparams.DecoderHParams
  )
  include_prefix_in_result: bool = False
  encoder_decoder_model: bool = False
  t5_model: bool = False
  stream_interval_steps: int = 1
  fetch_prefix_lengths_from_inputs: bool = False
  output_geometric_mean_prob_score: bool = False
  output_avg_entropy_score: bool = False


@dataclasses.dataclass
class TextToEmbeddingHParams(servable_model_params.ServableMethodParams):
  """HParameters for TextToEmbedding method.

  Attributes:
    max_input_seq_len: static prefix sequence length dimension size.
    max_suffix_seq_len: static suffix sequence length dimension size. Defaults
      to 1 and `max_input_seq_len` is autodecremented by 1. This is to ensure
      the prefix and suffix both have EOS for tokenization. Inputs are padded or
      truncated to (max_input_seq_len + max_suffix_seq_len) size.
    include_eos_score: whether to add EOS score to the result.
    output_embedding_name: The name of the embedding to use from the model's
      outputs.  Required.
    output_padding_name: The name of padding to use from the model's outputs.
      This is used when output embedding has a variable length. For example,
      returning embeddings of all tokens in a sequence, rather than pooling one
      embedding out.
    model_method_name: The name of the method to call to extract embeddings from
      an input image.  Required.
  """

  max_input_seq_len: int = 0
  max_suffix_seq_len: int = 1
  include_eos_score: bool = False
  output_embedding_name: Optional[str] = None
  output_padding_name: Optional[str] = None
  model_method_name: Optional[str] = None


@dataclasses.dataclass
class GradientHParams(ScoreHParams):
  """HParameters for LM gradient method, inheriting from ScoreHParams.

  Additional attributes:
    inputs_tensor_names: tensors to take gradients with respect to in inputs.
    mdl_vars_tensors_names: tensors to take gradients with respect to in
      mdl_vars.
  """

  inputs_tensor_names: Optional[List[str]] = None
  mdl_vars_tensor_names: Optional[List[str]] = None


class ServableLMModelParams(
    servable_model_params.ServableModelParams, metaclass=abc.ABCMeta
):
  """A base class that each LM model config needs to implement for serving."""

  @abc.abstractmethod
  def serving_tokenizer(self) -> pax_fiddle.Config[lm_tokenizer.LMTokenizer]:
    """Tokenizer params used by serving."""

  def methods(self) -> Dict[str, servable_model_params.ServableMethodParams]:
    methods = {}
    score = self.score()  # pylint: disable=assignment-from-none
    if score is not None:
      methods[LMMethodName.SCORE] = score
    generate = self.generate()  # pylint: disable=assignment-from-none
    if generate is not None:
      methods[LMMethodName.GENERATE] = generate
    generate_stream = self.generate_stream()  # pylint: disable=assignment-from-none
    if generate_stream is not None:
      methods[LMMethodName.GENERATE_STREAM] = generate_stream
    text_to_embedding = self.text_to_embedding()  # pylint: disable=assignment-from-none
    if text_to_embedding is not None:
      methods[LMMethodName.EMBED] = text_to_embedding
    gradient = self.gradient()  # pylint: disable=assignment-from-none
    if gradient is not None:
      methods[LMMethodName.GRADIENT] = gradient
    return methods

  def score(self) -> Optional[ScoreHParams]:
    """Returns the params for the score method."""
    return None

  def generate(self) -> Optional[DecodeHParams]:
    """Returns the params for the decode method."""
    return None

  def gradient(self) -> Optional[GradientHParams]:
    """Returns the params for the gradient method."""
    return None

  def generate_stream(self) -> Optional[DecodeHParams]:
    """Returns the params for the decode method."""
    return None

  def text_to_embedding(self) -> Optional[TextToEmbeddingHParams]:
    return None

  def create_model(self, primary_process_id: int) -> 'ServableLMModel':
    return ServableLMModel(
        self,
        primary_process_id,
        self.get_checkpoint_type(),
        test_mode=self.test_mode,
        enable_auto_sharding=self.enable_auto_sharding,
        compiler_options=self.compiler_options(),
        do_eval=self.do_eval,
    )


class ServableLMMethod(servable_model.ServableMethod):
  """Implements common method of LM."""

  @classmethod
  def service_id(cls) -> str:
    return lm_service.SERVICE_ID

  @property
  def sorted_seq_lens(self) -> List[int]:
    """A list of sorted supported (ascending order) sequence lengths."""
    return sorted(self._bucket_keys) if self._bucket_keys else [-1]

  def get_sorted_input_shapes(self) -> List[InputShapeInfo]:
    result = []
    for batch_size in self._sorted_batch_sizes:
      for seq_len in self.sorted_seq_lens:
        result.append(InputShapeInfo(batch_size, seq_len))
    return result

  def deserialize_input_shape(self, unpadded_shape_str: str) -> InputShapeInfo:
    """Deserialize input shape from a str."""
    return servable_lm_common.deserialize_input_shape(
        unpadded_shape_str, self._dummy_bucket_key
    )

  def get_unpadded_shape(
      self, unpadded_batch_size, inputs: HostTensors
  ) -> InputShapeInfo:
    return InputShapeInfo(
        unpadded_batch_size,
        servable_lm_common.get_max_seq_len_in_batch(
            inputs, self._dummy_bucket_key, self._bucket_keys
        ),
    )

  def get_padded_input_shape(
      self, unpadded_shape: InputShapeInfo
  ) -> InputShapeInfo:
    """Get padded input shape.

    Args:
      unpadded_shape: Unpadded shape information contains batch size or sequence
        length.

    Returns:
      Padded input shape.
    Raises:
      ValueError if unpadded batch size or sequence length too large.
    """
    padded_shape = super().get_padded_input_shape(unpadded_shape)
    if self._bucket_keys is None:
      return InputShapeInfo(padded_shape.batch_size)
    padded_seq_len = servable_lm_common.get_padded_input_seq_len(
        unpadded_shape.seq_len, self.sorted_seq_lens
    )
    return InputShapeInfo(padded_shape.batch_size, padded_seq_len)

  def get_dummy_inputs(self, input_shape: InputShapeInfo) -> HostTensors:
    """Returns host tensors with dummy data at a batch size."""
    batched_input = self.pre_processing(
        [self._dummy_input_sample] * input_shape.batch_size
    )

    return servable_lm_common.handle_host_input_with_input_shape(
        batched_input, input_shape
    )

  def resize_host_array(
      self,
      x: np.ndarray,
      global_input_shape_dtype: ShapesAndDtypes,
      unpadded_input_shape: InputShapeInfo,
  ):
    """Resizes x to the desired shape.

    Args:
      x: Host tensor.
      global_input_shape_dtype: Global input shape and dtype for this tensor.
      unpadded_input_shape: Unpadded input shape.

    Returns:
      host array after padding or slice of x.
    """
    x = servable_lm_common.resize_host_array(
        x, global_input_shape_dtype, unpadded_input_shape
    )

    # Let the parent class handle the batch dim.
    x = super().resize_host_array(
        x, global_input_shape_dtype, unpadded_input_shape
    )
    return x

  def _get_longest_seqlen(self, inputs: NestedNpTensor) -> int:
    """Gets the longest sequence length in a batch."""
    if 'paddings' in inputs:
      prefix_lengths = np.sum(1.0 - inputs['paddings'], axis=-1).astype(
          np.int32
      )  # pytype: disable=attribute-error
      return np.max(prefix_lengths).item()
    return inputs['ids'].shape[1]

  def get_unpadded_branch_key(self, inputs: NestedNpTensor) -> int:
    return self._get_longest_seqlen(inputs)

  def get_branch_inputs(
      self, inputs: NestedJTensor, branch_key: int
  ) -> NestedJTensor:
    """Returns the inputs for a branch key.

    Args:
      inputs: inputs with padded sequence lengths.
      branch_key: branch_key is seqlen.

    Returns:
      Tensors sliced at sequence length dimension.
    """
    seqlen = branch_key

    def _slice_fn(x):
      """The function to slice at sequence dimension."""
      if not isinstance(x, JTensor):
        return x
      if len(x.shape) == 2 and x.shape[1] >= seqlen:
        return jax.lax.slice(x, [0, 0], [x.shape[0], seqlen])
      return x

    return jax.tree_util.tree_map(_slice_fn, inputs)

  def get_maxlen(self) -> int:
    """Gets the max input sequence lengths."""
    raise NotImplementedError('get_maxlen not implemented')

  def output_seq_dim(self) -> int:
    """Gets the sequence dim in the output result."""
    raise NotImplementedError('output_seq_dim not implemented')

  def extra_pad_result(
      self, result: NestedJTensor, branch_key: int
  ) -> NestedJTensor:
    """Special paddings for some tensors."""
    return result

  def pad_result(
      self, result: NestedJTensor, pad_len: int, seq_dim: int
  ) -> NestedJTensor:
    """Pads the result at sequence dimension."""

    def _pad_fn(x):
      if not isinstance(x, JTensor) or len(x.shape) < seq_dim + 1:
        return x
      paddings = [[0, 0]] * len(x.shape)
      paddings[seq_dim] = [0, max(0, pad_len)]
      padded = jnp.pad(x, paddings)
      return padded

    return jax.tree_map(_pad_fn, result)

  def post_process_branch_outputs(
      self, outputs: NestedJTensor, branch_key: int
  ) -> NestedJTensor:
    """Post process branch outputs."""
    seqlen = branch_key
    maxlen = self.get_maxlen()
    result, state = outputs
    padded_result = self.pad_result(
        result, maxlen - seqlen, self.output_seq_dim()
    )
    padded_result = self.extra_pad_result(padded_result, branch_key)
    padded_state = self.pad_result(state, maxlen - seqlen, 1)
    return padded_result, padded_state

  @property
  def model_fn_input_polymorphic_shape(self) -> pytypes.Nested[str]:
    """Returns a batch polymorphic shape for jax2tf."""
    batched_host_dummy = self.get_dummy_inputs(InputShapeInfo(self.batch_size))
    batched_host_dummy = self.update_extra_inputs(
        batched_host_dummy,
        self.batch_size,
        [self.default_extra_inputs] * self.batch_size,
    )

    batch_pattern = 'b' if len(self.sorted_batch_sizes) > 1 else '_'
    if len(self.sorted_seq_lens) > 1:
      seq_pattern = f'{batch_pattern}, t'
    else:
      seq_pattern = f'{batch_pattern}, _'
    shape_patterns = jax.tree_util.tree_map(
        lambda x: seq_pattern if len(x.shape) == 2 else f'{batch_pattern}, ...',
        batched_host_dummy,
    )
    # Apply seq len polymorphism exclusion.
    polymorphic_seq_len_exclusion = set(
        self.method_params.polymorphic_seq_len_exclusion or []
    )
    # Do not apply polymorphic seq len to extra inputs.
    if self.default_extra_inputs:
      polymorphic_seq_len_exclusion |= self.default_extra_inputs.keys()
    if polymorphic_seq_len_exclusion:
      for key in polymorphic_seq_len_exclusion:
        shape_patterns[key] = f'{batch_pattern}, ...'

    return shape_patterns


class LMScoreMethod(ServableLMMethod):
  """Implements the score method of an LM."""

  def __init__(
      self,
      model: base_model.BaseModel,
      model_state: servable_model.ServableModelState,
      prng_key: PRNGKey,
      score_params: ScoreHParams,
      tokenizer_p: Any,
      exportable: bool = False,
      enable_auto_sharding: bool = False,
      compiler_options: jax.stages.CompilerOptions | None = None,
  ):
    self._tokenizer = tokenizer_p.Instantiate()
    self._score_params = score_params
    dummy_input_sample = ('', [''])
    # TODO(b/289379065): Remove this workaround here.
    if (
        isinstance(self._tokenizer, lm_tokenizer.LMTokenizer)
        and self._tokenizer.tokenized
    ):
      dummy_input_sample = ('1', ['1'])
    logging.info('Using np_tf_sess_wrapper on LMScoreMethod.tf_pre_processing')
    self._tf_sess_pre_processing = np_tf_sess_wrapper.wrap_tf_session(
        # `bucketize_inputs` is only used in SavedModel export. The sax-native
        # serving has an equivalent bucketization after `pre_processing`.
        lambda *args: self.tf_pre_processing(*args, bucketize_inputs=False)
    )
    super().__init__(
        model,
        'compute_predictions',
        model_state,
        score_params,
        prng_key,
        dummy_input_sample,
        exportable=exportable,
        enable_auto_sharding=enable_auto_sharding,
        compiler_options=compiler_options,
    )

  def fetch_output(
      self, model_fn_outputs: NestedJTensor, model_fn_inputs: NestedJTensor
  ) -> NestedJTensor:
    if 'scores' in model_fn_outputs[0]:
      # Custom scores.
      return model_fn_outputs[0]['scores']
    # per_token_xent or per_example_xnent is -logprobs. We return the negative
    # value so that higher score is better.
    if 'per_token_xent' not in model_fn_outputs[0]:
      assert 'per_example_xent' in model_fn_outputs[0]
      assert model_fn_outputs[0].per_example_xent.ndim == 1  # pytype: disable=attribute-error  # jax-ndarray
      return -model_fn_outputs[0].per_example_xent  # pytype: disable=attribute-error  # jax-ndarray
    assert len(model_fn_outputs[0].per_token_xent.shape) > 1  # pytype: disable=attribute-error  # jax-ndarray
    xnent_len = model_fn_outputs[0].per_token_xent.shape[1]  # pytype: disable=attribute-error  # jax-ndarray
    assert xnent_len == model_fn_inputs.ids.shape[1]  # pytype: disable=attribute-error  # jax-ndarray
    per_token_logprobs = -model_fn_outputs[0].per_token_xent  # pytype: disable=attribute-error  # jax-ndarray
    non_paddings = 1.0 - model_fn_inputs.paddings  # pytype: disable=attribute-error  # jax-ndarray
    if not self._score_params.include_eos_score and self._tokenizer.append_eos:
      non_paddings = jnp.pad(
          # TODO(b/263808957): change back to non_paddings[:, 1:] once the bug
          # is fixed.
          jax.lax.dynamic_slice_in_dim(
              non_paddings, 1, non_paddings.shape[1] - 1, axis=1
          ),
          [[0, 0], [0, 1]],
      )
    sum_per_token_logprobs = jnp.sum(
        per_token_logprobs * model_fn_inputs.score_masks * non_paddings,  # pytype: disable=attribute-error  # jax-ndarray
        axis=-1,
        keepdims=True,
    )
    if self._score_params.output_geometric_mean_prob_score:
      num_output_tokens = jnp.sum(
          model_fn_inputs.score_masks * non_paddings,  # pytype: disable=attribute-error  # jax-ndarray
          axis=-1,
          keepdims=True,
      )
      num_output_tokens = jnp.where(num_output_tokens > 0, num_output_tokens, 1)
      return jnp.exp(sum_per_token_logprobs / num_output_tokens)
    else:
      return sum_per_token_logprobs

  def get_maxlen(self) -> int:
    return (
        self._score_params.max_input_seq_len
        + self._score_params.max_suffix_seq_len
    )

  def output_seq_dim(self) -> int:
    return 1

  def pre_processing(
      self, raw_inputs: List[Tuple[str, List[str]]]
  ) -> NestedNpTensor:
    prefixes = np.array([prefix for prefix, _ in raw_inputs])
    for _, suffix in raw_inputs:
      assert len(suffix) <= 1, 'Only one suffix score is supported in lm.score'
    suffixes = np.array([suffix[0] for _, suffix in raw_inputs])
    return self._tf_sess_pre_processing(prefixes, suffixes)

  def post_processing(self, compute_outputs: NestedNpTensor) -> List[float]:
    assert isinstance(compute_outputs, pytypes.NpTensor)
    scores = list(compute_outputs.astype(float))
    return scores

  def tf_pre_processing(
      self,
      prefixes: NestedNpOrTfTensor,
      suffixes: NestedNpOrTfTensor,
      extra_inputs: Optional[Mapping[str, Any]] = None,
      bucketize_inputs: bool = True,
  ) -> NestedTfTensor:
    """Tokenizes `prefixes` and `suffixes` using TF ops.

    This also implements `ExportableToSavedModel.tf_pre_processing`.

    Args:
      prefixes: the prefix text batch of shape [batch_size].
      suffixes: the suffix text batch of shape [batch_size].
      extra_inputs: optional mapping of extra input key to tensor or tensor spec
        of shape [batch_size].
      bucketize_inputs: whether to bucketize the preprocessed inputs based on
        max sequence length in the batch.

    Returns:
      A NestedMap of preprocessed tensors.
    """
    preprocessed = servable_lm_common.tf_tokenize_inputs(
        prefixes,
        suffixes,
        self._tokenizer,
        self._score_params.max_input_seq_len,
        self._score_params.max_suffix_seq_len,
        self._score_params.include_eos_score,
    )

    if bucketize_inputs:
      preprocessed = servable_lm_common.bucketize_tokenized_inputs(
          self.sorted_seq_lens, preprocessed
      )

    if extra_inputs:
      preprocessed.update(extra_inputs)

    return preprocessed

  def tf_post_processing(
      self, compute_outputs: NestedNpOrTfTensor
  ) -> NestedNpOrTfTensor:
    """Implements `ExportableToSavedModel.tf_post_processing`."""
    return {'scores': compute_outputs}

  def input_signature(
      self, batch_size: Optional[int]
  ) -> Tuple[TensorSpec, TensorSpec, Mapping[str, TensorSpec]]:
    """Implements `ExportableToSavedModel.input_signature`."""
    return (
        tf.TensorSpec([batch_size], dtype=tf.string, name='prefixes'),
        tf.TensorSpec([batch_size], dtype=tf.string, name='suffixes'),
        servable_lm_common.extra_inputs_to_tf_signature(
            self._extra_inputs,
            batch_size,
            self.method_params.extra_inputs_dtypes,
        ),
    )

  @property
  def tf_trackable_resources(self) -> Any:
    """Implements `ExportableToSavedModel.tf_trackable_resources`."""
    return None


class LMDecodeMethod(ServableLMMethod):
  """Base decode method of an LM."""

  def __init__(
      self,
      model: base_model.BaseModel,
      model_state: servable_model.ServableModelState,
      prng_key: PRNGKey,
      method_hparams: DecodeHParams,
      tokenizer_p: Any,
      exportable: bool = False,
      streamable: bool = False,
      load: bool = True,
      enable_auto_sharding: bool = False,
      compiler_options: jax.stages.CompilerOptions | None = None,
  ):
    self._tokenizer = tokenizer_p.Instantiate()
    self._method_hparams = method_hparams
    dummy_input_sample = ''
    # TODO(b/289379065): Remove this workaround here.
    if (
        isinstance(self._tokenizer, lm_tokenizer.LMTokenizer)
        and self._tokenizer.tokenized
    ):
      dummy_input_sample = '1'
    if isinstance(method_hparams, DecodeHParams):
      self._include_prefix_in_result = method_hparams.include_prefix_in_result
    logging.info('Using np_tf_sess_wrapper on LMDecodeMethod.tf_pre_processing')
    self._tf_sess_pre_processing = np_tf_sess_wrapper.wrap_tf_session(
        # `bucketize_inputs` is only used in SavedModel export. The sax-native
        # serving has an equivalent bucketization after `pre_processing`.
        lambda *args: self.tf_pre_processing(*args, bucketize_inputs=False)
    )
    logging.info(
        'Using np_tf_sess_wrapper on LMDecodeMethod.tf_post_processing'
    )
    self._tf_sess_post_processing = np_tf_sess_wrapper.wrap_tf_session(
        # pylint: disable=g-long-lambda
        lambda *args: decode_tf_post_processing(
            *args,
            tokenizer=self._tokenizer,
            t5_model=self._method_hparams.t5_model,
            include_prefix_in_result=self._include_prefix_in_result,
        ),
        False,
    )
    self._streamable = streamable
    logging.info('Initialize LMDecodeMethod to be streamable=%s.', streamable)

    def _init_stream_and_decode(new_ids):
      batch_size = tf.shape(new_ids)[:-1]
      return self._tokenizer.DecodeOnStream(
          new_ids, self._tokenizer.InitStream(batch_size)
      )

    self._tf_sess_first_stream_step = np_tf_sess_wrapper.wrap_tf_session(
        _init_stream_and_decode, False
    )
    self._tf_sess_stream_step = np_tf_sess_wrapper.wrap_tf_session(
        self._tokenizer.DecodeOnStream, False
    )
    self._tf_sess_stream_finish = np_tf_sess_wrapper.wrap_tf_session(
        self._tokenizer.FinishStream, False
    )

    super().__init__(
        model,
        'decode',
        model_state,
        method_hparams,
        prng_key,
        dummy_input_sample,
        exportable=exportable,
        load=load,
        enable_auto_sharding=enable_auto_sharding,
        compiler_options=compiler_options,
    )

  def call_model_function(self, inputs, mdl_vars, prng_key):
    k1, k2 = prng_key

    kwargs = {}
    # if self.streamable_output:

    #   def callback_fn(x, _):
    #     assert self.model_state.is_primary_host
    #     self.enqueue_stream_output(x)

    #   kwargs['result_callback'] = decoder_utils.StreamingResultCallback(
    #       functools.partial(
    #           hcb.id_tap, callback_fn, device_index=self.callback_device_index
    #       ),
    #       interval_steps=self._method_hparams.stream_interval_steps,
    #   )

    if (
        'callback_device_index'
        in inspect.signature(self._model.decode_with_params).parameters
    ):
      kwargs['callback_device_index'] = self.callback_device_index

    outputs = self._model.apply(
        mdl_vars,
        input_batch=inputs,
        method=self._model.decode_with_params,
        mutable=[
            base_layer.NON_TRAINABLE,
            base_layer.DECODE_CACHE,
            base_layer.PREFIX_DECODE_CACHE,
        ],
        rngs={
            base_layer.PARAMS: k1,
            base_layer.RANDOM: k2,
        },
        decoder_params=self._method_hparams.decoder,
        **kwargs,
    )
    return outputs

  @property
  def streamable(self) -> bool:
    return self._streamable

  def fetch_output(
      self, model_fn_outputs: NestedJTensor, model_fn_inputs: NestedJTensor
  ) -> NestedJTensor:
    return servable_lm_common.decode_fetch_output(
        model_fn_outputs,
        model_fn_inputs,
        self._method_hparams.t5_model,
        self._method_hparams.fetch_prefix_lengths_from_inputs,
    )

  def pre_processing(self, raw_inputs: List[str]) -> NestedNpTensor:
    texts = np.array(raw_inputs)
    return self._tf_sess_pre_processing(texts)

  def get_maxlen(self) -> int:
    return self._method_hparams.max_input_seq_len

  def output_seq_dim(self) -> int:
    return 2

  def extra_pad_result(
      self, result: NestedJTensor, branch_key: int
  ) -> NestedJTensor:
    """Extra pad result from decoding."""
    seqlen = branch_key

    def _pad_fn(sub_result):
      paddings = [[0, 0], [0, self.get_maxlen() - seqlen]]
      for key in {'paddings', 'weights', 'ids'}:
        if key in sub_result:
          sub_result[key] = jnp.pad(sub_result[key], paddings)
      return sub_result

    return tuple([_pad_fn(sub_result) for sub_result in result])

  def post_processing(
      self, compute_outputs: NestedNpTensor
  ) -> List[Tuple[List[str], List[float]]]:
    # A list of results for the inputs. Each element has multiple samples from
    # the decoding algorithm, which has a list of strings and a list of scores.
    post_processed = self._tf_sess_post_processing(compute_outputs)
    # post_processed = self.tf_post_processing(compute_outputs)
    batched_decoded = post_processed['topk_decoded']
    batched_scores = post_processed['topk_scores']

    # Override scores according to hparams
    assert (
        not self._method_hparams.output_geometric_mean_prob_score
        or not self._method_hparams.output_avg_entropy_score
    )
    if self._method_hparams.output_geometric_mean_prob_score:
      num_output_tokens = np.count_nonzero(post_processed['topk_ids'], axis=2)
      num_output_tokens = np.where(num_output_tokens > 0, num_output_tokens, 1)
      batched_scores = np.exp(batched_scores / num_output_tokens)
    elif self._method_hparams.output_avg_entropy_score:
      batched_scores = post_processed['mean_entropy']
    return [
        ([d.decode() for d in decoded], list(scores))
        for decoded, scores in zip(batched_decoded, batched_scores)
    ]

  def post_processing_stream(
      self,
      compute_outputs: Optional[NestedNpTensor] = None,
      stream_state: Optional[Any] = None,
  ) -> Tuple[List[Tuple[List[str], List[float]]], Optional[Any]]:
    if compute_outputs is None and stream_state is None:
      raise ValueError('compute_outputs and stream_state cannot both be None')

    if compute_outputs is None:
      batch_decoded = self._tf_sess_stream_finish(stream_state)
      stream_state = None
      scores = np.zeros(batch_decoded.shape)
    elif stream_state is None:
      batch_decoded, stream_state = self._tf_sess_first_stream_step(
          compute_outputs['output_ids']
      )
      scores = compute_outputs['scores']
    else:
      batch_decoded, stream_state = self._tf_sess_stream_step(
          compute_outputs['output_ids'], stream_state
      )
      scores = compute_outputs['scores']

    return [(d, s) for (d, s) in zip(batch_decoded, scores)], stream_state

  def tf_pre_processing(
      self,
      texts: NestedNpOrTfTensor,
      extra_inputs: Optional[Mapping[str, Any]] = None,
      bucketize_inputs: bool = True,
  ) -> NestedTfTensor:
    """Tokenizes `texts` using TF ops.

    This also implements `ExportableToSavedModel.tf_pre_processing`. If extra
    inputs are provided in the input signature, the exported
    method will take a batched tensor too. See also the `input_signature` method
    of this class.

    Args:
      texts: the input text of shape [batch_size].
      extra_inputs: optional mapping of extra input key to tensor or tensor spec
        of shape [batch_size].
      bucketize_inputs: whether to bucketize the preprocessed inputs based on
        max sequence length in the batch.

    Returns:
      A NestedMap of preprocessed tensors.
    """
    ids, paddings, prefix_lengths, weights = (
        servable_lm_common.decode_tf_tokenize_inputs(
            texts,
            self._tokenizer,
            self._method_hparams.max_input_seq_len,
            self._method_hparams.t5_model,
        )
    )

    batch_size = tf.shape(ids)[0]
    if self._method_hparams.t5_model:
      target_length = self._method_hparams.decoder.seqlen
      preprocessed = py_utils.NestedMap(
          encoder_input_tokens=ids,
          decoder_input_tokens=tf.ones((batch_size, target_length)),
      )
    elif self._method_hparams.encoder_decoder_model:
      src = py_utils.NestedMap(
          ids=tf.cast(ids, tf.int32),
          paddings=paddings,
      )
      tgt = py_utils.NestedMap(
          ids=tf.zeros((batch_size, 1), dtype=tf.int32),
          paddings=tf.zeros((batch_size, 1)),
      )
      preprocessed = py_utils.NestedMap(
          src=src,
          tgt=tgt,
          prefix_lengths=tf.ones((batch_size), tf.int32),
      )
    else:
      preprocessed = py_utils.NestedMap(
          ids=ids,
          paddings=paddings,
          prefix_lengths=tf.cast(prefix_lengths, tf.int32),
          weights=weights,
      )

    if bucketize_inputs:
      preprocessed = servable_lm_common.bucketize_tokenized_inputs(
          self.sorted_seq_lens, preprocessed
      )

    if extra_inputs:
      preprocessed.update(extra_inputs)

    return preprocessed

  def tf_post_processing(
      self, compute_outputs: NestedNpOrTfTensor
  ) -> NestedNpOrTfTensor:
    """Post-process the outputs using TF ops.

    This also implements `ExportableToSavedModel.tf_post_processing`.

    Args:
      compute_outputs: the outputs of the model function.

    Returns:
      A mapping that contains the decoded tensors, scores and ids of the topk
      results.
    """
    return decode_tf_post_processing(
        compute_outputs,
        tokenizer=self._tokenizer,
        t5_model=self._method_hparams.t5_model,
        include_prefix_in_result=self._include_prefix_in_result,
    )

  def input_signature(
      self, batch_size: Optional[int]
  ) -> Tuple[TensorSpec, Mapping[str, TensorSpec]]:
    """Implements `ExportableToSavedModel.input_signature`."""
    return (
        tf.TensorSpec([batch_size], dtype=tf.string, name='text'),
        servable_lm_common.extra_inputs_to_tf_signature(
            self._extra_inputs,
            batch_size,
            self.method_params.extra_inputs_dtypes,
        ),
    )

  @property
  def tf_trackable_resources(self) -> Any:
    """Implements `ExportableToSavedModel.tf_trackable_resources`."""
    return None


## LMDecoderMethod for continous batching
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from jax.experimental import pjit
# string values can be used in host callbacks and are converted to integers
# when passing around through a global repository on host.
ExtraInput = Dict[str, Union[float, str, List[float]]]
# TODO(sax-dev): define these types or use pax's definitions.
HostTensors = Any
DeviceTensors = Any
PSpecs = Any
ShapesAndDtypes = Any
Shapes = Any
JaxTensors = Any

from saxml.server import servable_model as sax_servable_model
SaxServableModelInputShapeInfo = sax_servable_model.InputShapeInfo

def remove_padding(x: jnp.ndarray, shape: Sequence[int]) -> jnp.ndarray:
  if list(x.shape) == shape:
    return x
  return jax.lax.slice(x, [0] * x.ndim, shape)

ServableModelState = jax_servable_model.ServableModelState
MethodInputInfo = jax_servable_model.MethodInputInfo

class LMDecodeMethodContinuousBatching(LMDecodeMethod):
  
  # initialize model state
  def init_model_state(self):
    logging.info("Calling init model state")
    input_shape_keys = list(self._per_bs_infos.keys())
    logging.info(input_shape_keys)
    
    assert self._method_hparams.decoder.enable_continuous_batching
    continuous_batching_batch_size = self._method_hparams.decoder.continuous_batching_batch_size
    input_shape = InputShapeInfo(continuous_batching_batch_size, input_shape_keys[0].seq_len)
    # logging.info("LMDecode method continuous batching size: {}".format(continuous_batching_batch_size))
    
    batch_tensors = self._per_bs_infos[input_shape].dummy_inputs
    # logging.info("dummy input batch: {}".format(batch_tensors))
    decode_data = self.device_compute_prefill(
      input_batch=batch_tensors,
      unpadded_shape=input_shape,
    )

    decode_state, decode_cache = self.device_compute_init_decode_state(
      input_batch=batch_tensors,
      unpadded_shape=input_shape,
      decode_data=decode_data
    )
    return decode_state, decode_cache

  def device_compute_prefill(
      self, input_batch: DeviceTensors, unpadded_shape: SaxServableModelInputShapeInfo
  ):
    padded_shape = self.get_padded_input_shape(unpadded_shape)
    with self.model_state.global_mesh:
      # logging.info("Calling greedy prefill")
      decode_data, decode_cache = self._per_bs_infos[padded_shape].device_fn[0](
        self.model_state.mdl_vars, input_batch
      )
      self.model_state.mdl_vars.update(decode_cache)
      # logging.info("model kv cache vars after prefill: {}".format(
      #   self.model_state.mdl_vars['decoder_cache']['lm']['transformer']['x_layers_0']['self_attention']['key_state'].shape)
      # )
      return decode_data
  
  def device_compute_init_decode_state(
      self, input_batch: DeviceTensors, unpadded_shape: SaxServableModelInputShapeInfo, decode_data
  ):
    padded_shape = self.get_padded_input_shape(unpadded_shape)
    with self.model_state.global_mesh:
      # logging.info("Calling greedy init decode state with padded shape: {}".format(padded_shape))
      init_decode_state, decode_cache = self._per_bs_infos[padded_shape].device_fn[1](
        self.model_state.mdl_vars, input_batch, decode_data
      )
      # logging.info("model kv cache vars after init_decode_state: {}".format(
      #   self.model_state.mdl_vars['decoder_cache']['lm']['transformer']['x_layers_0']['self_attention']['key_state'].shape)
      # )
    return init_decode_state, decode_cache

  def device_compute_prefill_and_insert(
      self, 
      input_batch: DeviceTensors, 
      unpadded_shape: SaxServableModelInputShapeInfo,
      decode_state,
      decode_cache,
      slot_idx):

    # logging.info("model kv cache vars before prefill_and_insert: {}".format(
    #   self.model_state.mdl_vars['decoder_cache']['lm']['transformer']['x_layers_0']['self_attention']['key_state'].shape)
    # )
    self.model_state.mdl_vars.update(decode_cache)
    # logging.info("model kv cache vars after first update prefill_and_insert: {}".format(
    #   self.model_state.mdl_vars['decoder_cache']['lm']['transformer']['x_layers_0']['self_attention']['key_state'].shape)
    # )

    padded_shape = self.get_padded_input_shape(unpadded_shape)
    with self.model_state.global_mesh:
      # logging.info("Calling greedy prefill and insert")
      decode_state, decode_cache = self._per_bs_infos[padded_shape].device_fn[4](
        self.model_state.mdl_vars, 
        input_batch, 
        decode_state,
        # decode_cache,
        slot_idx
      )

      # logging.info("model kv cache vars before second update prefill_and_insert: {}".format(
      #   self.model_state.mdl_vars['decoder_cache']['lm']['transformer']['x_layers_0']['self_attention']['key_state'].shape)
      # )

      # # self.model_state.mdl_vars.update(decode_cache)
      # logging.info("model kv cache vars after second update prefill_and_insert: {}".format(
      #   self.model_state.mdl_vars['decoder_cache']['lm']['transformer']['x_layers_0']['self_attention']['key_state'].shape)
      # )
      return decode_state, decode_cache

  def device_compute_decoding_step(
      self, 
      unpadded_shape: SaxServableModelInputShapeInfo, 
      decode_state,
      decode_cache,
      align_decode_state
  ):
    # logging.info("model kv cache vars before decoding_step: {}".format(
    #   self.model_state.mdl_vars['decoder_cache']['lm']['transformer']['x_layers_0']['self_attention']['key_state'].shape)
    # )
    self.model_state.mdl_vars.update(decode_cache)
    # logging.info("model kv cache vars after update decoding_step: {}".format(
    #   self.model_state.mdl_vars['decoder_cache']['lm']['transformer']['x_layers_0']['self_attention']['key_state'].shape)
    # )
    padded_shape = self.get_padded_input_shape(unpadded_shape)
    with self.model_state.global_mesh:
      # logging.info("Calling greedy decoding step")
      decode_state, decode_cache = self._per_bs_infos[padded_shape].device_fn[2](
        self.model_state.mdl_vars, 
        decode_state,
        # decode_cache,
        align_decode_state
      )
      # logging.info("model kv cache vars after running decoding_step: {}".format(
      #   self.model_state.mdl_vars['decoder_cache']['lm']['transformer']['x_layers_0']['self_attention']['key_state'].shape)
      # )
      # jax.block_until_ready(decode_state)
    return decode_state, decode_cache

  def device_compute_decoding_loop(
      self, input_batch: DeviceTensors, unpadded_shape: SaxServableModelInputShapeInfo, decode_state):
    padded_shape = self.get_padded_input_shape(unpadded_shape)
    with self.model_state.global_mesh:
      # logging.info("Calling greedy decoding loop")
      for i in range(64):
        decode_state, updated_vars = self._per_bs_infos[padded_shape].device_fn[2](
          self.model_state.mdl_vars, input_batch, decode_state
        )
        # logging.info("updated_vars in {} iteration: {}".format(
        #   i, 
        #   updated_vars['decoder_cache']['lm']['transformer']['x_layers_0']['self_attention']['key_state'].shape))
        self.model_state.mdl_vars.update(updated_vars)
    return decode_state
  
  def device_compute_process_result(
      self, 
      # input_batch: DeviceTensors, 
      unpadded_shape: SaxServableModelInputShapeInfo, 
      decode_state):
    padded_shape = self.get_padded_input_shape(unpadded_shape)
    with self.model_state.global_mesh:
      # logging.info("Calling greedy process result")
      output_batch = self._per_bs_infos[padded_shape].device_fn[3](
        self.model_state.mdl_vars, 
        # input_batch, 
        decode_state
      )
      return output_batch

  def device_compute_process_result_with_idx(
      self, 
      unpadded_shape: SaxServableModelInputShapeInfo, 
      decode_state,
      slot_idx
  ):
    padded_shape = self.get_padded_input_shape(unpadded_shape)
    with self.model_state.global_mesh:
      # logging.info("Calling greedy process result with idx".format(slot_idx))
      output_batch = self._per_bs_infos[padded_shape].device_fn[5](
        self.model_state.mdl_vars, 
        decode_state,
        slot_idx
      )
      return output_batch

  def device_compute(
      self, input_batch: DeviceTensors, unpadded_shape: SaxServableModelInputShapeInfo
      ) -> DeviceTensors:
    """Executes the device computation."""
    padded_shape = self.get_padded_input_shape(unpadded_shape)
    with self.model_state.global_mesh:
      # logging.info("Call continuous batching decode device fn")
      # output_batch = self._per_bs_infos[padded_shape].device_fn[0](
      #     self.model_state.mdl_vars, input_batch
      # )
      decode_data, updated_vars = self._per_bs_infos[padded_shape].device_fn[0](
        self.model_state.mdl_vars, input_batch
      )
      # self.model_state.mdl_vars.update(updated_vars)
      # logging.info("after calling greedy prefill")

      init_decode_state, updated_vars = self._per_bs_infos[padded_shape].device_fn[1](
        self.model_state.mdl_vars, input_batch, decode_data
      )
      self.model_state.mdl_vars.update(updated_vars)
      # logging.info("after calling greedy init decode state")

      decode_state = init_decode_state
      for i in range(64):
        decode_state, updated_vars = self._per_bs_infos[padded_shape].device_fn[2](
          self.model_state.mdl_vars, 
          # input_batch, 
          decode_state
        )
        # logging.info("updated_vars in {} iteration: {}".format(
        #   i, 
        #   updated_vars['decoder_cache']['lm']['transformer']['x_layers_0']['self_attention']['key_state'].shape))
        self.model_state.mdl_vars.update(updated_vars)
      # logging.info("after calling greedy decoding loop")

      output_batch = self._per_bs_infos[padded_shape].device_fn[3](
        self.model_state.mdl_vars, 
        # input_batch, 
        decode_state
      )
      # logging.info("after calling greedy process result, output_batch: {}".format(output_batch))
      return output_batch



  def init_decode_cache_and_shardings(self, num_layers):
    def _get_decode_pspec(x):
      # return jax.sharding.PartitionSpec(*(None,) * (len(x.shape)))
      return jax.sharding.PartitionSpec(None, None, ('mdl',), None)

    def _get_prefill_pspec(x):
      return jax.sharding.PartitionSpec(None, None, ('mdl',), None)
    
    # decode cache vars
    kv_cache = {}
    for i in range(num_layers):
      layer_kv_cache = {'x_layers_{}'.format(i): {'self_attention': {
        'key_state': jnp.zeros((1, 2048, 32, 128), dtype=jnp.bfloat16), 
        'value_state': jnp.zeros((1, 2048, 32, 128), dtype=jnp.bfloat16), 
        'key_post_rotary_pos_emb': jnp.zeros((1, 2048, 32, 128), dtype=jnp.bfloat16)}}}
      kv_cache.update(layer_kv_cache)
    decode_cache = {'decoder_cache': {'lm': {'time_step': 0, 'transformer': kv_cache}}}
    self.model_state.mdl_vars.update(decode_cache)

    # decode cache sharding
    time_step_partition_spec = jax.sharding.PartitionSpec()
    transformer_decode_parition_spec = jax.tree_util.tree_map(
      _get_decode_pspec, decode_cache[base_layer.DECODE_CACHE]['lm']['transformer'])
    transformer_prefill_partition_spec = jax.tree_util.tree_map(
      _get_prefill_pspec, decode_cache[base_layer.DECODE_CACHE]['lm']['transformer'])
    
    decode_cache_pspecs = {'decoder_cache': {'lm': {'time_step': time_step_partition_spec, 
                                                    'transformer': transformer_decode_parition_spec}}}
    prefill_cache_pspecs = {'decoder_cache': {'lm': {'time_step': time_step_partition_spec, 
                                                    'transformer': transformer_prefill_partition_spec}}}
    self.decode_cache_pspecs = decode_cache_pspecs
    self.prefill_cache_pspecs = prefill_cache_pspecs

    # self.model_state.mdl_var_pspecs.update(decode_cache_pspecs)

  # manually initialize decode state shardings
  def init_decode_state_shardings(self):
    pass

  def _register_for_input_shape(self, input_shape: SaxServableModelInputShapeInfo) -> None:
    # init shardings for decode_cache, decode_state
    self.init_decode_cache_and_shardings(32)
    self.init_decode_state_shardings()

    self._register_for_input_shape_sub(input_shape)
    
    assert self._method_hparams.decoder.enable_continuous_batching
    continuous_batching_batch_size = self._method_hparams.decoder.continuous_batching_batch_size
    logging.info("LMDecode method continuous batching size: {}".format(continuous_batching_batch_size))

    continuous_batching_input_shape = InputShapeInfo(
      continuous_batching_batch_size, input_shape.seq_len)
    if continuous_batching_input_shape != input_shape:
      self._register_for_input_shape_sub(continuous_batching_input_shape)

  def _register_for_input_shape_sub(self, input_shape: SaxServableModelInputShapeInfo) -> None:
    logging.info("Register continuous batching decoding method, input_shape: {}".format(
      input_shape))

    batched_host_dummy = self.get_dummy_inputs(input_shape)
    batched_host_dummy = self.update_extra_inputs(
        batched_host_dummy,
        input_shape.batch_size,
        [self.default_extra_inputs] * input_shape.batch_size,
    )

    def _assert_type(x):
      assert isinstance(x, np.ndarray) or isinstance(
          x, jnp.ndarray
      ), f'Output of pre_processing contained an invalid type: {type(x)}'
      return x

    dummy_step = np.array(0, dtype=np.int32)
    host_dummy = (
        dummy_step,
        batched_host_dummy,
        self.get_nonbatch_inputs(batched_host_dummy),
    )
    host_dummy = jax.tree_util.tree_map(_assert_type, host_dummy)

    def _get_pspec(x):
      # Add a `cores` dimension.
      return jax.sharding.PartitionSpec(
          self.model_state.global_mesh.axis_names, *(None,) * (len(x.shape))
      )

    input_pspecs = jax.tree_util.tree_map(_get_pspec, host_dummy)
    num_cores = len(self.model_state.global_mesh.devices.flat)

    global_inputs_shape_dtype = jax.tree_util.tree_map(
        lambda x: ((num_cores,) + x.shape, x.dtype), host_dummy
    )
    global_inputs_shaped_arrays = jax.tree_util.tree_map(
        lambda x: jax.core.ShapedArray((num_cores,) + x.shape, x.dtype),
        host_dummy,
    )
    # logging.info("global_inputs_shape_dtype: {}".format(global_inputs_shape_dtype[1]))
    # logging.info("input_shape: {}, batched_host_dummy: {}".format(input_shape, batched_host_dummy))

    self._per_bs_infos[input_shape] = MethodInputInfo(
        input_pspecs=input_pspecs,
        global_inputs_shape_dtype=global_inputs_shape_dtype,
    )
    info = self._per_bs_infos[input_shape]
    info.host_dummy_inputs = batched_host_dummy

    info.dummy_inputs_per_device_buffers = self._input_to_device_buffers(
        batched_host_dummy, input_shape, is_dummy=True
    )
    # logging.info("Finished calling input_to_device_buffers: {}".format(input_shape))
    info.dummy_inputs = self._device_buffers_to_jax_arrays(
        info.dummy_inputs_per_device_buffers, input_shape
    )

    # self.init_decode_cache_and_shardings()
    logging.info("Partition specs mdl_vars: {}".format(self._model_state.mdl_var_pspecs))
    logging.info("Partition specs input: {}".format(input_pspecs))

    # Initialize the device function.
    greedy_prefill_device_fn = self._pjit_device_fn_greedy_prefill(
      self.greedy_prefill_jax_func, input_pspecs, input_shape.batch_size)

    greedy_init_decode_state_device_fn = self._pjit_device_fn_greedy_init_decode_state(
      self.greedy_init_decode_state_jax_func, input_pspecs, input_shape.batch_size
    )

    greedy_decode_step_device_fn = self._pjit_device_fn_greedy_decode_step(
      self.greedy_decode_step_jax_func, input_pspecs, input_shape.batch_size
    )

    # greedy_decode_device_fn = self._pjit_device_fn_greedy_decode(
    #   self.greedy_decode_jax_func, input_pspecs, input_shape.batch_size
    # )

    greedy_process_result_device_fn = self._pjit_device_fn_greedy_process_result(
      self.greedy_process_result_jax_func, input_pspecs, input_shape.batch_size
    )

    greedy_prefill_and_insert_device_fn = self._pjit_device_fn_prefill_and_insert(
      self.greedy_prefill_and_insert_jax_func, input_pspecs, input_shape.batch_size
    )

    greedy_process_result_with_idx_device_fn = self._pjit_device_fn_greedy_process_result_with_idx(
      self.greedy_process_result_with_idx_jax_func, input_pspecs, input_shape.batch_size
    )

    # device_fn = self._pjit_device_fn(input_pspecs, input_shape.batch_size)

    if self._enable_auto_sharding or self._compiler_options:
      device_fn = self.jax_aot_compile(
          step_fn=device_fn,
          train_state=self.model_state,
          inputs_shape_dtype=global_inputs_shaped_arrays,
          compiler_options=self._compiler_options,
      )

    info.device_fn = [greedy_prefill_device_fn, greedy_init_decode_state_device_fn, 
                      greedy_decode_step_device_fn, greedy_process_result_device_fn,
                      greedy_prefill_and_insert_device_fn,
                      greedy_process_result_with_idx_device_fn]
    # info.device_fn = [device_fn]

    # # Compute with dummy to trigger compilation.
    # if self.model_state.precompile:
    #   init_dummy_outputs = self.device_compute(info.dummy_inputs, input_shape)

    # if self.model_state.is_primary_host:
    #   # Transfer dummy to host to block until dummy computation is done.
    #   if self.model_state.precompile:
    #     # Retrieve streamed outputs until streaming is done
    #     if self.streamable:
    #       stream_state = None
    #       while True:
    #         stream_outs = self.dequeue_stream_output()
    #         _, stream_state = self.post_processing_stream(
    #             stream_outs, stream_state
    #         )
    #         if stream_outs is None:
    #           break
    #     outs = self.output_to_host(init_dummy_outputs, self.batch_size)
    #     if not self.streamable:
    #       # Warm up post processor.
    #       self.post_processing(outs)
    logging.info("Finished register LMDecode continuous batching input_shape: {}".format(input_shape))


  # greedy prefill and insert pjit functions
  def _pjit_device_fn_prefill_and_insert(
    self, jax_func, input_pspecs: PSpecs, batch_size: int
  )-> Callable[[DeviceTensors, DeviceTensors, DeviceTensors, DeviceTensors], DeviceTensors]:
    """Returns a pjit-ed model function with input handling."""
    def _wrapped_fn_greedy_prefill_and_insert(
        mdl_vars, inputs, decode_state, 
        # decode_cache, 
        slot_idx):
      # Remove padding on the vars.
      # mdl_vars = jax.tree_util.tree_map(
      #     remove_padding, mdl_vars, self.model_state.mdl_var_unpadded_shapes
      # )
      mdl_vars = jax.tree_util.tree_map(
          jax.lax.with_sharding_constraint,
          mdl_vars,
          self.model_state.mdl_var_pspecs,
      )

      # Only one core has real data, others have zeros. Summing on the
      # leading `cores` dimension can make data replicated.
      def _replicate(x):
        return jax.lax.with_sharding_constraint(
            jnp.sum(x, axis=0, promote_integers=False), None
        )

      inputs = jax.tree_util.tree_map(_replicate, inputs)
      step, batched_inputs, non_batched_inputs = inputs
      prng_key = jax.random.fold_in(self._prng_key, step)
      outputs = jax_func(
          mdl_vars, prng_key, batched_inputs, 
          # non_batched_inputs,
          # decode_input_batch, 
          decode_state, 
          # decode_cache, 
          slot_idx
      )
      # This assumes that outputs are generated after previous host calls, and
      # it is guaranteed by data dependency.
      if self.streamable:

        def _mark_done(dummy, _):
          del dummy
          self.mark_stream_output_done()

        hcb.id_tap(_mark_done, outputs, device_index=self.callback_device_index)
        # Unused final outputs. Return something to make output_to_host
        # blocking.
        return jnp.zeros((batch_size,), dtype=jnp.int32)

      return outputs

    # pjit-ed function.
    self.model_state.mdl_var_pspecs.update(self.decode_cache_pspecs)
    if self._enable_auto_sharding:
      return pjit.pjit(
          _wrapped_fn_greedy_prefill_and_insert,
          in_shardings=(pjit.AUTO(self.model_state.global_mesh), input_pspecs, None, None),
          out_shardings=None,
      )
    else:
      return pjit.pjit(
          _wrapped_fn_greedy_prefill_and_insert,
          in_shardings=(self.model_state.mdl_var_pspecs, 
                        input_pspecs, 
                        None, 
                        # self.decode_cache_pspecs, 
                        None),
          out_shardings=(None, self.decode_cache_pspecs),
      )

  def greedy_prefill_and_insert_jax_func(
      self,
      mdl_vars: NestedJTensor,
      prng_key: PRNGKey,
      batched_inputs: NestedJTensor,
      # non_batched_inputs: NestedJTensor,
      # decode_input_batch: NestedJTensor,
      decode_state: NestedJTensor,
      # decode_cache: NestedJTensor,
      slot_idx
  ) -> NestedJTensor:
    if self._model.fprop_dtype == jnp.bfloat16:
      # Convert float inputs/vars if fprop dtype is bfloat16.
      batched_inputs, mdl_vars = jax.tree_map(
          (lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x),
          (batched_inputs, mdl_vars),
      )

    context_p = base_layer.JaxContext.HParams(do_eval=True)
    k1, k2 = jax.random.split(prng_key)
    with base_layer.JaxContext.new_context(hparams=context_p):
      def _model_fn(inputs):
        outputs = self.call_model_function_greedy_prefill_and_insert(
          inputs, decode_state, 
          # decode_cache, 
          slot_idx,
          mdl_vars, [k1, k2])  # pytype: disable=wrong-arg-types  # jax-ndarray
        return outputs
      decode_state, decode_cache = _model_fn(batched_inputs)
      # logging.info("prefill and insert return decode cache: {}".format(decode_cache))
      # if base_layer.DECODE_CACHE in updated_vars:
      #   del updated_vars[base_layer.DECODE_CACHE]
      # if base_layer.PREFIX_DECODE_CACHE in decode_cache:
      #   del decode_cache[base_layer.PREFIX_DECODE_CACHE]
      
      return decode_state, decode_cache

  def call_model_function_greedy_prefill_and_insert(
      self, inputs, decode_state, 
      # decode_cache, 
      slot_idx, mdl_vars, prng_key):
    k1, k2 = prng_key

    kwargs = {}
    if self.streamable:

      def callback_fn(x, _):
        assert self.model_state.is_primary_host
        self.enqueue_stream_output(x)

      kwargs['result_callback'] = decoder_utils.StreamingResultCallback(
          functools.partial(
              hcb.id_tap, callback_fn, device_index=self.callback_device_index
          ),
          interval_steps=self._method_hparams.stream_interval_steps,
      )

    if (
        'callback_device_index'
        in inspect.signature(self._model.decode_with_params).parameters
    ):
      kwargs['callback_device_index'] = self.callback_device_index

    decode_state, decode_cache = self._model.apply(
        mdl_vars,
        input_batch=inputs,
        slot_idx=slot_idx,
        method=self._model.greedy_prefill_and_insert,
        mutable=[
            base_layer.NON_TRAINABLE,
            base_layer.DECODE_CACHE,
            base_layer.PREFIX_DECODE_CACHE,
        ],
        rngs={
            base_layer.PARAMS: k1,
            base_layer.RANDOM: k2,
        },
        decoder_params=self._method_hparams.decoder,
        decode_state=decode_state,
        # decode_cache=decode_cache,
        **kwargs,
    )
    return decode_state, decode_cache


  # greedy prefill pjit functions
  def _pjit_device_fn_greedy_prefill(
      self, jax_func, input_pspecs: PSpecs, batch_size: int
  ) -> Callable[[DeviceTensors, DeviceTensors], DeviceTensors]:
    """Returns a pjit-ed model function with input handling."""

    def _wrapped_fn_greedy_prefill(mdl_vars, inputs):
      # Remove padding on the vars.
      # mdl_vars = jax.tree_util.tree_map(
      #     remove_padding, mdl_vars, self.model_state.mdl_var_unpadded_shapes
      # )
      mdl_vars = jax.tree_util.tree_map(
          jax.lax.with_sharding_constraint,
          mdl_vars,
          self.model_state.mdl_var_pspecs,
      )

      # Only one core has real data, others have zeros. Summing on the
      # leading `cores` dimension can make data replicated.
      def _replicate(x):
        return jax.lax.with_sharding_constraint(
            jnp.sum(x, axis=0, promote_integers=False), None
        )

      inputs = jax.tree_util.tree_map(_replicate, inputs)
      step, batched_inputs, non_batched_inputs = inputs
      prng_key = jax.random.fold_in(self._prng_key, step)
      outputs = jax_func(
          mdl_vars, prng_key, batched_inputs, non_batched_inputs
      )
      # This assumes that outputs are generated after previous host calls, and
      # it is guaranteed by data dependency.
      if self.streamable:

        def _mark_done(dummy, _):
          del dummy
          self.mark_stream_output_done()

        hcb.id_tap(_mark_done, outputs, device_index=self.callback_device_index)
        # Unused final outputs. Return something to make output_to_host
        # blocking.
        return jnp.zeros((batch_size,), dtype=jnp.int32)

      return outputs

    # pjit-ed function.
    self.model_state.mdl_var_pspecs.update(self.prefill_cache_pspecs)
    if self._enable_auto_sharding:
      return pjit.pjit(
          _wrapped_fn_greedy_prefill,
          in_shardings=(pjit.AUTO(self.model_state.global_mesh), input_pspecs),
          out_shardings=None,
      )
    else:
      return pjit.pjit(
          _wrapped_fn_greedy_prefill,
          in_shardings=(self.model_state.mdl_var_pspecs, input_pspecs),
          out_shardings=(None, self.decode_cache_pspecs),
      )

  def greedy_prefill_jax_func(
      self,
      mdl_vars: NestedJTensor,
      prng_key: PRNGKey,
      batched_inputs: NestedJTensor,
      non_batched_inputs: NestedJTensor,
  ) -> NestedJTensor:
    if self._model.fprop_dtype == jnp.bfloat16:
      # Convert float inputs/vars if fprop dtype is bfloat16.
      batched_inputs, mdl_vars = jax.tree_map(
          (lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x),
          (batched_inputs, mdl_vars),
      )

    context_p = base_layer.JaxContext.HParams(do_eval=True)
    k1, k2 = jax.random.split(prng_key)
    with base_layer.JaxContext.new_context(hparams=context_p):

      def _model_fn(inputs):
        outputs = self.call_model_function_greedy_prefill(inputs, mdl_vars, [k1, k2])  # pytype: disable=wrong-arg-types  # jax-ndarray
        return outputs

      if isinstance(non_batched_inputs, tuple) and not non_batched_inputs:
        outputs = _model_fn(batched_inputs)
      else:

        def _build_branch_model_fn(branch_key):
          """Gets model_fn for each branch."""

          def _branch_model_fn(inputs):
            branch_inputs = self.get_branch_inputs(inputs, branch_key)
            branch_outputs = _model_fn(branch_inputs)
            return self.post_process_branch_outputs(branch_outputs, branch_key)

          return _branch_model_fn

        branch_fns = [
            _build_branch_model_fn(branch_key)
            for branch_key in self._branch_selector.branch_keys
        ]
        branch_index = non_batched_inputs
        outputs = jax.lax.switch(branch_index, branch_fns, batched_inputs)

    return outputs

  def call_model_function_greedy_prefill(self, inputs, mdl_vars, prng_key):
    k1, k2 = prng_key

    kwargs = {}
    if self.streamable:

      def callback_fn(x, _):
        assert self.model_state.is_primary_host
        self.enqueue_stream_output(x)

      kwargs['result_callback'] = decoder_utils.StreamingResultCallback(
          functools.partial(
              hcb.id_tap, callback_fn, device_index=self.callback_device_index
          ),
          interval_steps=self._method_hparams.stream_interval_steps,
      )

    if (
        'callback_device_index'
        in inspect.signature(self._model.decode_with_params).parameters
    ):
      kwargs['callback_device_index'] = self.callback_device_index

    # prefill
    decode_data, updated_vars = self._model.apply(
        mdl_vars,
        input_batch=inputs,
        method=self._model.greedy_prefill,
        mutable=[
            base_layer.NON_TRAINABLE,
            base_layer.DECODE_CACHE,
            base_layer.PREFIX_DECODE_CACHE,
        ],
        rngs={
            base_layer.PARAMS: k1,
            base_layer.RANDOM: k2,
        },
        decoder_params=self._method_hparams.decoder,
        **kwargs,
    )
    return decode_data, updated_vars


  # greedy init decode state pjit functions
  def _pjit_device_fn_greedy_init_decode_state(
      self, jax_func, input_pspecs: PSpecs, batch_size: int
  ) -> Callable[[DeviceTensors, DeviceTensors, DeviceTensors], DeviceTensors]:
    """Returns a pjit-ed model function with input handling."""

    def _wrapped_fn_greedy_init_decode_state(mdl_vars, inputs, decode_data):
      # Remove padding on the vars.
      # mdl_vars = jax.tree_util.tree_map(
      #     remove_padding, mdl_vars, self.model_state.mdl_var_unpadded_shapes
      # )
      mdl_vars = jax.tree_util.tree_map(
          jax.lax.with_sharding_constraint,
          mdl_vars,
          self.model_state.mdl_var_pspecs,
      )

      # Only one core has real data, others have zeros. Summing on the
      # leading `cores` dimension can make data replicated.
      def _replicate(x):
        return jax.lax.with_sharding_constraint(
            jnp.sum(x, axis=0, promote_integers=False), None
        )

      # logging.info("before calling init decode state replicate")
      inputs = jax.tree_util.tree_map(_replicate, inputs)
      step, batched_inputs, non_batched_inputs = inputs
      prng_key = jax.random.fold_in(self._prng_key, step)
      # logging.info("before calling init decode state jax func")
      outputs = jax_func(
          mdl_vars, prng_key, batched_inputs, non_batched_inputs, decode_data
      )
      # This assumes that outputs are generated after previous host calls, and
      # it is guaranteed by data dependency.
      if self.streamable:

        def _mark_done(dummy, _):
          del dummy
          self.mark_stream_output_done()

        hcb.id_tap(_mark_done, outputs, device_index=self.callback_device_index)
        # Unused final outputs. Return something to make output_to_host
        # blocking.
        return jnp.zeros((batch_size,), dtype=jnp.int32)

      return outputs

    # pjit-ed function.
    self.model_state.mdl_var_pspecs.update(self.decode_cache_pspecs)
    if self._enable_auto_sharding:
      return pjit.pjit(
          _wrapped_fn_greedy_init_decode_state,
          in_shardings=(pjit.AUTO(self.model_state.global_mesh), input_pspecs, None),
          out_shardings=None,
      )
    else:
      return pjit.pjit(
          _wrapped_fn_greedy_init_decode_state,
          in_shardings=(self.model_state.mdl_var_pspecs, input_pspecs, None),
          out_shardings=(None, self.decode_cache_pspecs),
      )

  def greedy_init_decode_state_jax_func(
      self,
      mdl_vars: NestedJTensor,
      prng_key: PRNGKey,
      batched_inputs: NestedJTensor,
      non_batched_inputs: NestedJTensor,
      decode_data: NestedJTensor
  ) -> NestedJTensor:
    if self._model.fprop_dtype == jnp.bfloat16:
      # Convert float inputs/vars if fprop dtype is bfloat16.
      batched_inputs, mdl_vars = jax.tree_map(
          (lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x),
          (batched_inputs, mdl_vars),
      )

    context_p = base_layer.JaxContext.HParams(do_eval=True)
    k1, k2 = jax.random.split(prng_key)
    with base_layer.JaxContext.new_context(hparams=context_p):

      def _model_fn(inputs):
        outputs = self.call_model_function_greedy_init_decode_state(
          inputs, decode_data, mdl_vars, [k1, k2])  # pytype: disable=wrong-arg-types  # jax-ndarray
        return outputs

      if isinstance(non_batched_inputs, tuple) and not non_batched_inputs:
        # logging.info("before calling init decode state model_fn")
        outputs = _model_fn(batched_inputs)
      else:

        def _build_branch_model_fn(branch_key):
          """Gets model_fn for each branch."""

          def _branch_model_fn(inputs):
            branch_inputs = self.get_branch_inputs(inputs, branch_key)
            branch_outputs = _model_fn(branch_inputs)
            return self.post_process_branch_outputs(branch_outputs, branch_key)

          return _branch_model_fn

        branch_fns = [
            _build_branch_model_fn(branch_key)
            for branch_key in self._branch_selector.branch_keys
        ]
        branch_index = non_batched_inputs
        outputs = jax.lax.switch(branch_index, branch_fns, batched_inputs)

    return outputs

  def call_model_function_greedy_init_decode_state(self, inputs, decode_data, mdl_vars, prng_key):
    k1, k2 = prng_key

    kwargs = {}
    if self.streamable:

      def callback_fn(x, _):
        assert self.model_state.is_primary_host
        self.enqueue_stream_output(x)

      kwargs['result_callback'] = decoder_utils.StreamingResultCallback(
          functools.partial(
              hcb.id_tap, callback_fn, device_index=self.callback_device_index
          ),
          interval_steps=self._method_hparams.stream_interval_steps,
      )

    if (
        'callback_device_index'
        in inspect.signature(self._model.decode_with_params).parameters
    ):
      kwargs['callback_device_index'] = self.callback_device_index

    # initialize decoding state
    # logging.info("before calling init decode state apply func")
    init_decode_state, updated_vars = self._model.apply(
        mdl_vars,
        input_batch=inputs,
        decode_data=decode_data,
        method=self._model.greedy_init_decode_state,
        mutable=[
            base_layer.NON_TRAINABLE,
            base_layer.DECODE_CACHE,
            base_layer.PREFIX_DECODE_CACHE,
        ],
        rngs={
            base_layer.PARAMS: k1,
            base_layer.RANDOM: k2,
        },
        decoder_params=self._method_hparams.decoder,
        **kwargs,
    )
    return init_decode_state, updated_vars


  # greedy decode single step pjit functions
  def _pjit_device_fn_greedy_decode_step(
      self, jax_func, input_pspecs: PSpecs, batch_size: int
  ) -> Callable[[DeviceTensors, DeviceTensors, bool], DeviceTensors]:
    """Returns a pjit-ed model function with input handling."""

    def _wrapped_fn_greedy_decode_step(mdl_vars, decode_state, align_decode_state):
      # Remove padding on the vars.
      # mdl_vars = jax.tree_util.tree_map(
      #     remove_padding, mdl_vars, self.model_state.mdl_var_unpadded_shapes
      # )
      # mdl_vars = jax.tree_util.tree_map(
      #     jax.lax.with_sharding_constraint,
      #     mdl_vars,
      #     self.model_state.mdl_var_pspecs,
      # )

      # Only one core has real data, others have zeros. Summing on the
      # leading `cores` dimension can make data replicated.
      # def _replicate(x):
      #   return jax.lax.with_sharding_constraint(
      #       jnp.sum(x, axis=0, promote_integers=False), None
      #   )

      # inputs = jax.tree_util.tree_map(_replicate, inputs)
      # step, batched_inputs, non_batched_inputs = inputs
      # prng_key = jax.random.fold_in(self._prng_key, step)
      prng_key = self._prng_key
      outputs = jax_func(
          mdl_vars, prng_key, decode_state, 
          # decode_cache,
          align_decode_state
      )

      return outputs

    # pjit-ed function.
    self.model_state.mdl_var_pspecs.update(self.decode_cache_pspecs)
    return pjit.pjit(
        _wrapped_fn_greedy_decode_step,
        in_shardings=(self.model_state.mdl_var_pspecs, None),
        out_shardings=(None, self.decode_cache_pspecs),
        static_argnums=2
        # out_shardings=None,
    )

  def greedy_decode_step_jax_func(
      self,
      mdl_vars: NestedJTensor,
      prng_key: PRNGKey,
      decode_state: NestedJTensor,
      # decode_cache: NestedJTensor,
      align_decode_state
  ) -> NestedJTensor:
    # if self._model.fprop_dtype == jnp.bfloat16:
    #   # Convert float inputs/vars if fprop dtype is bfloat16.
    #   # batched_inputs, mdl_vars = jax.tree_map(
    #   #     (lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x),
    #   #     (batched_inputs, mdl_vars),
    #   # )
    #   mdl_vars = jax.tree_map(
    #       (lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x),
    #       (mdl_vars),
    #   )

    context_p = base_layer.JaxContext.HParams(do_eval=True)
    k1, k2 = jax.random.split(prng_key)
    with base_layer.JaxContext.new_context(hparams=context_p):
      def _model_fn(decode_state):
        outputs = self.call_model_function_greedy_decode_step(
          decode_state, 
          # decode_cache, 
          align_decode_state,
          mdl_vars, [k1, k2])  # pytype: disable=wrong-arg-types  # jax-ndarray
        return outputs

      outputs = _model_fn(decode_state)
      # _model_fn(decode_state)
    return outputs

  def call_model_function_greedy_decode_step(
    self, decode_state, 
    # decode_cache, 
    align_decode_state,
    mdl_vars, prng_key):
    k1, k2 = prng_key

    kwargs = {}
    if (
        'callback_device_index'
        in inspect.signature(self._model.decode_with_params).parameters
    ):
      kwargs['callback_device_index'] = self.callback_device_index

    # decode single step
    decode_state, decode_cache = self._model.apply(
        mdl_vars,
        decode_state=decode_state,
        # decode_cache=decode_cache,
        align_decode_state=align_decode_state,
        method=self._model.greedy_decode_step,
        mutable=[
            base_layer.NON_TRAINABLE,
            base_layer.DECODE_CACHE,
            base_layer.PREFIX_DECODE_CACHE,
        ],
        rngs={
            base_layer.PARAMS: k1,
            base_layer.RANDOM: k2,
        },
        decoder_params=self._method_hparams.decoder,
        **kwargs,
    )
    return decode_state, decode_cache


  # greedy decode pjit functions
  def _pjit_device_fn_greedy_decode(
      self, jax_func, input_pspecs: PSpecs, batch_size: int
  ) -> Callable[[DeviceTensors, DeviceTensors, DeviceTensors], DeviceTensors]:
    """Returns a pjit-ed model function with input handling."""

    def _wrapped_fn(mdl_vars, inputs, decode_data):
      # Remove padding on the vars.
      # mdl_vars = jax.tree_util.tree_map(
      #     remove_padding, mdl_vars, self.model_state.mdl_var_unpadded_shapes
      # )
      mdl_vars = jax.tree_util.tree_map(
          jax.lax.with_sharding_constraint,
          mdl_vars,
          self.model_state.mdl_var_pspecs,
      )

      # Only one core has real data, others have zeros. Summing on the
      # leading `cores` dimension can make data replicated.
      def _replicate(x):
        return jax.lax.with_sharding_constraint(
            jnp.sum(x, axis=0, promote_integers=False), None
        )

      inputs = jax.tree_util.tree_map(_replicate, inputs)
      step, batched_inputs, non_batched_inputs = inputs
      prng_key = jax.random.fold_in(self._prng_key, step)
      outputs = jax_func(
          mdl_vars, prng_key, batched_inputs, non_batched_inputs, decode_data
      )
      # This assumes that outputs are generated after previous host calls, and
      # it is guaranteed by data dependency.
      if self.streamable:

        def _mark_done(dummy, _):
          del dummy
          self.mark_stream_output_done()

        hcb.id_tap(_mark_done, outputs, device_index=self.callback_device_index)
        # Unused final outputs. Return something to make output_to_host
        # blocking.
        return jnp.zeros((batch_size,), dtype=jnp.int32)

      return outputs

    # pjit-ed function.
    if self._enable_auto_sharding:
      return pjit.pjit(
          _wrapped_fn,
          in_shardings=(pjit.AUTO(self.model_state.global_mesh), input_pspecs, None),
          out_shardings=None,
      )
    else:
      return pjit.pjit(
          _wrapped_fn,
          in_shardings=(self.model_state.mdl_var_pspecs, input_pspecs, None),
          out_shardings=(None, self.decode_cache_pspecs),
      )

  def greedy_decode_jax_func(
      self,
      mdl_vars: NestedJTensor,
      prng_key: PRNGKey,
      batched_inputs: NestedJTensor,
      non_batched_inputs: NestedJTensor,
      decode_data: NestedJTensor,
  ) -> NestedJTensor:
    if self._model.fprop_dtype == jnp.bfloat16:
      # Convert float inputs/vars if fprop dtype is bfloat16.
      batched_inputs, mdl_vars = jax.tree_map(
          (lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x),
          (batched_inputs, mdl_vars),
      )

    context_p = base_layer.JaxContext.HParams(do_eval=True)
    k1, k2 = jax.random.split(prng_key)
    with base_layer.JaxContext.new_context(hparams=context_p):

      def _model_fn(inputs):
        outputs = self.call_model_function_greedy_decode(
          inputs, decode_data, mdl_vars, [k1, k2])  # pytype: disable=wrong-arg-types  # jax-ndarray
        return outputs

      if isinstance(non_batched_inputs, tuple) and not non_batched_inputs:
        outputs = _model_fn(batched_inputs)
      else:

        def _build_branch_model_fn(branch_key):
          """Gets model_fn for each branch."""

          def _branch_model_fn(inputs):
            branch_inputs = self.get_branch_inputs(inputs, branch_key)
            branch_outputs = _model_fn(branch_inputs)
            return self.post_process_branch_outputs(branch_outputs, branch_key)

          return _branch_model_fn

        branch_fns = [
            _build_branch_model_fn(branch_key)
            for branch_key in self._branch_selector.branch_keys
        ]
        branch_index = non_batched_inputs
        outputs = jax.lax.switch(branch_index, branch_fns, batched_inputs)

      if (
          self.method_params.cast_bfloat16_outputs
          and self._model.fprop_dtype == jnp.bfloat16
      ):
        # Convert bfloat16 back to float32.
        def maybe_to_float32(x):
          x = jnp.asarray(x)
          if x.dtype == jnp.bfloat16:
            return x.astype(jnp.float32)
          return x

        outputs = jax.tree_map(maybe_to_float32, outputs)
      return outputs

  def call_model_function_greedy_decode(self, inputs, decode_data, mdl_vars, prng_key):
    k1, k2 = prng_key

    kwargs = {}
    if self.streamable:

      def callback_fn(x, _):
        assert self.model_state.is_primary_host
        self.enqueue_stream_output(x)

      kwargs['result_callback'] = decoder_utils.StreamingResultCallback(
          functools.partial(
              hcb.id_tap, callback_fn, device_index=self.callback_device_index
          ),
          interval_steps=self._method_hparams.stream_interval_steps,
      )

    if (
        'callback_device_index'
        in inspect.signature(self._model.decode_with_params).parameters
    ):
      kwargs['callback_device_index'] = self.callback_device_index

    # initialize decoding state
    init_decode_state, updated_vars = self._model.apply(
        mdl_vars,
        input_batch=inputs,
        decode_data=decode_data,
        method=self._model.greedy_init_decode_state,
        mutable=[
            base_layer.NON_TRAINABLE,
            base_layer.DECODE_CACHE,
            base_layer.PREFIX_DECODE_CACHE,
        ],
        rngs={
            base_layer.PARAMS: k1,
            base_layer.RANDOM: k2,
        },
        decoder_params=self._method_hparams.decoder,
        **kwargs,
    )
    mdl_vars.update(updated_vars)

    # decoding while loop
    decode_state, updated_vars = self._model.apply(
        mdl_vars,
        decode_state=init_decode_state,
        method=self._model.greedy_decode_step_while,
        mutable=[
            base_layer.NON_TRAINABLE,
            base_layer.DECODE_CACHE,
            base_layer.PREFIX_DECODE_CACHE,
        ],
        rngs={
            base_layer.PARAMS: k1,
            base_layer.RANDOM: k2,
        },
        **kwargs,
    )
    return decode_state, updated_vars


  # process results pjit function
  def _pjit_device_fn_greedy_process_result(
      self, jax_func, input_pspecs: PSpecs, batch_size: int
  ) -> Callable[[DeviceTensors, DeviceTensors], DeviceTensors]:
    """Returns a pjit-ed model function with input handling."""

    def _wrapped_fn_greedy_process_result(mdl_vars, decode_state):
      # Remove padding on the vars.
      # mdl_vars = jax.tree_util.tree_map(
      #     remove_padding, mdl_vars, self.model_state.mdl_var_unpadded_shapes
      # )
      mdl_vars = jax.tree_util.tree_map(
          jax.lax.with_sharding_constraint,
          mdl_vars,
          self.model_state.mdl_var_pspecs,
      )

      # Only one core has real data, others have zeros. Summing on the
      # leading `cores` dimension can make data replicated.
      # def _replicate(x):
      #   return jax.lax.with_sharding_constraint(
      #       jnp.sum(x, axis=0, promote_integers=False), None
      #   )

      # inputs = jax.tree_util.tree_map(_replicate, inputs)
      # _, batched_inputs, _ = inputs
      # prng_key = jax.random.fold_in(self._prng_key, step)
      prng_key = self._prng_key
      outputs = jax_func(
          mdl_vars, prng_key, 
          # batched_inputs, 
          decode_state
      )
      # This assumes that outputs are generated after previous host calls, and
      # it is guaranteed by data dependency.
      if self.streamable:

        def _mark_done(dummy, _):
          del dummy
          self.mark_stream_output_done()

        hcb.id_tap(_mark_done, outputs, device_index=self.callback_device_index)
        # Unused final outputs. Return something to make output_to_host
        # blocking.
        return jnp.zeros((batch_size,), dtype=jnp.int32)

      return outputs

    # pjit-ed function.
    if self._enable_auto_sharding:
      return pjit.pjit(
          _wrapped_fn_greedy_process_result,
          in_shardings=(pjit.AUTO(self.model_state.global_mesh), None),
          out_shardings=None,
      )
    else:
      return pjit.pjit(
          _wrapped_fn_greedy_process_result,
          in_shardings=(self.model_state.mdl_var_pspecs, None),
          out_shardings=None,
      )
    
  def fetch_output(
      self, model_fn_outputs: NestedJTensor
  ) -> NestedJTensor:
    model_fn_inputs = model_fn_outputs
    fetched_output = servable_lm_common.decode_fetch_output(
        model_fn_outputs,
        model_fn_inputs,
        self._method_hparams.t5_model,
        self._method_hparams.fetch_prefix_lengths_from_inputs,
    )
    fetched_output.prefix_lengths = jnp.squeeze(fetched_output.prefix_lengths, axis=1)
    fetched_output.scores = jnp.squeeze(fetched_output.scores, axis=2)
    logging.info("fetched_output: {}".format(fetched_output))
    return fetched_output

  def greedy_process_result_jax_func(
      self,
      mdl_vars: NestedJTensor,
      prng_key: PRNGKey,
      # batched_inputs: NestedJTensor,
      decode_state: NestedJTensor,
  ) -> NestedJTensor:
    if self._model.fprop_dtype == jnp.bfloat16:
      # Convert float inputs/vars if fprop dtype is bfloat16.
      # batched_inputs, mdl_vars = jax.tree_map(
      #     (lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x),
      #     (batched_inputs, mdl_vars),
      # )
      mdl_vars = jax.tree_map(
          (lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x),
          (mdl_vars),
      )

    context_p = base_layer.JaxContext.HParams(do_eval=True)
    k1, k2 = jax.random.split(prng_key)
    with base_layer.JaxContext.new_context(hparams=context_p):

      def _model_fn(decode_state):
        outputs = self.call_model_function_greedy_process_result(
          # inputs,
          decode_state, mdl_vars, [k1, k2])  # pytype: disable=wrong-arg-types  # jax-ndarray
        # DECODE_CACHE are not read by caller. But they can be large. Tell XLA
        # to remove it from output. Note MLP decoder don't have DECODE_CACHE.
        updated_vars = outputs[1]
        if base_layer.DECODE_CACHE in updated_vars:
          del updated_vars[base_layer.DECODE_CACHE]
        if base_layer.PREFIX_DECODE_CACHE in updated_vars:
          del updated_vars[base_layer.PREFIX_DECODE_CACHE]
        return outputs

      outputs = _model_fn(decode_state)

      if (
          self.method_params.cast_bfloat16_outputs
          and self._model.fprop_dtype == jnp.bfloat16
      ):
        # Convert bfloat16 back to float32.
        def maybe_to_float32(x):
          x = jnp.asarray(x)
          if x.dtype == jnp.bfloat16:
            return x.astype(jnp.float32)
          return x

        outputs = jax.tree_map(maybe_to_float32, outputs)
      return self.fetch_output(outputs)

  def call_model_function_greedy_process_result(self, decode_state, mdl_vars, prng_key):
    k1, k2 = prng_key

    kwargs = {}
    if self.streamable:

      def callback_fn(x, _):
        assert self.model_state.is_primary_host
        self.enqueue_stream_output(x)

      kwargs['result_callback'] = decoder_utils.StreamingResultCallback(
          functools.partial(
              hcb.id_tap, callback_fn, device_index=self.callback_device_index
          ),
          interval_steps=self._method_hparams.stream_interval_steps,
      )

    if (
        'callback_device_index'
        in inspect.signature(self._model.decode_with_params).parameters
    ):
      kwargs['callback_device_index'] = self.callback_device_index

    # process results
    outputs = self._model.apply(
        mdl_vars,
        decode_state=decode_state,
        # input_batch=inputs,
        method=self._model.greedy_process_result,
        mutable=[
            base_layer.NON_TRAINABLE,
            base_layer.DECODE_CACHE,
            base_layer.PREFIX_DECODE_CACHE,
        ],
        rngs={
            base_layer.PARAMS: k1,
            base_layer.RANDOM: k2,
        },
        decoder_params=self._method_hparams.decoder,
        **kwargs,
    )
    return outputs  


  # original batch decoding
  def call_model_function(self, inputs, mdl_vars, prng_key):
    k1, k2 = prng_key

    kwargs = {}
    if self.streamable:

      def callback_fn(x, _):
        assert self.model_state.is_primary_host
        self.enqueue_stream_output(x)

      kwargs['result_callback'] = decoder_utils.StreamingResultCallback(
          functools.partial(
              hcb.id_tap, callback_fn, device_index=self.callback_device_index
          ),
          interval_steps=self._method_hparams.stream_interval_steps,
      )

    if (
        'callback_device_index'
        in inspect.signature(self._model.decode_with_params).parameters
    ):
      kwargs['callback_device_index'] = self.callback_device_index

    # prefill
    decode_data, updated_vars = self._model.apply(
        mdl_vars,
        input_batch=inputs,
        method=self._model.greedy_prefill,
        mutable=[
            base_layer.NON_TRAINABLE,
            base_layer.DECODE_CACHE,
            base_layer.PREFIX_DECODE_CACHE,
        ],
        rngs={
            base_layer.PARAMS: k1,
            base_layer.RANDOM: k2,
        },
        decoder_params=self._method_hparams.decoder,
        **kwargs,
    )
    mdl_vars.update(updated_vars)
    logging.info('model state vars: {}'.format(mdl_vars))

    # initialize decoding state
    init_decode_state, updated_vars = self._model.apply(
        mdl_vars,
        input_batch=inputs,
        decode_data=decode_data,
        method=self._model.greedy_init_decode_state,
        mutable=[
            base_layer.NON_TRAINABLE,
            base_layer.DECODE_CACHE,
            base_layer.PREFIX_DECODE_CACHE,
        ],
        rngs={
            base_layer.PARAMS: k1,
            base_layer.RANDOM: k2,
        },
        decoder_params=self._method_hparams.decoder,
        **kwargs,
    )
    mdl_vars.update(updated_vars)

    # decoding while loop
    decode_state, updated_vars = self._model.apply(
        mdl_vars,
        decode_state=init_decode_state,
        method=self._model.greedy_decode_step_while,
        mutable=[
            base_layer.NON_TRAINABLE,
            base_layer.DECODE_CACHE,
            base_layer.PREFIX_DECODE_CACHE,
        ],
        rngs={
            base_layer.PARAMS: k1,
            base_layer.RANDOM: k2,
        },
        **kwargs,
    )
    mdl_vars.update(updated_vars)

    # process results
    outputs = self._model.apply(
        mdl_vars,
        decode_state=decode_state,
        input_batch=inputs,
        decode_data=decode_data,
        method=self._model.greedy_process_result,
        mutable=[
            base_layer.NON_TRAINABLE,
            base_layer.DECODE_CACHE,
            base_layer.PREFIX_DECODE_CACHE,
        ],
        rngs={
            base_layer.PARAMS: k1,
            base_layer.RANDOM: k2,
        },
        decoder_params=self._method_hparams.decoder,
        **kwargs,
    )
    return outputs


  # process results with selected index pjit function
  def _pjit_device_fn_greedy_process_result_with_idx(
      self, jax_func, input_pspecs: PSpecs, batch_size: int
  ) -> Callable[[DeviceTensors, DeviceTensors, DeviceTensors], DeviceTensors]:
    """Returns a pjit-ed model function with input handling."""

    def _wrapped_fn_greedy_process_result_with_idx(mdl_vars, decode_state, slot_idx):
      mdl_vars = jax.tree_util.tree_map(
          jax.lax.with_sharding_constraint,
          mdl_vars,
          self.model_state.mdl_var_pspecs,
      )

      prng_key = self._prng_key
      outputs = jax_func(
          mdl_vars, prng_key, 
          decode_state,
          slot_idx
      )
      # This assumes that outputs are generated after previous host calls, and
      # it is guaranteed by data dependency.
      if self.streamable:

        def _mark_done(dummy, _):
          del dummy
          self.mark_stream_output_done()

        hcb.id_tap(_mark_done, outputs, device_index=self.callback_device_index)
        # Unused final outputs. Return something to make output_to_host
        # blocking.
        return jnp.zeros((batch_size,), dtype=jnp.int32)

      return outputs

    # pjit-ed function.
    if self._enable_auto_sharding:
      return pjit.pjit(
          _wrapped_fn_greedy_process_result_with_idx,
          in_shardings=(pjit.AUTO(self.model_state.global_mesh), None, None),
          out_shardings=None,
      )
    else:
      return pjit.pjit(
          _wrapped_fn_greedy_process_result_with_idx,
          in_shardings=(self.model_state.mdl_var_pspecs, None, None),
          out_shardings=None,
      )
    
  def greedy_process_result_with_idx_jax_func(
      self,
      mdl_vars: NestedJTensor,
      prng_key: PRNGKey,
      decode_state: NestedJTensor,
      slot_idx
  ) -> NestedJTensor:
    if self._model.fprop_dtype == jnp.bfloat16:
      mdl_vars = jax.tree_map(
          (lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x),
          (mdl_vars),
      )

    context_p = base_layer.JaxContext.HParams(do_eval=True)
    k1, k2 = jax.random.split(prng_key)
    with base_layer.JaxContext.new_context(hparams=context_p):

      def _model_fn(decode_state):
        outputs = self.call_model_function_greedy_process_result_with_idx(
          decode_state, slot_idx,
          mdl_vars, [k1, k2])  # pytype: disable=wrong-arg-types  # jax-ndarray
        # DECODE_CACHE are not read by caller. But they can be large. Tell XLA
        # to remove it from output. Note MLP decoder don't have DECODE_CACHE.
        updated_vars = outputs[1]
        if base_layer.DECODE_CACHE in updated_vars:
          del updated_vars[base_layer.DECODE_CACHE]
        if base_layer.PREFIX_DECODE_CACHE in updated_vars:
          del updated_vars[base_layer.PREFIX_DECODE_CACHE]
        return outputs

      outputs = _model_fn(decode_state)

      if (
          self.method_params.cast_bfloat16_outputs
          and self._model.fprop_dtype == jnp.bfloat16
      ):
        # Convert bfloat16 back to float32.
        def maybe_to_float32(x):
          x = jnp.asarray(x)
          if x.dtype == jnp.bfloat16:
            return x.astype(jnp.float32)
          return x

        outputs = jax.tree_map(maybe_to_float32, outputs)
      return self.fetch_output(outputs)

  def call_model_function_greedy_process_result_with_idx(
      self, decode_state, slot_idx, mdl_vars, prng_key):
    k1, k2 = prng_key

    kwargs = {}
    if self.streamable:

      def callback_fn(x, _):
        assert self.model_state.is_primary_host
        self.enqueue_stream_output(x)

      kwargs['result_callback'] = decoder_utils.StreamingResultCallback(
          functools.partial(
              hcb.id_tap, callback_fn, device_index=self.callback_device_index
          ),
          interval_steps=self._method_hparams.stream_interval_steps,
      )

    if (
        'callback_device_index'
        in inspect.signature(self._model.decode_with_params).parameters
    ):
      kwargs['callback_device_index'] = self.callback_device_index

    # process results
    outputs = self._model.apply(
        mdl_vars,
        decode_state=decode_state,
        slot_idx=slot_idx,
        method=self._model.greedy_process_result_with_idx,
        mutable=[
            base_layer.NON_TRAINABLE,
            base_layer.DECODE_CACHE,
            base_layer.PREFIX_DECODE_CACHE,
        ],
        rngs={
            base_layer.PARAMS: k1,
            base_layer.RANDOM: k2,
        },
        decoder_params=self._method_hparams.decoder,
        **kwargs,
    )
    return outputs  


class TextToEmbedding(servable_model.ServableMethod):
  """Implements text embedding method."""

  def __init__(
      self,
      model: base_model.BaseModel,
      model_fn_name: str,
      model_state: servable_model.ServableModelState,
      text_to_embedding_hparams: TextToEmbeddingHParams,
      tokenizer_p: Any,
      prng_key: PRNGKey,
      enable_auto_sharding: bool = False,
      compiler_options: jax.stages.CompilerOptions | None = None,
  ):
    self._tokenizer = tokenizer_p.Instantiate()
    self._text_to_embedding_hparams = text_to_embedding_hparams
    dummy_input_sample = ''
    # TODO(b/289379065): Remove this workaround here.
    if (
        isinstance(self._tokenizer, lm_tokenizer.LMTokenizer)
        and self._tokenizer.tokenized
    ):
      dummy_input_sample = '1'
    self._tf_sess_pre_processing = np_tf_sess_wrapper.wrap_tf_session(
        self.tf_pre_processing
    )
    super().__init__(
        model,
        model_fn_name,
        model_state,
        text_to_embedding_hparams,
        prng_key,
        dummy_input_sample,
        enable_auto_sharding=enable_auto_sharding,
        compiler_options=compiler_options,
    )

  @classmethod
  def service_id(cls) -> str:
    return lm_service.SERVICE_ID

  def fetch_output(
      self, model_fn_outputs: NestedJTensor, model_fn_inputs: NestedJTensor
  ) -> NestedJTensor:
    """Fetches useful output tensors from the model function outputs."""
    if not self._text_to_embedding_hparams.output_padding_name:
      return py_utils.NestedMap(
          text_embedding=model_fn_outputs[0][
              self._text_to_embedding_hparams.output_embedding_name
          ],
      )
    else:
      return py_utils.NestedMap(
          text_embedding=model_fn_outputs[0][
              self._text_to_embedding_hparams.output_embedding_name
          ],
          padding=model_fn_outputs[0][
              self._text_to_embedding_hparams.output_padding_name
          ],
      )

  def pre_processing(self, raw_inputs: List[str]) -> NestedNpTensor:
    """Preprocesses an unpadded batch of data into host numpy arrays."""
    prefixes = np.array(raw_inputs)
    # Provide an empty suffix per prefix so we can use the common tokenizer and
    # get the EOS token appended appropriately.
    suffixes = np.array(['' for _ in range(len(raw_inputs))])
    return self._tf_sess_pre_processing(prefixes, suffixes)

  def tf_pre_processing(
      self,
      prefixes: NestedNpOrTfTensor,
      suffixes: NestedNpOrTfTensor,
  ) -> NestedTfTensor:
    """Tokenizes `prefixes` and `suffixes` using TF ops.

    Args:
      prefixes: the prefix text batch of shape [batch_size].
      suffixes: the suffix text batch of shape [batch_size].

    Returns:
      A NestedMap of preprocessed tensors.
    """
    result = servable_lm_common.tf_tokenize_inputs(
        prefixes,
        suffixes,
        self._tokenizer,
        self._text_to_embedding_hparams.max_input_seq_len,
        self._text_to_embedding_hparams.max_suffix_seq_len,
        self._text_to_embedding_hparams.include_eos_score,
    )

    preprocessed = py_utils.NestedMap(
        ids=result.ids,
        labels=result.labels,
        paddings=result.paddings,
        weights=result.weights,
        inputs_indicator=result.inputs_indicator,
    )

    return preprocessed

  def post_processing(self, compute_outputs: NestedNpTensor) -> List[Any]:
    """Postprocesses the output numpy arrays to final host output."""
    if self._text_to_embedding_hparams.output_padding_name:
      paddings = compute_outputs['padding']  # [batch==1, max_seq_len]
      assert paddings.shape[0] == 1  # only supports batch_size == 1
      emb = compute_outputs['text_embedding']  # [batch==1, max_seq_len, dim]
      lengths = np.sum(1 - paddings, dtype=jnp.int32)  # Assume 1 is for pad
      emb_no_pad = emb[0, :lengths, :]  # [actual_seq_len, dim]
      return [emb_no_pad]
    else:
      return list(compute_outputs['text_embedding'])


class LMGradientMethod(ServableLMMethod):
  """Implements the gradient method of LM."""

  def __init__(
      self,
      model: base_model.BaseModel,
      model_state: servable_model.ServableModelState,
      prng_key: PRNGKey,
      gradient_params: GradientHParams,
      tokenizer_p: Any,
      exportable: bool = False,
      enable_auto_sharding: bool = False,
      compiler_options: jax.stages.CompilerOptions | None = None,
  ):
    self._tokenizer = tokenizer_p.Instantiate()
    self._gradient_params = gradient_params
    # gradient param contains all score params as well.
    # used for computing score
    self._score_params = gradient_params
    self._delimiter = '/'
    dummy_input_sample = ('', '')
    # TODO(b/289379065): Remove this workaround here.
    if (
        isinstance(self._tokenizer, lm_tokenizer.LMTokenizer)
        and self._tokenizer.tokenized
    ):
      dummy_input_sample = ('1', '1')
    logging.info(
        'Using np_tf_sess_wrapper on LMGradientMethod.tf_pre_processing'
    )
    self._tf_sess_pre_processing = np_tf_sess_wrapper.wrap_tf_session(
        # `bucketize_inputs` is only used in SavedModel export. The sax-native
        # serving has an equivalent bucketization after `pre_processing`.
        lambda *args: self.tf_pre_processing(*args, bucketize_inputs=False)
    )
    super().__init__(
        model,
        '__call__',
        model_state,
        gradient_params,
        prng_key,
        dummy_input_sample,
        exportable=exportable,
        enable_auto_sharding=enable_auto_sharding,
        compiler_options=compiler_options,
    )

  def call_model_function(
      self, inputs: NestedJTensor, mdl_vars: NestedJTensor, prng_key: PRNGKey
  ) -> NestedJTensor:
    tensors_to_take_gradients = {
        'inputs': {},
        'mdl_vars': {},
    }
    inputs_tensor_names = (
        self._gradient_params.inputs_tensor_names
        if self._gradient_params.inputs_tensor_names is not None
        else {}
    )
    mdl_vars_tensor_names = (
        self._gradient_params.mdl_vars_tensor_names
        if self._gradient_params.mdl_vars_tensor_names is not None
        else {}
    )
    split_inputs_tensor_names = {
        name: name.split(self._delimiter) for name in inputs_tensor_names
    }
    split_mdl_vars_tensor_names = {
        name: name.split(self._delimiter) for name in mdl_vars_tensor_names
    }

    def fetch(tree, keys):
      for key in keys:
        tree = tree[key]
      return tree

    def insert(tree, keys, x):
      for key in keys[:-1]:
        tree = tree[key]
      tree[keys[-1]] = x
      return x

    for k, v in split_inputs_tensor_names.items():
      try:
        tensors_to_take_gradients['inputs'][k] = fetch(inputs, v)
      except Exception as e:
        raise ValueError(f'Failed to find tensor {k} from inputs') from e
    for k, v in split_mdl_vars_tensor_names.items():
      try:
        tensors_to_take_gradients['mdl_vars'][k] = fetch(mdl_vars, v)
      except Exception as e:
        raise ValueError(f'Failed to find tensor {k} from mdl_vars') from e

    call_fn = super().call_model_function

    def forward_fn(tensors_to_take_gradients, inputs_no_grad, mdl_vars_no_grad):
      for k, v in tensors_to_take_gradients['inputs'].items():
        insert(inputs_no_grad, split_inputs_tensor_names[k], v)
      for k, v in tensors_to_take_gradients['mdl_vars'].items():
        insert(mdl_vars_no_grad, split_mdl_vars_tensor_names[k], v)
      outputs = call_fn(inputs_no_grad, mdl_vars_no_grad, prng_key)
      return outputs[0][0]['total_loss'][0], outputs

    compute_gradient_fn = jax.value_and_grad(forward_fn, has_aux=True)
    (_, outputs), grads = compute_gradient_fn(
        tensors_to_take_gradients, inputs, mdl_vars
    )
    outputs = (outputs[0], outputs[1])  # 1 is for mutable.
    outputs[0][0]['gradients'] = grads
    return outputs

  def fetch_output(
      self, model_fn_outputs: NestedJTensor, model_fn_inputs: NestedJTensor
  ) -> NestedJTensor:
    # fetch loss and gradients from the model output
    metrics, per_example_output = model_fn_outputs[0]
    output = dict(
        # LMScore's fetch_output uses only the 0-th element of output.
        # per_example_output contains a 'scores' field by default from the loss
        # output, which will be the scores by default for fetch_output.
        # Models can retrieve intermediate per_token_xent from the forward pass
        # for fetch_output to mask out paddings.
        scores=LMScoreMethod.fetch_output(
            self, [per_example_output], model_fn_inputs
        )
    )

    for grads_type, grads_dict in metrics['gradients'].items():
      for tensor_name, grads in grads_dict.items():
        output[f'gradients/{grads_type}/{tensor_name}'] = grads

    return output

  def get_maxlen(self) -> int:
    return self._gradient_params.max_input_seq_len

  def output_seq_dim(self) -> int:
    return 1

  def pre_processing(self, raw_inputs: List[Tuple[str, str]]) -> NestedNpTensor:
    prefixes = np.array([prefix for prefix, _ in raw_inputs])
    suffixes = np.array([suffix for _, suffix in raw_inputs])
    return self._tf_sess_pre_processing(prefixes, suffixes)

  def post_processing(
      self, compute_outputs: NestedNpTensor
  ) -> List[Dict[str, List[float]]]:
    flattened_outputs = jax.tree_util.tree_map(
        lambda x: x.flatten().tolist(), compute_outputs
    )

    return [flattened_outputs]  # The extra list is to just conform to base api.

  def tf_pre_processing(
      self,
      prefixes: NestedNpOrTfTensor,
      suffixes: NestedNpOrTfTensor,
      extra_inputs: Optional[Mapping[str, Any]] = None,
      bucketize_inputs: bool = True,
  ) -> NestedTfTensor:
    """Tokenizes `prefixes` and `suffixes` using TF ops.

    This also implements `ExportableToSavedModel.tf_pre_processing`.

    Args:
      prefixes: the prefix text batch of shape [batch_size].
      suffixes: the suffix text batch of shape [batch_size].
      extra_inputs: optional mapping of extra input key to tensor or tensor spec
        of shape [batch_size].
      bucketize_inputs: whether to bucketize the preprocessed inputs based on
        max sequence length in the batch.

    Returns:
      A NestedMap of preprocessed tensors.
    """
    preprocessed = servable_lm_common.tf_tokenize_inputs(
        prefixes,
        suffixes,
        self._tokenizer,
        self._gradient_params.max_input_seq_len,
        self._gradient_params.max_suffix_seq_len,
        self._gradient_params.include_eos_score,
    )

    if bucketize_inputs:
      preprocessed = servable_lm_common.bucketize_tokenized_inputs(
          self.sorted_seq_lens, preprocessed
      )

    if extra_inputs:
      preprocessed.update(extra_inputs)

    return preprocessed

  def tf_post_processing(self, outputs: NestedTfTensor) -> NestedTfTensor:
    if self._gradient_params.mdl_vars_tensor_names:
      raise ValueError(
          'Exporting graident method with gradients to model '
          'variables is not supported since it is undefined '
          'how to introduce the batch dims for export signatures.'
      )

    return outputs

  def input_signature(
      self, batch_size: Optional[int]
  ) -> Tuple[TensorSpec, TensorSpec, Mapping[str, TensorSpec]]:
    """Implements `ExportableToSavedModel.input_signature`."""
    return (
        tf.TensorSpec([batch_size], dtype=tf.string, name='prefixes'),
        tf.TensorSpec([batch_size], dtype=tf.string, name='suffixes'),
        servable_lm_common.extra_inputs_to_tf_signature(
            self._extra_inputs,
            batch_size,
            self.method_params.extra_inputs_dtypes,
        ),
    )

  @property
  def tf_trackable_resources(self) -> Any:
    """Implements `ExportableToSavedModel.tf_trackable_resources`."""
    return None


class ServableLMModel(servable_model.ServableModel):
  """Represents an implementation for the LM service, backed by a model.

  This class is responsible for model loading, batch padding, etc.
  """

  def init_method(
      self,
      method: str,
      model: base_model.BaseModel,
      model_state: servable_model.ServableModelState,
      method_params: servable_model_params.ServableMethodParams,
      prng_key: PRNGKey,
  ) -> servable_model.ServableMethod:
    assert isinstance(self.model_config, ServableLMModelParams)
    tokenizer_p = self.model_config.serving_tokenizer()
    if method == LMMethodName.SCORE:
      assert isinstance(method_params, ScoreHParams)
      return LMScoreMethod(
          model,
          model_state,
          prng_key,
          method_params,
          tokenizer_p,
          exportable=True,
          enable_auto_sharding=self._enable_auto_sharding,
          compiler_options=self._compiler_options,
      )
    elif method == LMMethodName.GENERATE:
      assert isinstance(method_params, DecodeHParams)
      # return LMDecodeMethod(
      return LMDecodeMethodContinuousBatching(
          model,
          model_state,
          prng_key,
          method_params,
          tokenizer_p,
          exportable=True,
          enable_auto_sharding=self._enable_auto_sharding,
          compiler_options=self._compiler_options,
      )
    elif method == LMMethodName.GENERATE_STREAM:
      assert isinstance(method_params, DecodeHParams)
      return LMDecodeMethod(
          model,
          model_state,
          prng_key,
          method_params,
          tokenizer_p,
          exportable=False,
          streamable=True,
          enable_auto_sharding=self._enable_auto_sharding,
          compiler_options=self._compiler_options,
      )
    elif method == LMMethodName.EMBED:
      assert isinstance(method_params, TextToEmbeddingHParams)
      assert method_params.output_embedding_name is not None
      if method_params.model_method_name is None:
        raise ValueError(
            'Must specify `model_method_name` in TextToEmbeddingHParams.'
        )
      return TextToEmbedding(
          model,
          method_params.model_method_name,
          model_state,
          method_params,
          tokenizer_p,
          prng_key=prng_key,
          enable_auto_sharding=self._enable_auto_sharding,
          compiler_options=self._compiler_options,
      )
    elif method == LMMethodName.GRADIENT:
      assert isinstance(method_params, GradientHParams)
      assert (
          method_params.inputs_tensor_names is not None
          or method_params.mdl_vars_tensor_names is not None
      )
      return LMGradientMethod(
          model,
          model_state,
          prng_key,
          method_params,
          tokenizer_p,
          exportable=True,
          enable_auto_sharding=self._enable_auto_sharding,
          compiler_options=self._compiler_options,
      )
    else:
      raise NotImplementedError(f'method {method} not implemented')

  def supports_dummy_compute_on_primary(self) -> bool:
    if self.methods is None or not isinstance(self.methods, Dict):
      return True
    for method in list(self.methods.values()):
      has_multiple_seq_lens = (
          hasattr(method, 'sorted_seq_lens')
          and method.sorted_seq_lens is not None
          and len(method.sorted_seq_lens) > 1
      )
      if has_multiple_seq_lens:
        return False
    return True