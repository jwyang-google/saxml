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
"""This file keeps an mapping between config name (string) to config class."""

import quantization_configs


NAME_TO_CONFIG = {
    'gptj': quantization_configs.QuantizationConfigsGPTJ(),
    'gamma2b': quantization_configs.QuantizationConfigsGamma2B(),
    'gamma7b': quantization_configs.QuantizationConfigsGamma7B(),
}

NAME_TO_CONFIG_STACKED = {
    'gptj': quantization_configs.QuantizationConfigsGPTJStacked(),
}


def get_name_to_config(stacked: bool = False) -> dict[str, object]:
  """Get the model config map."""
  if stacked:
    return NAME_TO_CONFIG_STACKED
  else:
    return NAME_TO_CONFIG
