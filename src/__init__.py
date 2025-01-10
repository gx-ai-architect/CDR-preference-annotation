# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__version__ = "0.1.0.dev"
from .dpo_vllm import DPOInferenceVLLM
from .logratio_hf import LogratioHF
from .models import DPO_MODEL_CONFIG, REWARD_MODEL_CONFIG
from .vllm_server import VLLM
from .tgi_server import TGI
from .utils import (
    check_tokenizer_chat_template,
    save_to_local,
    convert_to_json_format,
    load_simple_dataset,
    jdump,
    zip_,
    read_jsonl
)
