
#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""
This script is used to train BPE tokenizers for machine translation.

Usage example:
    python cli/create_tokenizer.py \
        --dataset_name=stas/wmt14-en-de-pre-processed \
        --vocab_size=16_384 \
        --save_dir=output_dir
"""
import os
import argparse
import logging
from packaging import version

import datasets
from datasets import load_dataset

import transformers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Setup logging
logger = logging.getLogger(__file__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

datasets.utils.logging.set_verbosity_warning()
transformers.utils.logging.set_verbosity_info()



def main():
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    logger.info(f"Loading dataset")
    raw_datasets = load_dataset('cnn_dailymail', '3.0.0')
    if "validation" not in raw_datasets:
        # will create "train" and "test" subsets
        # fix seed to make sure that the split is reproducible
        # note that we should use the same seed here and in train.py
        raw_datasets = raw_datasets["train"].train_test_split(test_size=2000, seed=42)

    logger.info(f"Building tokenizer for the language")

    # Optional Task: If you are using a dataset different from stas/wmt14-en-de-pre-processed
    # depending on the dataset format, you might need to modify the iterator (line 109)
    # YOUR CODE STARTS HERE
    source_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    source_tokenizer_trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"], vocab_size=32000)
    source_tokenizer.pre_tokenizer = Whitespace()
    print(raw_datasets["train"])

    source_iterator = (item["translation"][args.source_lang] for item in raw_datasets["train"])
    source_tokenizer.train_from_iterator(
        source_iterator,
        trainer=source_tokenizer_trainer,
    )

    source_tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=source_tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
    )
    logger.info(f"Saving source to ../output_dir/cnn_tokenizer_tokenizer")
    source_tokenizer.save_pretrained(os.path.join("../output_dir", f"cnn_tokenizer"))
    # YOUR CODE ENDS HERE


if __name__ == "__main__" :
    if version.parse(datasets.__version__) < version.parse("1.18.0"):
        raise RuntimeError("This script requires Datasets 1.18.0 or higher. Please update via pip install -U datasets.")
    main()
