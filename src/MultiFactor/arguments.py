__all__ = ['TrainingArguments', 'ModelArguments', 'DataTrainingArguments']
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Uses some code from
# https://github.com/huggingface/transformers/blob/master/examples/seq2seq/finetune_trainer.py

from dataclasses import dataclass, field
from typing import Optional
import transformers


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    Arguments for the Trainer.
    """

    optim: str = field(
        default='adamw_hf',
        metadata={"help": "The optimizer to use: adamw_hf, adamw_torch, adamw_apex_fused, or adafactor."}
    )

    output_dir: str = field(
        default='experiments',
        metadata={"help": "The output directory where the results and model weights will be written."}
    )

    zero_shot: bool = field(
        default=False,
        metadata={"help": "Zero-shot setting"}
    )

    # per_gpu_train_batch_size: Optional[int] = field(
    #     default=1,
    #     metadata={
    #         "help": "Deprecated, the use of `--per_device_train_batch_size` is preferred. "
    #         "Batch size per GPU/TPU core/CPU for training."ƒ©
    #     },
    # )
    # per_gpu_eval_batch_size: Optional[int] = field(
    #     default=1,
    #     metadata={
    #         "help": "Deprecated, the use of `--per_device_eval_batch_size` is preferred."
    #         "Batch size per GPU/TPU core/CPU for evaluation."
    #     },
    # )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )

    eval_epoch_interval: int = field(
        default=1,
        metadata={"help": "evaluate while training for every interval epochs"}
    )

    eval_dataset: str = field(
        default='dev',
        metadata={"help": "The output directory where the results and model weights will be written."}
    )

    save_model_pt: bool = field(
        default=False,
        metadata={"help": "Save model."}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_device: str = field(
        default=None,
        metadata={"help": "cpu, gpu"}
    )

    train_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    test_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "drop out"}
    )
    datasets: Optional[str] = field(
        default=None,
        metadata={"help": "Comma separated list of dataset names, for training."}
    )

    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to data directory"}
    )

    if_load_data_cache: Optional[bool] = field(
        default=True,
        metadata={"help": "if load dataset cache file directly"}
    )

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, shorter sequences will be padded."
        },
    )

    max_output_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum output sequence length (default is the same as input)"
        },
    )

    train_subset: float = field(
        default=1, metadata={"help": "The portion of training data to use"}
    )

    episodes: str = field(
        default='0', metadata={"help": "Episode indices -- a single number such as 3 or an interval such as 1-4\n"
                                       "The index is also used as random seeds and this setting is therefore used to "
                                       "repeat multiple experiments."}
    )

    num_beams: int = field(
        default=5,
        metadata={"help": "Number of beams for beam search during generation (only affects evaluation)"}
    )

    max_seq_length_eval: int = field(
        default=128,
        metadata={
            "help": "Maximum input sequence length at evaluation time (default is equal to max_seq_length)"
        },
    )

    max_output_seq_length_eval: int = field(
        default=None,
        metadata={
            "help": "The maximum output sequence length at evaluation time (default is the same as input)"
        },
    )

    # few-shot arguments
    num_shots: int = field(
        default=None, metadata={"help": "number of shots (few-shot argument for the FewRel dataset)"}
    )

    num_ways: int = field(
        default=None, metadata={"help": "number of ways (few-shot argument for the FewRel dataset)"}
    )

    num_query: int = field(
        default=5, metadata={"help": "number of query examples (few-shot argument for the FewRel dataset)"}
    )


@dataclass
class ReinforcementArguments:
    score_type: str = field(
        default='global',
        metadata={"help": "if global, scoring the generation at last step;"
                          "elif step: scoring the action at output id level instead of token."}
    )
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control), default: 0.2"}
    )
    target: Optional[float] = field(
        default=6.0,
        metadata={"help": "Target KL value for adaptive KL control, default: 6.0"}
    )
    horizon: Optional[int] = field(
        default=10000,
        metadata={"help": "Horizon for adaptive KL control, default: 10000"}
    )
    gamma: Optional[float] = field(
        default=1.0,
        metadata={"help": "Gamma parameter for advantage calculation, default: 1."}
    )
    lam: Optional[float] = field(
        default=0.95,
        metadata={"help": "Lambda parameter for advantage calcualation, default: 0.95"}
    )
    cliprange: Optional[float] = field(
        default=0.2,
        metadata={"help": "Range for clipping in PPO policy gradient loss, default: 0.2"}
    )
    cliprange_value: Optional[float] = field(
        default=0.2,
        metadata={"help": "Range for clipping values in loss calculation, default: 0.2"}
    )
    clip_kl: Optional[bool] = field(
        default=True,
        metadata={"help": "clip the kl penalty when setting it as True."}
    )
    cliprange_kl_low: Optional[float] = field(
        default=-0.2,
        metadata={"help": "Range for clipping values in loss calculation, default: 0.2"}
    )
    cliprange_kl_high: Optional[float] = field(
        default=0.2,
        metadata={"help": "Range for clipping values in loss calculation, default: 0.2"}
    )
    vf_coef: Optional[float] = field(
        default=0.1,
        metadata={"help": "Scaling factor for value loss, default: 0.1"}
    )
    adap_kl_ctrl: Optional[bool] = field(
        default=True,
        metadata={"help": "Use adaptive KL control, otherwise linear, default: True"}
    )

    batch_size: Optional[int] = field(
        default=8,
        metadata={"help": "batch size, default: 8"}
    )

    forward_batch_size: Optional[int] = field(
        default=8,
        metadata={"help": "batch size = forward batch size * world size"
                          "When not using parallel training, it's equal to batch size, default: 8"}
    )

    ppo_epochs: Optional[int] = field(
        default=4,
        metadata={"help": "Number of optimisation epochs per batch of samples, default: 4"}
    )

    """
        ref
            - https://huggingface.co/docs/transformers/v4.19.2/en/main_classes/text_generation
              #transformers.generation_utils.GenerationMixin
    """
    do_sample: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to use sampling ; use greedy decoding otherwise."}
    )

    rl_num_beams: Optional[int] = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."}
    )

    top_p: Optional[float] = field(
        default=1.0,
        metadata={"help": "If set to float < 1, only the most probable tokens with probabilities"
                          " that add up to top_p or higher are kept for generation, default to 1.0"}
    )

    top_k: Optional[int] = field(
        default=50,
        metadata={"help": "The number of highest probability vocabulary tokens to keep "
                          "for top-k-filtering, default to 50"}
    )

    num_return_sequences: Optional[int] = field(
        default=1,
        metadata={"help": "The number of independently computed returned sequences "
                          "for each element in the batch, default to 1"}
    )