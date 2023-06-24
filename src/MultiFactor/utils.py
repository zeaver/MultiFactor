import os
import json
import datetime
import copy
from dataclasses import dataclass, asdict
from typing import Union, List, Tuple, Dict

import torch
from torch.utils.data._utils.collate import default_collate
import numpy as np

try:
    from .arguments import ModelArguments, DataTrainingArguments, TrainingArguments
    from .multi_factor_config import MultiFactorConfig
except:
    from arguments import ModelArguments, DataTrainingArguments, TrainingArguments
    from multi_factor_config import MultiFactorConfig


def init_default_args(input_args,
                      data_args: DataTrainingArguments,
                      training_args: TrainingArguments,
                      model_args: ModelArguments,
                      ski_args: MultiFactorConfig):
    mode = input_args.mode
    if mode == "train":
        model_args.model_name_or_path = model_args.train_model_name_or_path
    else:
        model_args.model_name_or_path = model_args.test_model_name_or_path

    # process arguments related to max length
    if data_args.max_output_seq_length_eval is None:
        # defaults first to max_output_seq_length, then max_seq_length_eval, then max_seq_length
        data_args.max_output_seq_length_eval = data_args.max_output_seq_length \
                                               or data_args.max_seq_length_eval \
                                               or data_args.max_seq_length

    if data_args.max_output_seq_length is None:
        # defaults to max_seq_length
        data_args.max_output_seq_length = data_args.max_seq_length

    if data_args.max_seq_length_eval is None:
        # defaults to max_seq_length
        data_args.max_seq_length_eval = data_args.max_seq_length


def save_args(input_args,
              data_args: DataTrainingArguments,
              training_args: TrainingArguments,
              model_args: ModelArguments,
              ski_args: MultiFactorConfig):
    config_file = input_args.config_file
    try:
        os.mkdir(training_args.output_dir)
    except FileExistsError:
        pass

    try:
        __model_name = f'{model_args.model_name_or_path.split("/")[-1]}'
    except:
        __model_name = f'{model_args.model_name_or_path}'
    output_dir = os.path.join(
        training_args.output_dir,
        f'{input_args.job}/'
        f'{ski_args.model_type}/'
        f'{input_args.data_format}/'
        f"{ski_args.task_prefix_str}"
        f"__{__model_name}"
        f'__GPUNums{torch.cuda.device_count()}'
        f'__b{training_args.train_batch_size}'
        f'__sub{data_args.train_subset}'
        f'__lr_{training_args.learning_rate}'
        f'__seed_{input_args.seed}'
        f'__epoch_{training_args.num_train_epochs}'
        f'__beam{data_args.num_beams}'
    )

    try:
        os.mkdir(training_args.output_dir)
    except FileExistsError:
        pass

    time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = os.path.join(output_dir, time)

    try:
        os.makedirs(output_dir)
        with open(output_dir + '/data_args.json', 'w', encoding='utf-8') as file:
            json.dump(data_args.__dict__, file, ensure_ascii=False, indent=4)
        with open(output_dir + '/commond_line_args.json', 'w', encoding='utf-8') as file:
            json.dump(vars(input_args), file, ensure_ascii=False, indent=4)
        with open(output_dir + '/model_args.json', 'w', encoding='utf-8') as file:
            json.dump(model_args.__dict__, file, ensure_ascii=False, indent=4)
        training_args_dict = asdict(training_args)
        for k, v in training_args_dict.items():
            training_args_dict[k] = str(v)
        with open(output_dir + '/training_args.json', 'w', encoding='utf-8') as file:
            json.dump(training_args_dict, file, ensure_ascii=False, indent=4)
        with open(output_dir + '/ski_args.json', 'w', encoding='utf-8') as file:
            json.dump(asdict(ski_args), file, ensure_ascii=False, indent=4)
    except FileExistsError:
        pass

    return output_dir


def shift_labels(labels: Union[List, torch.tensor],
                 cut_index: Union[int, torch.Tensor],
                 decoder_start_token_id: Union[int, torch.Tensor],
                 return_tensors: str = None) -> (Union[List[int], torch.tensor], Union[List[int], torch.tensor]):
    #   [start_token_id] + label[:-1]
    # ref:
    #  - https://github.com/huggingface/transformers/blob/v4.21.3/src/transformers/models/t5/modeling_t5.py
    #   function: def _shift_right(self, input_ids):
    if cut_index == -1:
        cut_len = len(labels)
    else:
        cut_len = cut_index
    if return_tensors is None:
        shifted_input_ids = [0] * cut_len
        shifted_input_ids[1:] = copy.deepcopy(labels[:cut_index])
        shifted_input_ids[0] = decoder_start_token_id
        return shifted_input_ids, [1] * cut_len
    elif return_tensors == "pt" and torch.is_tensor(labels):
        shifted_input_ids = torch.zeros(cut_len)
        shifted_input_ids[1:] = labels[:cut_index].clone()
        shifted_input_ids[0] = decoder_start_token_id
        return shifted_input_ids, torch.ones_like(shifted_input_ids)
    else:
        NotImplementedError(f"self.data_args.return_tensors should be None or 'pt',"
                            f"but {return_tensors}")


def collate_feature(all_feature_list: List[Dict], feature_name: str,
                    pad_id: Union[int, float, torch.Tensor],
                    pad_side: str = "right") -> List[np.ndarray]:
    assert feature_name in list(all_feature_list[0].keys()), \
        f"feature should in feature class, no {feature_name}"
    assert pad_side == "left" or pad_side == "right", \
        f"Please input 'right' or 'left' pad side, " \
        f"not the {pad_side}"
    feature_list = [sample[feature_name] for sample in all_feature_list]
    max_len = max([len(f) for f in feature_list])
    collate_feature_list = []
    for f_i in feature_list:
        pad_list = np.ones((max_len - len(f_i)), dtype=np.int64) * pad_id
        f_i = np.asarray(f_i, dtype=np.int64) if isinstance(f_i, list) else f_i
        if pad_side == "left":
            collate_feature_list.append(np.append(pad_list, f_i))
        else:
            collate_feature_list.append(np.append(f_i, pad_list))
    return default_collate(collate_feature_list)
