import os
from pathlib import Path
import random
import argparse
import configparser
from tqdm import tqdm
import numpy as np
import logging

import nltk

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, default_data_collator, AutoConfig, AutoTokenizer, HfArgumentParser

# import from MultiFactor project directory
from run_utils import set_seed
from MultiFactor.arguments import ModelArguments, DataTrainingArguments, TrainingArguments
from MultiFactor.utils import save_args, init_default_args
from MultiFactor.multi_factor_trainer import MyTrainer, evaluate_data, Metrics
from MultiFactor.multi_factor_dataset import MyDataset, DataFeature, DataInst, NodeFlag, DecoderFlag
from MultiFactor.multi_factor_config import MultiFactorConfig

ANS_SEP = "<ans>"
KEYPHRASE_SEP = "<keyphrase>"
QUESTION_SEP = "<question>"
FULL_ANS_SEP = "<fa>"
CONTEXT_SEP = "<passage>"
MAX_LENGTH = 512

CONFIG_PATH = Path(__file__).absolute().parent / "config.ini"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s   %(levelname)s   %(message)s')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--job', type=str, default='cqg', help='choice the dataset')
    parser.add_argument('-f', '--data_format', type=str, default='multi_factor', help='choice the data format')
    parser.add_argument('-c', '--config_file', type=str, default=str(CONFIG_PATH), help='configuration file')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='which GPU to use for evaluation')
    parser.add_argument('-seed', '--seed', type=int, default=42, help='random seed num')
    parser.add_argument('-m', '--mode', type=str, default="train", choices=['train', 'test'], help='train or test')
    args, remaining_args = parser.parse_known_args()
    set_seed(args.seed)

    config = configparser.ConfigParser(allow_no_value=False)
    config.read(args.config_file)
    data_format = args.data_format
    assert args.job in config, f"FileExistedError: Load the {args.config_file} failed, please check the file directory"
    logging.info(f"data_format: {data_format}")

    # set defaults for other arguments
    defaults = {
        "model_type": "multifactor",
        # T5Config
        "if_node_cls": True,
        "cls_loss_weight": 1,
        "cls_type": "node",
        "cls_has_answer": False,
        "hard_flag": 2,
        "if_cross_enhanced_k": True,
        "if_full_answer_decoder": True,
        "fa_loss_weight":0.2,
        # data and training
        'if_load_data_cache': True,
        'learning_rate': 1e-4,
        "warmup_ratio": 0.1,
        'train_subset': 1,
        # debug
        # 'train_subset': .001,
    }

    # the config file gives default values for the command line arguments
    defaults.update(dict(config.items(args.job)))

    # parse remaining arguments and divide them into three categories
    second_parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, MultiFactorConfig))
    second_parser.set_defaults(**defaults)
    model_args, data_args, training_args, multi_factor_args = second_parser.parse_args_into_dataclasses(remaining_args)
    
    # The training code not supports multi-gpu training.
    # When running the code on multi-gpu machine, 
    # the training batch_size will be n_gpu * per_device_train_batch_size
    # So here we force the n_gpu as 1
    # details are in: src/transformers/training_args.py, transformers 4.35.2, Line 1770
    # or set visible cuda device number as 1
    training_args._n_gpu = 1

    # check experiments args
    # init the model path or name, and some other kwargs
    # save all arguments
    training_args.output_dir += "/fad"
    init_default_args(args, data_args, training_args, model_args, multi_factor_args)
    experiments_output_dir = save_args(args, data_args, training_args, model_args, multi_factor_args)

    logging.info(f"dataset path: {data_args.data_dir}")
    logging.info(f"model path  : {model_args.model_name_or_path}")
    logging.info(f"experiments output path: {experiments_output_dir}")
    import json
    from dataclasses import asdict
    logging.info(f"model args: {json.dumps(asdict(multi_factor_args), indent=4)}")


    # init config, tokenizer, model
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=False,
        model_max_length=MAX_LENGTH
    )
    tokenizer.add_tokens([ANS_SEP, KEYPHRASE_SEP, QUESTION_SEP, FULL_ANS_SEP, CONTEXT_SEP], special_tokens=True)
    if multi_factor_args.model_type == "Baseline":
        from transformers import T5ForConditionalGeneration, AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
        ).to(device)
    elif multi_factor_args.model_type in ["MultiFactor", "FADecoder", "NodeCls"]:
        from MultiFactor.modeling_bridget5 import BridgeT5 as MyModel
        logging.info(f"Loading the model: BridgeT5")
        model = MyModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            multi_factor_config=multi_factor_args,
            cache_dir=model_args.cache_dir,
        ).to(device)
        # model._init_full_answer_decoder()
        # if multi_factor_args.model_freeze:
        #     model.training_freeze_policy()
        logging.info(f"model architecture: {model.architecture_logging}")
    else:
        model = None
        raise ValueError(f"model type is {multi_factor_args.model_type}, which is unexpected")

    # load dataset
    data_path = data_args.data_dir
    if args.mode == "train":
        logging.info("Loading dataset")
        train_dataset = MyDataset(data_path, 'train', args.data_format,
                                  tokenizer, 
                                  seed=args.seed,
                                  if_load_cache_pt=data_args.if_load_data_cache,
                                  train_subset=data_args.train_subset)
        # print(train_dataset._features[0])
        dev_dataset = MyDataset(data_path, 'dev', args.data_format,
                                tokenizer,
                                if_load_cache_pt=data_args.if_load_data_cache,
                                train_subset=1)
        # print(dev_dataset._features[0])
        test_dataset = MyDataset(data_path, 'test', args.data_format,
                                 tokenizer,
                                 if_load_cache_pt=data_args.if_load_data_cache,
                                 train_subset=1)
        logging.info("Loaded dataset successfully")

        # init trainer
        my_trainer = MyTrainer(model, tokenizer, train_dataset, experiments_output_dir, training_args, device,
                                  dev_dataset, test_dataset, data_args)
        my_trainer.train(data_format=args.data_format)
    else:
        test_kwargs = {
            "save_evaluate_result": True,
            "save_dir": experiments_output_dir,
            "data_format": args.data_format,
        }
        test_dataset = MyDataset(data_path, 'test', args.data_format,
                                 tokenizer,
                                 if_load_cache_pt=data_args.if_load_data_cache,
                                 train_subset=1)
        test_result = evaluate_data(tokenizer, training_args, data_args, test_dataset, model, device, 
                                    **test_kwargs)

        # print(test_result._asdict())


if __name__ == '__main__':
    main()
