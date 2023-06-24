__all__ = ["MyDataset", "NodeFlag", "DataFeature", "DecoderFlag"]

import copy
import re
import os
from pathlib import Path
import copy
import random
import logging
from typing import Dict, List, Optional, Union
import json

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from transformers import PreTrainedTokenizer

from dataclasses import dataclass, asdict
import numpy as np
import collections
from tqdm import tqdm

try:
    from arguments import DataTrainingArguments
except:
    from .arguments import DataTrainingArguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ANS_SEP = "<ans>"
KEYPHRASE_SEP = "<keyphrase>"
QUESTION_SEP = "<question>"
FULL_ANS_SEP = "<fa>"
CONTEXT_SEP = "<passage>"

DATASET_PREFIX = Path(__file__).absolute().parent.parent / "dataset"

QUESTION_PROMPT = "Ask a question: "
FULL_ANS_PROMPT = "Full answer sentence: "

MAX_LENGTH = 512

DataInst = collections.namedtuple('DataInst', 'question answer context p_phrase n_phrase full_answer')
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s', level=logging.INFO)


@dataclass
class NodeFlag:
    index_range: Union[List[Union[int, float]], torch.Tensor, np.ndarray, None] = None
    token_ids: Union[List[Union[int, float]], torch.Tensor, np.ndarray, None] = None
    tag: Union[List[Union[int, float]], torch.Tensor, np.ndarray, None] = None

    def __post_init__(self):
        assert (self.index_range is not None) and (self.token_ids is not None), \
            f"Please init the node flag, the dix range is {self.index_range}" \
            f"the token ids is {self.token_ids}"
        assert (int(self.tag[0]) in [int(1), int(0)]), \
            f"Please init the tag: 0, -1 or 1, not {self.tag}"

    def to_pytorch_tensor(self, node_flag_device: torch.device = None):
        self.index_range = torch.as_tensor(self.index_range, device=node_flag_device, dtype=torch.long)
        self.token_ids = torch.as_tensor(self.token_ids, device=node_flag_device, dtype=torch.long)
        self.tag = torch.as_tensor(self.tag, dtype=torch.long, device=node_flag_device)


@dataclass
class DecoderFlag:
    input_ids_flag: Union[List[Union[int, float, np.ndarray]], torch.Tensor, np.ndarray, None] = None
    nodes_flag: Union[List[List[NodeFlag]], List[NodeFlag], None] = None
    ans_special_id_idx: Union[List[Union[int, float, np.ndarray]], torch.Tensor, np.ndarray, None] = None

    # node cross attention module
    decoder_nodes_attention_mask: Union[List[Union[int, float, np.ndarray]], torch.Tensor, np.ndarray, None] = None

    def flag2tensor(self, flag_device: torch.device = None):
        for node_i in self.nodes_flag:
            if isinstance(node_i, NodeFlag):
                node_i.to_pytorch_tensor(flag_device)
            else:
                for n in node_i:
                    n.to_pytorch_tensor(flag_device)
        self.input_ids_flag = torch.as_tensor(self.input_ids_flag, device=flag_device, dtype=torch.long)
        self.ans_special_id_idx = torch.as_tensor(self.ans_special_id_idx, device=flag_device, dtype=torch.long)
        self.decoder_nodes_attention_mask = \
            torch.as_tensor(self.decoder_nodes_attention_mask, device=flag_device, dtype=torch.long)

    def __str__(self):
        return {
            "input_ids_flag": self.input_ids_flag,
            "nodes_info": self.nodes_flag,
            "ans_special_id_idx": self.ans_special_id_idx,
            "decoder_nodes_attention_mask": self.decoder_nodes_attention_mask
        }


@dataclass
class DataFeature:
    input_ids: Union[List[Union[int, float]], torch.Tensor, np.ndarray, None] = None
    labels: Union[List[Union[int, float]], torch.Tensor, np.ndarray, None] = None
    decoder_flags: Optional[DecoderFlag] = None
    decoder_input_ids: Union[List[Union[int, float]], torch.Tensor, np.ndarray, None] = None
    attention_mask: Union[List[Union[int, float]], torch.Tensor, np.ndarray, None] = None
    decoder_attention_mask: Union[List[Union[int, float]], torch.Tensor, np.ndarray, None] = None
    decoder_full_answer_labels: Union[List[Union[int, float]], torch.Tensor, np.ndarray, None] = None


class MyDataset(Dataset):

    def __init__(self,
                 file_path: Union[str, None],
                 split: str,
                 data_format: str,
                 tokenizer: PreTrainedTokenizer,
                 max_input_length: int = 512,
                 max_output_length: int = 64,
                 if_load_cache_pt: bool = False,
                 train_subset: float = 1,
                 seed: int = None,
                 shuffle: bool = True,
                 data_args: DataTrainingArguments = None,
                 ) -> None:

        if seed is not None:
            # set random seed for repeatability
            random.seed(seed)
        self.dataset_name = file_path.split("/")[-1]
        if self.dataset_name.startswith("squad") or self.dataset_name.startswith("pcqg"):
            INPUT_STRING = "context"
        else:
            INPUT_STRING = "fact"

        self.data_args = data_args
        if data_args:
            self.negative_hit_tag = 0
        else:
            self.negative_hit_tag = 0
        assert self.negative_hit_tag == -1 or self.negative_hit_tag == 0, \
            f"self.negative_hit_tag except -1 or 0, actually {self.negative_hit_tag}"

        self.tokenizer = tokenizer
        assert split in ["train", "dev", "test"]
        self.split = split
        self.data_format = data_format

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        if if_load_cache_pt is True:
            cache_data_file = os.path.join(file_path, f"{data_format}/{split}.pt")
            self.load_cache_file(cache_data_file)
        else:
            json_file = os.path.join(file_path, f"{split}.json")
            print(f"[Data Info] Reading file: {json_file}")
            # print(file)
            with open(json_file, 'r', encoding='utf-8') as read_file:
                data = json.load(read_file)

            if self.data_format in ["full_answer_concat",
                                    "multi_factor", "top1_q_model",
                                    "multi_factor_mixqg", "top1_q_model_mixqg",
                                    "mix_full_answer"]:
                fa_file = os.path.join(file_path, f"{self.data_format}/{split}.json")
                with open(fa_file, 'r', encoding='utf-8') as f:
                    fa = json.load(f)
                fa = [list(fa_i.values()) for fa_i in fa]

                self.examples = []
                pred_fa_examples = []
                for i, sample in tqdm(enumerate(data)):
                    if split != "train":
                        fas = [fa[i][0]]
                    else:
                        fas = fa[i]

                    if self.data_format != "mix_full_answer":
                        pred_fa_examples.append(DataInst(question=str(sample['question']),
                                                         answer=str(sample['answer']),
                                                         context=str(sample[INPUT_STRING]),
                                                         p_phrase=sample['p_phrase'],
                                                         n_phrase=sample['n_phrase'],
                                                         full_answer=fas)
                                                )
                    else:
                        for fa_j in fas:
                            pred_fa_examples.append(DataInst(question=str(sample['question']),
                                                             answer=str(sample['answer']),
                                                             context=str(sample[INPUT_STRING]),
                                                             p_phrase=sample['p_phrase'],
                                                             n_phrase=sample['n_phrase'],
                                                             full_answer=[fa_j])
                                                    )
            elif self.data_format in ["format_1", "format_2", "format_3", "q_model_upper", "pet", "pet_mixqg"]:
                self.examples = []
                for sample in tqdm(data):
                    self.examples.append(DataInst(question=str(sample['question']),
                                                  answer=str(sample['answer']),
                                                  context=str(sample[INPUT_STRING]),
                                                  p_phrase=sample['p_phrase'],
                                                  n_phrase=sample['n_phrase'],
                                                  full_answer=sample["full answer"])
                                         )
            else:
                raise ValueError(f"data format is {self.data_format}, not our expected")
            self._features: List[DataFeature] = self.convert_instances_to_feature_tensors()
        logging.info(
            f"data format: {self.data_format}, totally example num: {len(self.examples)}, feature num: {len(self._features)}")
        # shuffle indices
        self.indices = list(range(len(self._features)))
        if seed is not None and shuffle:
            random.shuffle(self.indices)

        # compute effective size of the dataset
        self.effective_size = round(train_subset * len(self._features))
        if train_subset != 1:
            logging.info(
                f"Effective dataset size reduced to {self.effective_size} ({train_subset * 100:.0f}%)")

    def __len__(self) -> int:
        return self.effective_size

    def __getitem__(self, idx) -> DataFeature:
        return self._features[self.indices[idx]]

    def save_dataset(self, save_file: str):
        torch.save({'features': self._features, 'examples': self.examples}, save_file)

    def load_cache_file(self, cache_file: str):
        _data = torch.load(cache_file)
        self.examples = _data['examples']
        self._features = _data['features']

    def collate_fn(self, batch: List[DataFeature]):
        try:
            from utils import collate_feature
        except:
            from .utils import collate_feature
        batch_features_dicts: List[Dict] = list(map(asdict, batch))
        collate_dict = dict()
        collate_dict["input_ids"] = collate_feature(batch_features_dicts, "input_ids", self.tokenizer.pad_token_id)
        collate_dict["labels"] = collate_feature(batch_features_dicts, "labels", -100)
        collate_dict["decoder_full_answer_labels"] = collate_feature(batch_features_dicts, "decoder_full_answer_labels",
                                                                     -100)

        # node
        batch_decoder_flags_list = [feature['decoder_flags'] for feature in batch_features_dicts]
        temp_id_flags = collate_feature(batch_decoder_flags_list, "input_ids_flag", 0)
        temp_nodes_flag = [feature.decoder_flags.nodes_flag for feature in batch]
        temp_batch_ans_idx = [feature.decoder_flags.ans_special_id_idx for feature in batch]
        temp_nodes_attention_mask = collate_feature(batch_decoder_flags_list, "decoder_nodes_attention_mask", 0)
        collate_dict["decoder_flags"] = DecoderFlag(
            input_ids_flag=temp_id_flags,
            nodes_flag=temp_nodes_flag,
            ans_special_id_idx=temp_batch_ans_idx,
            decoder_nodes_attention_mask=temp_nodes_attention_mask
        )

        collate_dict["attention_mask"] = collate_feature(batch_features_dicts, "attention_mask", 1)

        if (self.data_format == "format_2" or self.data_format == "format_3") and self.split != "train":
            collate_dict["decoder_input_ids"] = \
                collate_feature(batch_features_dicts, "decoder_input_ids", 0, "left")
            collate_dict["decoder_attention_mask"] = \
                collate_feature(batch_features_dicts, "decoder_attention_mask", 0, "left")
        else:
            collate_dict["decoder_input_ids"] = \
                collate_feature(batch_features_dicts, "decoder_input_ids", 0)
            collate_dict["decoder_attention_mask"] = \
                collate_feature(batch_features_dicts, "decoder_attention_mask", 0)
        return DataFeature(**collate_dict)

    def get_output_example(self, output_sentence: str) -> Dict[str, Union[List[str], str]]:
        assert isinstance(output_sentence, str), f"output_sentence is not a str"
        if QUESTION_SEP in output_sentence:
            sentence_splits = output_sentence.split(QUESTION_SEP)
            output_question = sentence_splits[-1]
            keyphrase_str = sentence_splits[0]
        else:
            output_question = " "
            keyphrase_str = output_sentence
        if KEYPHRASE_SEP in keyphrase_str:
            keyphrase_list = keyphrase_str.split(KEYPHRASE_SEP)
        else:
            keyphrase_list = [keyphrase_str]
        return {"keyphrase": keyphrase_list, 'question': output_question}

    def convert_instances_to_feature_tensors(self,
                                             decoder_start_token_id: int = 0) -> List[DataFeature]:
        features = []
        for idx, example in enumerate(tqdm(self.examples, desc=f"Tokenizing the dataset")):
            context = example.context
            positive_nodes = example.p_phrase
            negative_nodes = example.n_phrase
            _question = example.question
            _answer = example.answer
            _full_answer = example.full_answer

            # match node string in raw context
            if self.dataset_name == "cqg":
                positive_nodes_ = self._nodes_preprocess(context, positive_nodes)
                negative_nodes_ = self._nodes_preprocess(context, negative_nodes)
            else:
                positive_nodes_ = positive_nodes
                negative_nodes_ = negative_nodes
            # get the keyphrase string
            positive_nodes_list = [KEYPHRASE_SEP + " " + e for e in positive_nodes_]
            positive_nodes_string = " ".join(positive_nodes_list)

            # get the flags
            context_tokens = self.tokenizer.tokenize(context)
            context_tokens = self._remove_prefix_underline(context_tokens)
            positive_hit_flag = np.zeros(len(context_tokens), dtype=np.int32)
            positive_flags = self._extract_flags(context_tokens, positive_hit_flag, positive_nodes_, 1)
            negative_hit_flag = copy.deepcopy(positive_hit_flag)
            negative_flags = self._extract_flags(context_tokens, negative_hit_flag, negative_nodes_, 0)
            # check hit flag with bitwise and
            negative_hit_flag = positive_hit_flag ^ negative_hit_flag
            assert not np.any(positive_hit_flag.astype(np.int32) & negative_hit_flag.astype(np.int32))
            contex_hit_flag = (positive_hit_flag | negative_hit_flag * self.negative_hit_tag)

            if self.data_format == "format_1" or self.data_format == "format_2" or self.data_format == "format_3":
                # format_1: decoder template
                # format_2: keyphrase prompt both training and inference,
                #   not calculate the loss of prompt during training.
                # format_3: keyphrase prompt only in inference
                #   calculate the loss of prompt during training.
                input_string = context + " " + ANS_SEP + " " + _answer
                output_string = positive_nodes_string + " " + QUESTION_SEP + " " + _question

            elif self.data_format == "fa_model":
                # full answer trainer
                if _full_answer == "-1" and _answer.lower() in ["yes", "no"]:
                    output_string = _question
                elif _full_answer == "-1" and _answer.lower() not in ["yes", "no"]:
                    continue
                else:
                    output_string = _full_answer
                input_string = f"{ANS_SEP} {_answer} {CONTEXT_SEP} {context}"

            elif self.data_format == "full_answer_concat":
                fa_string = " ".join(_full_answer)
                input_string = f"{ANS_SEP} {_answer} {FULL_ANS_SEP} {fa_string} {CONTEXT_SEP} {context}"
                output_string = _question

            elif self.data_format == "q_model_upper":
                if _full_answer == "-1":
                    continue
                else:
                    input_string = f"{ANS_SEP} {_answer} {FULL_ANS_SEP} {_full_answer} {CONTEXT_SEP} {context}"
                    output_string = _question

            elif self.data_format in ["multi_factor", "top1_q_model"]:
                _fa = _full_answer[0]
                input_string = f"{ANS_SEP} {_answer} {FULL_ANS_SEP} {_fa} {CONTEXT_SEP} {context}"
                output_string = _question

            elif self.data_format == "pet":
                input_string = f"{ANS_SEP} {_answer} {CONTEXT_SEP} {context}"
                output_string = _question

            elif self.data_format in ["multi_factor_mixqg", "top1_q_model_mixqg"]:
                _fa = _full_answer[0]
                input_string = f"{_answer} //n {FULL_ANS_SEP} {_fa} {CONTEXT_SEP} {context}"
                output_string = _question

            elif self.data_format == "pet_mixqg":
                input_string = f"{_answer} //n {CONTEXT_SEP} {context}"
                output_string = _question

            elif self.data_format == "full2question_converter":
                if _full_answer == "-1":
                    continue
                else:
                    input_string = f"{ANS_SEP} {_answer} {FULL_ANS_SEP} {_full_answer}"
                    output_string = _question

            else:
                print(self.data_format)
                raise NotImplementedError()

            # get token ids
            input_ids = self.tokenizer.encode(input_string, add_special_tokens=True, truncation=True)
            label_ids = self.tokenizer.encode(output_string, add_special_tokens=True, truncation=True)
            decoder_input_ids = [decoder_start_token_id] + copy.deepcopy(label_ids[:-1])

            # update input ids flag and check the token id
            input_ids_flag = np.zeros(len(input_ids))
            len_context_tokens = len(context_tokens)
            if self.data_format != "full2question_converter":
                if self.data_format in ["format_1", "format_2", "format_3"]:
                    input_ids_flag[:len_context_tokens] = contex_hit_flag
                    positive_flags = self._after_process_flags(positive_flags, 0, input_ids)
                    negative_flags = self._after_process_flags(negative_flags, 0, input_ids)
                else:
                    # tokenize func has no <eos>, so the tokens len is len(input_ids) - 1
                    # in other words: len(input_ids) - 1 == len(input_string_tokens)
                    passage_sep_idx = input_ids.index(self.tokenizer.added_tokens_encoder[CONTEXT_SEP])
                    passage_len = (len(input_ids) - 1) - passage_sep_idx - 1
                    input_ids_flag[passage_sep_idx + 1: len(input_ids) - 1] = contex_hit_flag[:passage_len]
                    positive_flags = self._after_process_flags(positive_flags, passage_sep_idx + 1, input_ids)
                    negative_flags = self._after_process_flags(negative_flags, passage_sep_idx + 1, input_ids)

                    if self.data_format in ["multi_factor", "top1_q_model", "multi_factor_mixqg", "top1_q_model_mixqg"]:
                        # set all tokens of full answer as the important tokens
                        fa_sep_idx = input_ids.index(self.tokenizer.added_tokens_encoder[FULL_ANS_SEP])
                        input_ids_flag[fa_sep_idx + 1: passage_sep_idx] = 1

                if self.negative_hit_tag == -1:
                    input_ids_flag += 1

            ans_id_idx = input_ids.index(self.tokenizer.added_tokens_encoder[ANS_SEP]) \
                if ANS_SEP in input_string else 0
            decoder_flag = DecoderFlag(input_ids_flag=input_ids_flag,
                                       nodes_flag=(positive_flags + negative_flags),
                                       ans_special_id_idx=[ans_id_idx],
                                       decoder_nodes_attention_mask=[1] * len(positive_flags + negative_flags))

            # some checks when keyphrase on the start of decoder outputs
            if self.data_format in ["format_1", "format_2", "format_3"]:
                token_template = self.tokenizer.encode(positive_nodes_string, add_special_tokens=True,
                                                       truncation=True)
                prompt_mask_len = len(token_template)  # minus <eos> token and plus <question> token, so keep raw value
                assert label_ids[prompt_mask_len - 1] == self.tokenizer.added_tokens_encoder[QUESTION_SEP], \
                    f"check prompt: " \
                    f"{label_ids[prompt_mask_len - 1]} vs {self.tokenizer.added_tokens_encoder[QUESTION_SEP]}"

                if self.data_format == "format_2" or self.data_format == "format_3":
                    if self.split != "train":
                        decoder_input_ids = copy.deepcopy(label_ids[:prompt_mask_len])
                        assert prompt_mask_len == len(decoder_input_ids), \
                            f"dev/test dataset, " \
                            f"key phrase template prompt len: {prompt_mask_len}, " \
                            f"and decoder input ids len: {len(decoder_input_ids)}"
                    if self.data_format == "format_2":
                        # modify the label ids
                        # when prompting, do not calculate the loss of the keyphrase
                        for i in range(prompt_mask_len):
                            label_ids[i] = -100

            if self.data_format == "fa_dual_decoder":
                if _full_answer == "-1":
                    _full_answer_ids = [-100]
                else:
                    _full_answer_ids = self.tokenizer.encode(_full_answer, add_special_tokens=True, truncation=True)
            else:
                _full_answer_ids = label_ids
            features.append(DataFeature(input_ids=input_ids,
                                        decoder_flags=decoder_flag,
                                        attention_mask=[1] * len(input_ids),
                                        labels=label_ids,
                                        decoder_input_ids=decoder_input_ids,
                                        decoder_attention_mask=[1] * len(decoder_input_ids),
                                        decoder_full_answer_labels=_full_answer_ids))
        return features

    @staticmethod
    def _nodes_preprocess(text: str, phrase: List[str]):
        phrase_list = []
        _phrase_match = []
        for e in phrase:
            try:
                _phrase_match.append(re.finditer(f"{e}", f"{text}", re.I))
            except:
                # print("phrase: ", e)
                phrase_list.append(e)
                continue
        for match in _phrase_match:
            for phrase_span in match:
                try:
                    phrase_start, phrase_end = phrase_span.span()
                    temp_e = text[phrase_start:phrase_end]
                    if temp_e not in phrase_list:
                        phrase_list.append(temp_e)
                finally:
                    pass
        # token_phrase_list = [self.tokenizer.tokenize(ner) for ner in phrase_list]
        return phrase_list

    def _extract_flags(self,
                       context_token_list: List[str],
                       if_has_matched: np.ndarray,
                       phrase_list: List[str],
                       tag) -> List[NodeFlag]:
        flags: List[NodeFlag] = []
        nodes_tokens_list = [self._remove_prefix_underline(self.tokenizer.tokenize(ner)) for ner in phrase_list if
                             len(ner) > 0]
        for ei, e in enumerate(nodes_tokens_list):
            idx = -1
            for i in range(len(context_token_list)):
                idx = i
                phrase_match_bool = True
                for j in range(len(e)):
                    try:
                        if (e[j].lower() != context_token_list[i + j].lower()) or \
                                (if_has_matched[i + j] == 1):
                            phrase_match_bool = False
                            break
                    except:
                        phrase_match_bool = False
                        break
                if phrase_match_bool is True:
                    if_has_matched[idx: idx + len(e)] = 1
                    flags.append(NodeFlag(index_range=np.arange(idx, idx + len(e)),
                                          token_ids=self.tokenizer.encode(phrase_list[ei])[:-1],
                                          tag=[tag]))
                    break
        return flags

    def check_feature_flags(self, check_file: str):
        check_flag_bool = True
        check_list = []
        count = 0
        for feature_i in tqdm(self._features, desc=f"Checking the {self.split} dataset: "):
            check_flags: List[NodeFlag] = feature_i.decoder_flags.nodes_flag
            if len(check_flags) > 0:
                for node_flag in check_flags:
                    check_start = node_flag.index_range[0]
                    check_end = node_flag.index_range[-1] + 1
                    if not np.all(
                            np.asarray(feature_i.input_ids[check_start:check_end]) == np.asarray(node_flag.token_ids)):
                        feature_string = self.tokenizer.decode(feature_i.input_ids[check_start:check_end])
                        flag_string = self.tokenizer.decode(node_flag.token_ids)
                        if feature_string.lower() != flag_string.lower():
                            check_list.append({"feature": feature_string, "flag": flag_string})
                            count += 1
                        # print("==============")
                        # print(np.asarray(feature_i.input_ids[check_start:check_end]))
                        # print(node_flag.token_ids)
                        # check_flag_bool = False
        with open(check_file, 'w') as f:
            f.write(json.dumps(check_list, indent=4))
        print(count)
        return check_flag_bool

    def _after_process_flags(self, node_flags: List[NodeFlag], start_idx, input_ids, max_length=MAX_LENGTH) -> List[
        NodeFlag]:
        """
        1. remove flags that longer than MAX_LENGTH, so even the start idx is 0, we also need the after-process
        2. shift the flags, align with the passage start special token idx
        """
        new_nodes = []

        def check_ids(idx_range, raw_token_ids):
            for i, raw_token_id in zip(idx_range, raw_token_ids):
                if int(input_ids[i]) == int(raw_token_id):
                    pass
                else:
                    return False
            return True

        for node in node_flags:
            assert isinstance(node, NodeFlag)
            node.index_range += start_idx
            # assert check_ids(node.index_range, node.token_ids)
            node.token_ids = input_ids[node.index_range[0]: node.index_range[-1] + 1]
            if node.index_range[-1] >= max_length:
                continue
            else:
                assert len(node.token_ids) == len(node.index_range), f"{node.token_ids}, {node.index_range}"
                new_nodes.append(node)
        return new_nodes

    def _remove_prefix_underline(self, token_list: List[str]) -> List[str]:
        """
           remove the token string prefix "_" after sentence piece
        """
        assert isinstance(token_list[0], str)
        n = len(token_list)
        for idx in range(n):
            temp_str = token_list[idx]
            if token_list[idx] == chr(9601) or not token_list[idx].startswith(chr(9601)):
                continue
            else:
                token_list[idx] = token_list[idx][1:]
        return token_list


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_name', type=str, default='cqg', help='choice the dataset')
    parser.add_argument('-j', '--data_format', type=str, default='format_16_top1', help='choice the dataset format')
    parser.add_argument('-l', '--max_length', type=int, default=512, help='max input length')
    parser.add_argument('-p', '--dataset_prefix', type=str, default=str(DATASET_PREFIX), help='choice the dataset format')
    MAX_LENGTH = args.max_length
    args, remaining_args = parser.parse_known_args()
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    tokenizer = T5Tokenizer.from_pretrained("t5-base", use_fast=False, model_max_length=MAX_LENGTH)
    tokenizer.add_tokens([ANS_SEP, KEYPHRASE_SEP, QUESTION_SEP, FULL_ANS_SEP, CONTEXT_SEP], special_tokens=True)
    data_name = args.data_name
    data_path_prefix = args.dataset_prefix
    data_path = os.path.join(data_path_prefix, f"dataset/{data_name}")
    splits = ["dev", "test", "train"]

    for split in splits:
        dataset = MyDataset(data_path, split, args.data_format, tokenizer)
        try:
            os.mkdir(os.path.join(data_path, f"{args.data_format}"))
        except:
            pass
        dataset.save_dataset(os.path.join(data_path, f"{args.data_format}/{split}.pt"))
        check_save_path = os.path.join(data_path_prefix, f"experiments/cqg_{args.data_format}_{split}.jsonl")
        dataset.check_feature_flags(check_save_path)
        print(dataset[0])
        print(dataset.examples[0])
        print(f"==============={len(dataset._features)}")
        print(f"==============={len(dataset.examples)}")


if __name__ == "__main__":
    """
    """
    main()
