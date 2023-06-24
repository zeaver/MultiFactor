__all__ = ["MyOutput", "MultiFactorConfig", "MyEncoderOutput", "PhraseEncoderOutput", "FullAnswerDecoderOutput"]

from typing import Optional, Tuple, Union, List
from dataclasses import dataclass

import torch
from transformers.modeling_outputs import Seq2SeqLMOutput, ModelOutput, BaseModelOutputWithPastAndCrossAttentions

try:
    from multi_factor_dataset import NodeFlag, DecoderFlag
except:
    from .multi_factor_dataset import NodeFlag, DecoderFlag

# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
_HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


@dataclass
class MultiFactorConfig:
    model_type: str = 'Baseline'
    # phrase classifier
    if_node_cls: bool = True
    cls_loss_weight: float = 1.0
    cls_type: str = "node"
    cls_has_answer: bool = False

    # cross attention module switch
    if_cross_enhanced_k: bool = False
    # if_phrase_cross: bool = False
    # if_cross_enhanced_q: bool = False
    if_full_answer_decoder: bool = False
    fa_loss_weight: float = 0.2

    # training
    model_freeze: bool = False
    hard_flag: int = 2

    def __post_init__(self):
        self._post_check_model_type()
        self._post_check_node_cls()
        flags = ["00", "01", "10", "11"]
        self.hard_flag_bin = flags[self.hard_flag]

    def _post_check_model_type(self):
        import re
        if re.match(r"Baseline", self.model_type, re.I):
            self.model_type = "Baseline"
            self.if_node_cls = False
            self.if_cross_enhanced_k = False
            self.if_full_answer_decoder = False
        elif re.match(r"node\S*cls", self.model_type, re.I):
            self.model_type = "NodeCls"
            # assert self.if_node_cls, f"Please set the multifactor config: if_node_cls"
            self.if_node_cls = True
            self.if_cross_enhanced_k = False
            self.if_full_answer_decoder = False
        elif re.match(r"MultiFactor", self.model_type, re.I):
            self.model_type = "MultiFactor"
            self.if_node_cls = True
            self.if_cross_enhanced_k = True
            self.if_full_answer_decoder = False
        elif re.match(r"fad", self.model_type, re.I):
            self.model_type = "FADecoder"
            # self.if_node_cls = True
            # self.if_cross_enhanced_k = True
            self.if_full_answer_decoder = True
        else:
            raise ValueError(
                f"Except model type in 'baseline', 'NodeCls', 'EnhancedEncoder' or 'multifactor', actually {self.model_type}")

    def _post_check_node_cls(self):
        assert self.cls_type == "node" or (self.cls_type == "id" and self.cls_has_answer)
        assert 4 > self.hard_flag >= 0
        if not self.if_node_cls or self.cls_loss_weight <= 0:
            self.if_node_cls = False
            self.cls_loss_weight = 0.0
        assert 0.0 <= self.cls_loss_weight, \
            f"super parameter: the weight of node classification is supposed to be larger than 0.0,\
                 {self.cls_loss_weight} is illegal"

    @property
    def task_prefix_str(self) -> str:
        if self.model_type == "Baseline":
            return "Base"
        elif self.model_type == "NodeCls":
            return f"Cls_{self.cls_loss_weight}_{self.cls_type}_{self.cls_has_answer}"
        elif self.model_type == "MultiFactor":
            return f"Cls_{self.cls_loss_weight}_{self.cls_type}_{self.cls_has_answer}" \
                   f"__freeze_{self.model_freeze}" \
                   f"__hard_{self.hard_flag_bin}"
        else:
            return f"fa_weight_{self.fa_loss_weight}"


@dataclass
class MyOutput(Seq2SeqLMOutput):
    cls_loss: Union[None, torch.FloatTensor, float] = None
    q_loss: Union[None, torch.FloatTensor, float] = None
    gen_loss: Union[None, torch.FloatTensor, float] = None
    fa_loss: Union[None, torch.FloatTensor, float] = None
    pred_negative_nodes: Union[None, List[str], torch.Tensor] = None
    pred_positive_nodes: Union[None, List[str], torch.Tensor] = None


@dataclass
class MyEncoderOutput(BaseModelOutputWithPastAndCrossAttentions):
    cls_loss: Union[None, torch.FloatTensor, float] = None
    pred_negative_nodes: Union[None, List[str], torch.Tensor] = None
    pred_positive_nodes: Union[None, List[str], torch.Tensor] = None
    bridge_input: Optional[torch.FloatTensor] = None
    # mixed_phrase_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
class PhraseEncoderOutput(ModelOutput):
    cls_loss: Union[None, torch.FloatTensor, float] = None
    pred_negative_nodes: Union[None, List[str], torch.Tensor] = None
    pred_positive_nodes: Union[None, List[str], torch.Tensor] = None
    bridge_input: Optional[torch.FloatTensor] = None
    # mixed_phrase_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
class FullAnswerDecoderOutput(BaseModelOutputWithPastAndCrossAttentions):
    fa_loss: Union[None, torch.FloatTensor, float] = None
