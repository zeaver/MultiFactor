__all__ = ["MyTrainer", "MyOptimizer", "evaluate_data"]

import math
import os.path
import random
import numpy as np
import datetime
from tqdm import tqdm
from dataclasses import asdict, dataclass
import collections
from typing import Union, Optional, Tuple, Any
import json
import logging
from inspect import getfullargspec

import nltk
from rouge import Rouge

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from transformers import PreTrainedTokenizerBase, PreTrainedModel, Adafactor
from transformers.optimization import get_scheduler
from transformers.training_args import OptimizerNames
try:
    from .arguments import ModelArguments, DataTrainingArguments, TrainingArguments
    from .multi_factor_dataset import MyDataset
    from .modeling_bridget5 import BridgeT5
except:
    from MultiFactor.arguments import ModelArguments, DataTrainingArguments, TrainingArguments
    from MultiFactor.multi_factor_dataset import MyDataset
    from MultiFactor.modeling_bridget5 import BridgeT5

Metrics = collections.namedtuple('metric', 'bleu4 rouge_l meteor')


class MyTrainer:
    """
    The Trainer of SKI-QG
    """

    def __init__(self,
                 model: Union[PreTrainedModel, torch.nn.Module, BridgeT5] = None,
                 tokenizer: Union[PreTrainedTokenizerBase] = None,
                 dataset: Optional[Dataset] = None,
                 training_dir: Optional[str] = None,
                 training_args: Optional[TrainingArguments] = None,
                 trainer_device=None,
                 dev_dataset: Optional[Dataset] = None,
                 test_dataset: Optional[Dataset] = None,
                 data_args: Optional[DataTrainingArguments] = None,
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.device = trainer_device

        self.data_training_args = data_args
        self.training_args = training_args
        self.training_args.max_steps = self.calculate_max_train_step()

        # loss
        self.supervised_learning_loss_fct = CrossEntropyLoss(ignore_index=-100)

        # self.optimizer = Adam(model.parameters(), lr=self.training_args.learning_rate)
        __optimizer = MyOptimizer(self.training_args)
        self.optimizer = __optimizer.optimizer_cls(model.parameters(), **__optimizer.optimizer_kwargs)
        self.scheduler = __optimizer.create_scheduler(self.training_args.max_steps, self.optimizer)
        # self.scheduler = None

        self.global_steps = 0
        self.experiment_dir = training_dir
        self.writer = SummaryWriter(training_dir, comment=f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def calculate_max_train_step(self):
        if self.training_args.max_steps > 0:
            return self.training_args.max_steps
        else:
            assert len(self.dataset) > 0
            len_dataloader = len(self.dataset)
            total_train_batch_size = self.training_args.train_batch_size * \
                                     self.training_args.gradient_accumulation_steps * \
                                     self.training_args.world_size
            num_update_steps_per_epoch = len_dataloader // total_train_batch_size
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            max_steps = math.ceil(self.training_args.num_train_epochs * num_update_steps_per_epoch)
            return max_steps

    def train(self, data_format, **kwargs):
        # single gpu
        if "data_collater" not in kwargs:
            train_dataset_loader = DataLoader(self.dataset,
                                            batch_size=self.training_args.train_batch_size,
                                            shuffle=False,
                                            collate_fn=self.dataset.collate_fn)
        else:
            train_dataset_loader = DataLoader(self.dataset,
                                            batch_size=self.training_args.train_batch_size,
                                            shuffle=False,
                                            collate_fn=kwargs["data_collater"])
        best_bleu4 = 0
        logging.info("Start training")
        for epoch_id in range(int(self.training_args.num_train_epochs)):
            save_dir_epoch = os.path.join(self.experiment_dir, f"epoch_{epoch_id}")
            try:
                os.mkdir(save_dir_epoch)
            except:
                pass
            self.model.train()
            for batch_info in tqdm(train_dataset_loader, desc=f"Epoch {epoch_id}"):
                batch_labels = batch_info.labels.to(self.device)
                batch_input_ids = batch_info.input_ids.to(self.device)
                batch_attention_mask = batch_info.attention_mask.to(self.device)
                batch_decoder_input_ids = batch_info.decoder_input_ids.to(self.device) if data_format == "format_2" or data_format == "format_3"  else None
                batch_decoder_attention_mask = batch_info.decoder_attention_mask.to(self.device) if data_format == "format_2" or data_format == "format_3" else None
                other_model_kwargs = {}
                if "input_flags" in getfullargspec(self.model.forward).args:
                    batch_flags = batch_info.decoder_flags
                    batch_flags.flag2tensor(self.device)
                    other_model_kwargs.update({"input_flags":batch_flags})
                if "decoder_full_answer_labels" in getfullargspec(self.model.forward).args:
                    other_model_kwargs.update({"decoder_full_answer_labels":batch_info.decoder_full_answer_labels.to(self.device)})
                batch_output = self.model(input_ids=batch_input_ids,
                                          labels=batch_labels,
                                          attention_mask=batch_attention_mask,
                                          decoder_input_ids=batch_decoder_input_ids,
                                          decoder_attention_mask = batch_decoder_attention_mask,
                                          **other_model_kwargs)
                loss = batch_output.loss

                self.writer.add_scalar('train/loss', loss, self.global_steps)
                if hasattr(batch_output, "cls_loss"):
                    self.writer.add_scalar('train/Cls-loss', batch_output.cls_loss, self.global_steps)
                    self.writer.add_scalar('train/Gen-loss', batch_output.gen_loss, self.global_steps)

                if hasattr(batch_output, "q_loss"):
                    if batch_output.q_loss is not None:
                        self.writer.add_scalar('train/q-loss', batch_output.q_loss, self.global_steps)

                if hasattr(batch_output, "fa_loss"):
                    if batch_output.fa_loss is not None:
                        self.writer.add_scalar('train/fa-loss', batch_output.fa_loss, self.global_steps)

                self.writer.add_scalar('train/learning rate', self.optimizer.state_dict()['param_groups'][0]['lr'],
                                       self.global_steps)
                self.writer.flush()
                self.global_steps += 1
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
            if epoch_id >= 1:
                evaluate_data_result = evaluate_data(self.tokenizer, self.training_args, self.data_training_args,
                                                     self.dev_dataset, self.model, self.device,
                                                     save_evaluate_result=True,
                                                     save_dir=save_dir_epoch, data_format=data_format,
                                                     **kwargs)
                test_data_result = evaluate_data(self.tokenizer, self.training_args, self.data_training_args,
                                                 self.test_dataset, self.model, self.device,
                                                 save_evaluate_result=True,
                                                 save_dir=save_dir_epoch, data_format=data_format,
                                                 **kwargs)

                # recode metric result in tensorboard
                for metric_k, metric_v in evaluate_data_result._asdict().items():
                    self.writer.add_scalar('metric_dev/' + metric_k, metric_v, epoch_id)
                self.writer.flush()

                for metric_k, metric_v in test_data_result._asdict().items():
                    self.writer.add_scalar('metric_test/' + metric_k, metric_v, epoch_id)
                self.writer.flush()

                if test_data_result.bleu4 > best_bleu4:
                    best_bleu4 = test_data_result.bleu4
                    save_path = os.path.join(save_dir_epoch, f"model_{best_bleu4}")
                    try:
                        os.mkdir(save_path)
                    except:
                        pass
                if self.training_args.save_model_pt and epoch_id >= 2:
                    save_path = os.path.join(save_dir_epoch, f"model")
                    try:
                        os.mkdir(save_path)
                        self.model.save_pretrained(save_path)
                        self.tokenizer.save_pretrained(save_path)
                    except:
                        pass


def evaluate_data(evaluate_tokenizer: PreTrainedTokenizerBase = None,
                  evaluate_config: TrainingArguments = None,
                  evaluate_data_config: DataTrainingArguments = None,
                  evaluate_dataset: Dataset = None,
                  evaluate_model: PreTrainedModel = None,
                  evaluate_device: torch.device = None,
                  save_evaluate_result: bool = False,
                  save_dir: str = None, data_format=None, **kwargs):
    _eval_batch_size = evaluate_config.eval_batch_size
    if "data_collater" not in kwargs:
        evaluate_dataset_loader = DataLoader(evaluate_dataset, batch_size=_eval_batch_size,
                                            shuffle=False,
                                            collate_fn=evaluate_dataset.collate_fn)
    else:
        evaluate_dataset_loader = DataLoader(evaluate_dataset, batch_size=_eval_batch_size,
                                            shuffle=False,
                                            collate_fn=kwargs["data_collater"])
    evaluate_model.eval()
    corpus_input_ids = []
    corpus_input_text = []
    corpus_generate_ids = []
    corpus_label_ids = []
    raw_text = []
    raw_labels_text = []
    predictions_text = []
    targets_text = []
    example_id = 0
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        for batch_info in tqdm(evaluate_dataset_loader, desc=f"Evaluating {evaluate_dataset.split} {evaluate_dataset.data_format}", ):
            # prediction and decode
            if data_format == "format_2" or data_format == "format_3" :
                generate_kws = {'max_length': evaluate_data_config.max_seq_length_eval,
                                'decoder_input_ids': batch_info.decoder_input_ids.to(evaluate_device),
                                'decoder_attention_mask':batch_info.decoder_attention_mask.to(evaluate_device)}
            else:
                generate_kws = {'max_length': evaluate_data_config.max_seq_length_eval}
            if "input_flags" in getfullargspec(evaluate_model.forward).args:
                batch_flags = batch_info.decoder_flags
                batch_flags.flag2tensor(evaluate_device)
                # batch_flags.expand_for_generation(evaluate_data_config.num_beams)
                generate_kws["input_flags"]=batch_flags
            generation_ids = evaluate_model.generate(
                batch_info.input_ids.to(evaluate_device),
                num_beams=evaluate_data_config.num_beams,
                **generate_kws).to(evaluate_device)
            model_input_ids = batch_info.input_ids
            model_input_text = [evaluate_tokenizer.decode(input_ids_i, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                                for input_ids_i in batch_info.input_ids]
            response = [evaluate_tokenizer.decode(g_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        for g_id in generation_ids]
            if data_format == "format_2" or data_format == "format_3" :
                question_text = [
                    evaluate_tokenizer.decode(g_id.masked_fill_(g_id == -100, evaluate_model.config.pad_token_id),
                                              skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    for g_id in batch_info.labels]
                keyphrase_text = [
                    evaluate_tokenizer.decode(g_id,
                                              skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    for g_id in batch_info.decoder_input_ids]
                labels_text = [question_text[i] + ' ' + keyphrase_text[i] for i in range(len(question_text))]

            else:
                labels_text = [
                    evaluate_tokenizer.decode(g_id.masked_fill_(g_id == -100, evaluate_model.config.pad_token_id),
                     skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    for g_id in batch_info.labels]

            
            if data_format in["format_1", "format_2", "format_3", "uniqg"]:
                response_questions = [evaluate_dataset.get_output_example(response_question)["question"]
                                      for response_question in response]
                target_questions = [evaluate_dataset.get_output_example(l)["question"] for l in labels_text]
            else:
                response_questions = response
                target_questions = labels_text
            # add info
            corpus_input_ids += model_input_ids.cpu().detach().tolist()
            corpus_input_text += model_input_text
            corpus_generate_ids += generation_ids.cpu().detach().tolist()
            corpus_label_ids += batch_info.labels.cpu().detach().tolist()
            raw_text += response
            raw_labels_text += labels_text
            predictions_text += response_questions
            targets_text += target_questions
            example_id += _eval_batch_size
    preds = []
    golds = []
    meteor_score = 0.0

    rouge_score = 0.0
    rouge = Rouge()
    num = len(predictions_text)
    for pred, gold in zip(predictions_text, targets_text):
        if len(pred.strip()) > 0:
            rouge_score += rouge.get_scores([pred], [gold])[0]['rouge-l']['f']
            pred = nltk.word_tokenize(pred)
            gold = nltk.word_tokenize(gold)
            preds.append(pred)
            golds.append([gold])
            # print(gold)
            meteor_score += nltk.translate.meteor_score.meteor_score([gold], pred)
        else:
            rouge_score += 0
            pred = " "
            gold = " "
            preds.append(pred)
            golds.append([gold])
            # print(gold)
            meteor_score += 0
    bleu_score = nltk.translate.bleu_score.corpus_bleu(golds, preds)
    rouge_score = rouge_score / num
    meteor_score = meteor_score / num
    result = Metrics(bleu4=bleu_score, rouge_l=rouge_score, meteor=meteor_score)
    logging.info(f"{evaluate_dataset.split}: {result}")

    if save_evaluate_result and save_dir:
        info_list = [evaluate_tokenizer.added_tokens_encoder]
        for in_ids, in_text, g_ids, l_ids, r, l, p, g \
            in zip(corpus_input_ids, corpus_input_text, 
                    corpus_generate_ids, corpus_label_ids,
                    raw_text, raw_labels_text,
                    predictions_text, targets_text):
            info_list.append({"input_ids": ' '.join(map(str, in_ids)), "input_text": in_text,
                                'generated ids': ' '.join(map(str, g_ids)), 'label ids': " ".join(map(str, l_ids)),
                                'model_output': r, 'model_label': l,
                                'pred': p, 'gold': g})
        with open(save_dir + f"/evaluate_{evaluate_dataset.split}.jsonl", 'w') as f:
            json.dump(info_list, f, indent=4)
    return result


class MyOptimizer:
    """
    ref:
        https://github.com/huggingface/transformers/blob/v4.19.2/src/transformers/trainer.py
    """

    def __init__(self, training_args: TrainingArguments):
        self.args = training_args
        self.optimizer_cls, self.optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        self.lr_scheduler = None
        self.optimizer = None

    @staticmethod
    def get_optimizer_cls_and_kwargs(args: TrainingArguments) -> Tuple[Any, Any]:
        """
        Returns the optimizer class and optimizer parameters based on the training arguments.
        Args:
            args (`transformers.training_args.TrainingArguments`):
                The training arguments for the training session.
        """
        optimizer_kwargs = {"lr": args.learning_rate}
        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }
        if args.optim == OptimizerNames.ADAFACTOR:
            optimizer_cls = Adafactor
            optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        elif args.optim == OptimizerNames.ADAMW_HF:
            from transformers.optimization import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
        elif args.optim == OptimizerNames.ADAMW_TORCH:
            from torch.optim import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
        elif args.optim == OptimizerNames.ADAMW_TORCH_XLA:
            try:
                from torch_xla.amp.syncfree import AdamW

                optimizer_cls = AdamW
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer failed to import syncfree AdamW from torch_xla.")
        elif args.optim == OptimizerNames.ADAMW_APEX_FUSED:
            try:
                from apex.optimizers import FusedAdam

                optimizer_cls = FusedAdam
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate apex FusedAdam but apex is not installed!")
        elif args.optim == OptimizerNames.ADAMW_BNB:
            try:
                from bitsandbytes.optim import Adam8bit

                optimizer_cls = Adam8bit
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate bnb Adam8bit but bnb is not installed!")
        elif args.optim == OptimizerNames.SGD:
            optimizer_cls = torch.optim.SGD
        elif args.optim == OptimizerNames.ADAGRAD:
            optimizer_cls = torch.optim.Adagrad
        else:
            raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")
        return optimizer_cls, optimizer_kwargs

    def create_scheduler(self, num_training_steps: int, input_optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.
        Args:
            num_training_steps (int): The number of training steps to do.
        """
        assert input_optimizer is not None
        assert self.args.max_steps > 0

        warmup_steps = (
            self.args.warmup_steps  # 先使用steps
            if self.args.warmup_steps > 0
            else math.ceil(num_training_steps * self.args.warmup_ratio)
        )

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=input_optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

        return lr_scheduler
