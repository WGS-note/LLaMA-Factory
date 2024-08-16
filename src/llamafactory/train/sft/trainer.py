# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
import shutil
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimzer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, DataArguments


logger = get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], data_args: "DataArguments", **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.data_args = data_args
        self.writer = SummaryWriter(log_dir=self.args.logging_dir, flush_secs=120)   # TODO: 【√】draw channels loss

        if os.path.exists(self.args.logging_dir):
            shutil.rmtree(self.args.logging_dir)

        # 梯度累积，training_step 会调用多次，手动维护一个累积字典，用于画图和源码对齐
        if self.data_args.channel_loss:
            self.cumulative_dict = {
                "cumulative_loss": 0.0,
                "accumulated_steps": 0,
                **{v: 0.0 for _, v in self.data_args.channel_loss.items()}  # channel 的累积
            }
            # e.g.: self.data_args.channel_loss:  {'channel_test_semantic_20240808': 0, 'channel_test_evaluation_good_20240808': 1, 'channel_test_evaluation_general_20240808': 2}
            self.print_debug_info(f"[DEBUG] cumulative_dict init: {self.cumulative_dict}")

        # 多卡时，避免梯度累积大于每卡的steps
        if dist.is_initialized():
            num_examples = self.num_examples(self.get_train_dataloader())
            total_train_batch_size = self._train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
            if total_train_batch_size > num_examples // 3:
                tmp = num_examples // self.args.world_size
                self.print_debug_info(f"[DEBUG] gradient_accumulation_steps 由 {self.args.gradient_accumulation_steps} 变为 {tmp}")
                self.args.gradient_accumulation_steps = tmp

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"].detach().clone() if "labels" in inputs else None  # backup labels
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: torch.Tensor, tgt_tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))

            writer.write("\n".join(res))

    # ========== 增加 channel loss 的支持 ==========
    def train(self, *args, **kwargs):
        """重载 train 方法以使用自定义的 compute_loss
        """
        if self.data_args.channel_loss:
            self.print_debug_info("[DEBUG] 重载 train 方法以使用自定义的 compute_loss")

        return super().train(*args, **kwargs)

    def training_step(self, model, inputs):
        """梯度累积会调用多次"""

        if self.data_args.channel_loss:
            channels = inputs.pop("channels", None)

        loss = super().training_step(model, inputs)

        if channels is not None:
            self.training_step_end(loss, channels)

        return loss

    def training_step_end(self, loss, channels):
        # 累积总损失
        self.cumulative_dict["cumulative_loss"] += loss.item()
        self.cumulative_dict["accumulated_steps"] += 1

        # 累积各个 channel 的损失
        for channel in channels:
            self.cumulative_dict[channel.item()] += loss.item()

        if self.cumulative_dict["accumulated_steps"] % self.args.gradient_accumulation_steps == 0:
            if dist.is_initialized():
                # 同步和汇总总损失
                cumulative_loss_tensor = torch.tensor(self.cumulative_dict["cumulative_loss"]).to('cuda')
                dist.all_reduce(cumulative_loss_tensor, op=dist.ReduceOp.SUM)
                self.cumulative_dict["cumulative_loss"] = cumulative_loss_tensor.item() / dist.get_world_size()


                for channel in channels:
                    loss_name = [k for k, v in self.data_args.channel_loss.items() if v == channel.item()][0]
                    print(f"channel: {channel.item()} loss_name: {loss_name} GPU: {dist.get_rank()} Step: {self.state.global_step + 1}, Loss: {self.cumulative_dict[channel.item()]}, accumulated_steps: {self.cumulative_dict['accumulated_steps']}")



                if dist.get_rank() == 0:
                    # for channel in self.cumulative_dict:
                    #     if channel in ["cumulative_loss", "accumulated_steps"]:
                    #         continue  # 跳过非 channel 的键

                    #     # 汇总每个 channel 的损失
                    #     cumulative_channel_loss_tensor = torch.tensor(self.cumulative_dict[channel]).to('cuda')
                    #     dist.all_reduce(cumulative_channel_loss_tensor, op=dist.ReduceOp.SUM)
                    #     self.cumulative_dict[channel] = cumulative_channel_loss_tensor.item() / dist.get_world_size()

                    #     # 记录到 TensorBoard
                        # loss_name = [k for k, v in self.data_args.channel_loss.items() if v == channel.item()][0]
                    #     self.writer.add_scalar(f"train/channel_loss_{loss_name}", self.cumulative_dict[channel], self.state.global_step)
                        # print(f"channel: {channel.item()} loss_name: {loss_name} GPU: {dist.get_rank()} Step: {self.state.global_step + 1}, Loss: {self.cumulative_dict[channel.item()]}, accumulated_steps: {self.cumulative_dict['accumulated_steps']}")

                    self.writer.add_scalar("train/train_loss", self.cumulative_dict["cumulative_loss"], self.state.global_step + 1)
                    print("-----", self.cumulative_dict["cumulative_loss"], self.state.global_step + 1, self.cumulative_dict["accumulated_steps"], self.args.gradient_accumulation_steps)

            else:
                for channel in channels:
                    loss_name = [k for k, v in self.data_args.channel_loss.items() if v == channel.item()][0]
                    self.writer.add_scalar(f"train/channel_loss_{loss_name}", self.cumulative_dict[channel.item()], self.state.global_step)
                    print(f"loss_name: {loss_name} Step: {self.state.global_step + 1}, Loss: {self.cumulative_dict[channel.item()]}, accumulated_steps: {self.cumulative_dict['accumulated_steps']}")

                self.writer.add_scalar("train/train_loss", self.cumulative_dict["cumulative_loss"], self.state.global_step + 1)
                print("-----", self.cumulative_dict["cumulative_loss"], self.state.global_step + 1, self.cumulative_dict["accumulated_steps"], self.args.gradient_accumulation_steps)

            # 重置累积值
            self.cumulative_dict["cumulative_loss"] = 0.0
            self.cumulative_dict["accumulated_steps"] = 0
            for channel in self.cumulative_dict:
                if channel not in ["cumulative_loss", "accumulated_steps"]:
                    self.cumulative_dict[channel] = 0.0




    def compute_loss(self, model, inputs, return_outputs=False):
        if self.data_args.channel_loss:
            _ = inputs.pop("channels", None)

        return super().compute_loss(model, inputs, return_outputs)

    # def evaluate(self, metric_key_prefix="eval", **kwargs):
    #     if self.data_args.channel_loss: print("[DEBUG] 重载 evaluate 方法, 增加 channel loss 的支持")
    #     eval_results = super().evaluate(metric_key_prefix=metric_key_prefix, **kwargs)

    # def log(self, key, value):
    #     if self.tb_writer:
    #         self.tb_writer.add_scalar(key, value, self.state.global_step)

    def print_debug_info(self, message):
        if dist.is_initialized():
            # 多卡环境时, 在rank0打印
            if self.is_local_process_zero():
                print(message)
        else:
            print(message)


    # def training_step_end(self, loss, channels):
    #     # 累积损失
    #     self.cumulative_dict["cumulative_loss"] += loss.item()
    #     self.cumulative_dict["accumulated_steps"] += 1

    #     # # 直白来讲，channel 其实就是一个标识不同数据类型的整数
    #     # for channel in channels:   # batch_size为1，e.g.: tensor([[0]], device='cuda:0')
    #     #     self.cumulative_dict[channel.item()] += loss.item()

    #     if self.cumulative_dict["accumulated_steps"] % self.args.gradient_accumulation_steps == 0:
    #         # if self.state.global_step % self.args.logging_steps == 0:

    #         # for channel in channels:
    #         #     loss_name = [k for k, v in self.data_args.channel_loss.items() if v == channel.item()][0]
    #         #     self.writer.add_scalar(f"train/channel_loss_{loss_name}", self.cumulative_dict[channel.item()], self.state.global_step)
    #         #     print(" channel -----", self.cumulative_dict[channel.item()], self.state.global_step)

    #         self.writer.add_scalar("train/train_loss", self.cumulative_dict["cumulative_loss"], self.state.global_step)
    #         print("-----", self.cumulative_dict["cumulative_loss"], self.state.global_step)

    #         # 重置累积值
    #         self.cumulative_dict["cumulative_loss"] = 0.0
    #         self.cumulative_dict["accumulated_steps"] = 0
    #         # for channel in channels:
    #         #     self.cumulative_dict[channel.item()] = 0.0

    # def training_step_end(self, loss):
    # # 累积损失
    # self.cumulative_dict["cumulative_loss"] += loss.item()
    # self.cumulative_dict["accumulated_steps"] += 1

    # # gpu_rank = dist.get_rank() if dist.is_initialized() else 0
    # # print(f"GPU: {gpu_rank}, Step: {self.state.global_step}, Loss: {loss.item()}, Step: {self.cumulative_dict["accumulated_steps"]}")

    # if self.cumulative_dict["accumulated_steps"] % self.args.gradient_accumulation_steps == 0:

    #     if dist.is_initialized():
    #         cumulative_loss_tensor = torch.tensor(self.cumulative_dict["cumulative_loss"]).to('cuda')
    #         dist.all_reduce(cumulative_loss_tensor, op=dist.ReduceOp.SUM)
    #         self.cumulative_dict["cumulative_loss"] = cumulative_loss_tensor.item() / dist.get_world_size()

    #         if dist.get_rank() == 0:
    #             self.writer.add_scalar("train/train_loss", self.cumulative_dict["cumulative_loss"], self.state.global_step + 1)
    #             print("-----", self.cumulative_dict["cumulative_loss"], self.state.global_step + 1, self.cumulative_dict["accumulated_steps"], self.args.gradient_accumulation_steps)

    #     else:
    #         self.writer.add_scalar("train/train_loss", self.cumulative_dict["cumulative_loss"], self.state.global_step + 1)
    #         print("-----", self.cumulative_dict["cumulative_loss"], self.state.global_step + 1, self.cumulative_dict["accumulated_steps"], self.args.gradient_accumulation_steps)

    #     # 重置累积值
    #     self.cumulative_dict["cumulative_loss"] = 0.0
    #     self.cumulative_dict["accumulated_steps"] = 0




    # def training_step_end(self, loss, channels):
    # # 累积损失
    # self.cumulative_dict["cumulative_loss"] += loss.item()
    # self.cumulative_dict["accumulated_steps"] += 1

    # # gpu_rank = dist.get_rank() if dist.is_initialized() else 0
    # # print(f"GPU: {gpu_rank}, Step: {self.state.global_step}, Loss: {loss.item()}, Step: {self.cumulative_dict["accumulated_steps"]}")

    # # 直白来讲，channel 其实就是一个标识不同数据类型的整数
    # for channel in channels:   # batch_size为1，e.g.: tensor([[0]], device='cuda:0')
    #     self.cumulative_dict[channel.item()] += loss.item()

    # if self.cumulative_dict["accumulated_steps"] % self.args.gradient_accumulation_steps == 0:
    #     # if self.state.global_step % self.args.logging_steps == 0:

    #     if dist.is_initialized():
    #         cumulative_loss_tensor = torch.tensor(self.cumulative_dict["cumulative_loss"]).to('cuda')
    #         dist.all_reduce(cumulative_loss_tensor, op=dist.ReduceOp.SUM)
    #         self.cumulative_dict["cumulative_loss"] = cumulative_loss_tensor.item() / dist.get_world_size()

    #         if dist.get_rank() == 0:
    #             ......


    #     else:
    #         for channel in channels:
    #             loss_name = [k for k, v in self.data_args.channel_loss.items() if v == channel.item()][0]
    #             self.writer.add_scalar(f"train/channel_loss_{loss_name}", self.cumulative_dict[channel.item()], self.state.global_step)
    #             print(f"loss_name: {loss_name} Step: {self.state.global_step + 1}, Loss: {self.cumulative_dict[channel.item()]}, accumulated_steps: {self.cumulative_dict['accumulated_steps']}")

    #         self.writer.add_scalar("train/train_loss", self.cumulative_dict["cumulative_loss"], self.state.global_step + 1)
    #         print("-----", self.cumulative_dict["cumulative_loss"], self.state.global_step + 1, self.cumulative_dict["accumulated_steps"], self.args.gradient_accumulation_steps)

    #     # 重置累积值
    #     self.cumulative_dict["cumulative_loss"] = 0.0
    #     self.cumulative_dict["accumulated_steps"] = 0
    #     for channel in channels:
    #         self.cumulative_dict[channel.item()] = 0.0
