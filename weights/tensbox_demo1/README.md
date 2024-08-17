---
base_model: /home/repository/Qwen2-0.5B-Instruct
library_name: peft
license: other
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: tensbox_demo1
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# tensbox_demo1

This model is a fine-tuned version of [/home/repository/Qwen2-0.5B-Instruct](https://huggingface.co//home/repository/Qwen2-0.5B-Instruct) on the channel_test_semantic_20240808, the channel_test_evaluation_good_20240808 and the channel_test_evaluation_general_20240808 datasets.
It achieves the following results on the evaluation set:
- Loss: 1.8768

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 6
- total_train_batch_size: 12
- total_eval_batch_size: 2
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 10.0
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.12.0
- Transformers 4.43.4
- Pytorch 2.3.0
- Datasets 2.20.0
- Tokenizers 0.19.1