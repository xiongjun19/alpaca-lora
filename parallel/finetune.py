"""fine_tune Llama"""


import os
import json
import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron.arguments import core_transformer_config_from_args
from llama_mega_model import LlamaModel

from utils.prompter import Prompter
from datasets import load_dataset


def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    print_rank_0('building GPT model ...')
    args = get_args()
    f_model_path = os.path.join(args.model_cfg_path, 'config.json')
    config = _get_model_conf(f_model_path)
    mega_config = core_transformer_config_from_args(get_args())
    model = LlamaModel(config, mega_config,  pre_process, post_process)
    return model


def _get_model_conf(model_cfg_path):
    with open(model_cfg_path) as _in:
        cfg = json.load(_in)
        return cfg


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eos_token_id,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()
    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()
    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)
    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')

    train_ds, val_ds, test_ds = build_alpaca_ds()
    print_rank_0("> finished creating alpaca_datasets ...")

    return train_ds, val_ds, test_ds


def build_alpaca_ds():
    data_path = "alpaca_data_cleaned.json"
    data = load_dataset("json", data_files=data_path)
    val_set_size = 2000
    cutoff_len = 256
    train_val = data['train'].train_test_split(
        test_size=val_set_size, shuffle=True, seed=42
    )
    train_data = (
        train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    )
    val_data = (
        train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    )
    test_data = (
        train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    )
    return train_data, val_data, test_data


def generate_and_tokenize_prompt(data_point):
    prompter = Prompter('alpaca')
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt


def tokenize(prompt, add_eos_token=True):
    tokenizer = get_tokenizer()
    cutoff_len = 256
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    result["text"] = result["input_ids"].copy()
    return result


if __name__ == "__main__":
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'LlamaTokenizer'})

