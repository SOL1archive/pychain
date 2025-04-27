import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LRScheduler, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader

import transformers
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, SchedulerType
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTConfig, SFTTrainer

from train_config import TrainConfig
from utils import PromptFormatter

def main():
    config = TrainConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument('--sector', dest='sector', action='store', type=str, default=None)
    args = parser.parse_args()

    if args.sector is not None:
        sector = args.sector
        dataset_path = str(Path(config.dataset_path).parent / sector)
    else:
        sector = Path(config.dataset_path).name
        dataset_path = config.dataset_path

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint, padding_side='left')
    tokenizer.padding_side = 'left'

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    #model = AutoModelForCausalLM.from_pretrained(config.checkpoint, attn_implementation='flash_attention_2')
    model = AutoModelForCausalLM.from_pretrained(
        config.checkpoint, 
        attn_implementation='flash_attention_2',
        quantization_config=bnb_config, 
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_hidden_dim,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    #ds_dict = load_dataset(dataset_path)['train'].shuffle(seed=42).train_test_split(test_size=0.1, seed=42)
    #train_ds = ds_dict['train']
    #val_ds = ds_dict['test']
    train_ds = load_dataset(dataset_path)['train'].shuffle(seed=42)
    prompt_formatter = PromptFormatter(
        instruction_col_name='problem',
        response_col_name='solution',
        tokenizer=tokenizer,
        system_instruction='You are an thorough math problem solver. Inspect the problem, fine a solution and answer with detailed explanations.'
    )

    training_args = SFTConfig(
        max_seq_length=2048,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_checkpointing=config.gradient_checkpoint,
        num_train_epochs=config.num_epochs,

        #do_eval=True,
        #per_device_eval_batch_size=config.batch_size,
        #eval_strategy='steps',
        #eval_steps=1000,

        logging_first_step=0,
        logging_steps=1,
        report_to='wandb',

        output_dir='./out',
        save_strategy='best',
        metric_for_best_model='eval_loss',
    )

    trainer = SFTTrainer(
        model,
        training_args,
        train_dataset=train_ds,
        #eval_dataset=val_ds,
        formatting_func=prompt_formatter,
    )

    trainer.train()
    trainer.save_model(f'models/{sector}')

if __name__ == '__main__':
    main()
