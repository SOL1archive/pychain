from typing import Optional
import numpy as np
from transformers import AutoTokenizer
import evaluate

class PromptFormatter:
    def __init__(
            self, 
            instruction_col_name, 
            response_col_name, 
            tokenizer: AutoTokenizer,
            system_instruction: Optional[str]=None, 
        ):
        if system_instruction is not None:
            self.system_instruction = system_instruction
        else:
            self.system_instruction = None
        self.instruction_col_name = instruction_col_name
        self.response_col_name = response_col_name
        self.tokenizer = tokenizer

    def __call__(self, example):
        prompt = [
            {'role': 'user', 'content': example[self.instruction_col_name]},
            {'role': 'assistant', 'content': example[self.response_col_name]}
        ]
        if self.system_instruction is not None:
            prompt.append({'role': 'system', 'content': self.system_instruction})
        return self.tokenizer.apply_chat_template(prompt, tokenize=False)

class TokenizeMapWrapper:
    def __init__(self, tokenizer, feature, option=None):
        if option is None:
            option = {
                'max_length': 1024,
                'truncation': True,
                'padding': 'max_length',
            }
        self.option = option
        
        self.feature = feature
        self.tokenizer = tokenizer

    def __call__(self, row):
        return self.tokenizer(row[self.feature], **self.option)

    def __repr__(self):
        return f'{self.__class__.__name__}(tokenizer={self.tokenizer})'

accuracy = evaluate.load('accuracy')

def compute_accuarcy(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    return accuracy.compute(
        predictions=preds,
        references=labels
    )
