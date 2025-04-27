from typing import Optional
from transformers import AutoTokenizer

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
