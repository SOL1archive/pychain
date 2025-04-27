from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    checkpoint: str = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    lora_hidden_dim: str = 32
    dataset_path: str = 'datasets/MATH/train/algebra'

    gradient_checkpoint: bool = True
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 50
    scheduler_type: str = 'linear'
