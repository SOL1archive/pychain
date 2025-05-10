from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    checkpoint: str = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    lora_hidden_dim: str = 32
    dataset_path: str = 'nvidia/OpenMathReasoning'

    gradient_checkpoint: bool = True
    max_seq_length: int = 2048
    streaming: bool = True
    num_samples: int = 300_000
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 1
    scheduler_type: str = 'linear'

@dataclass
class RouterTrainConfig:
    checkpoint: str = 'Alibaba-NLP/gte-modernbert-base'
    dataset_root: str = 'datasets/MATH/train'

    seq_max_len = 1024
    gradient_checkpoint: bool = False
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    num_epochs: int = 100
    scheduler_type: str = 'linear'
