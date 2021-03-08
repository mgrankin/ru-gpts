import deepspeed.ops.sparse_attention.sparse_attn_op
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("ru-gpts/")
import os
os.environ["USE_DEEPSPEED"] = "1"
from src.xl_wrapper import RuGPT3XL
gpt = RuGPT3XL.from_pretrained("sberbank-ai/rugpt3xl", seq_len=512)
logits = gpt("Кто был президентом США в 2020? ").logits