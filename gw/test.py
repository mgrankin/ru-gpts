import sys
sys.path.append("gw/")

from generation_wrapper import RuGPT3XL


gpt = RuGPT3XL.from_pretrained("sberbank-ai/rugpt3xl", seq_len=512)
res = gpt.generate(
    "На словах - ты Лев Тостой ",
    max_length=50,
    no_repeat_ngram_size=3,
    repetition_penalty=2.,
)
print(res)
