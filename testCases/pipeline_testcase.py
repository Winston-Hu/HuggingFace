"""
pipeline作用就是3个：数据预处理（拆分成多个token），模型处理（调用什么模型），模型后处理（返回结果）
"""


from transformers import pipeline
import time


start_time = time.time()

# 直接创建classificator或者什么别的对象(不推荐，建议直接根据model的sample写)
classificator = pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", task="text-classification")
res = classificator("I love my life")

end_time = time.time()
print(res, round(end_time - start_time, 2))
