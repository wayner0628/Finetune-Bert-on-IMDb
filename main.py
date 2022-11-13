from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
import torch.nn.functional as F

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
res = classifier("We are very happy to introduce pipeline to the transformers repository.")

print(res)

tokens = tokenizer.tokenize("We are very happy to introduce pipeline to the transformers repository.")
token_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = tokenizer("We are very happy to introduce pipeline to the transformers repository.")

print(f"   Tokens:{tokens}")
print(f"Token IDs:{token_ids}")
print(f"Input IDs:{input_ids}")

X_train = [
    "We are very happy to introduce pipeline to the transformers repository.",
    "We are very happy to introduce pipeline to the transformers repository.",
]

batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt")

print(batch)

with torch.no_grad():
    outputs = model(**batch, labels=torch.tensor([1, 0]))
    print(outputs)
    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)
    labels = torch.argmax(predictions, dim=1)
    print(labels)
    labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
    print(labels)
