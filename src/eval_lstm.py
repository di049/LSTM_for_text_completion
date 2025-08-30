import torch
from transformers import AutoTokenizer
import evaluate
import torch.nn as nn

def evaluate_model(model, loader):
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    correct, total = 0, 0
    sum_loss = 0
    rouge1, rouge2 = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            x_batch, y_batch = x_batch, y_batch
            output, _ = model(x_batch)

            output = output.reshape(-1, output.size(-1))
            y_batch = y_batch.reshape(-1)

            loss = criterion(output, y_batch)
            preds = torch.argmax(output, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            sum_loss += loss.item()

            rouge1 += evaluate_rouge(preds, y_batch)['rouge1']
            rouge2 += evaluate_rouge(preds, y_batch)['rouge2']

    return sum_loss / len(loader), correct / total, rouge1 / len(loader), rouge2 / len(loader)


def evaluate_rouge(preds, refs):
    tokenizer = AutoTokenizer.from_pretrained('distilbert/distilgpt2')
    preds = tokenizer.decode(preds)
    refs = tokenizer.decode(refs)
    rouge = evaluate.load("rouge")
    rouge_score = rouge.compute(predictions=[preds], references=[refs])

    return rouge_score