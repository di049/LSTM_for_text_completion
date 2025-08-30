import evaluate
from tqdm import tqdm

def evaluate_pipeline(generator):
    with open('data/tweets_processed.txt', 'r', encoding='utf-8') as f:
        data = f.read()

    data_split = data.split('\n')[:10000]

    results = []
    for sample in tqdm(data_split):
        sample=sample[:round(len(sample)*3/4)] # берем 3/4 последовательности
        new_sample_len = max(1, round(len(sample)*1/4)) 
        results.append(generator(sample, max_new_tokens=new_sample_len, do_sample=True, top_k=50)[0]["generated_text"]) # предсказываем оставшуюся 1/4

    rouge = evaluate.load("rouge")
    rouge_score = rouge.compute(predictions=results, references=data_split)

    return results, rouge_score