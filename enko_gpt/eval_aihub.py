import argparse
import numpy as np
import torch
import json
from tqdm import trange
import math
from model import DREAM_MLP, MEAT_MLP
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sklearn.metrics import pairwise_distances_argmin


parser = argparse.ArgumentParser()
parser.add_argument("--pooler_path", required=True)
parser.add_argument("--batch_size", default=8)
parser.add_argument("--layer_num", type=int, default=0)
args = parser.parse_args()

device = 2
model_path = "path/to/enko1.3B-bidirectional"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path).eval().to("cuda:2")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id


def mean_pooling_sentence_embeddings(last_hidden_states, mask_token):
    mask_token = mask_token.cpu()
    input_mask_expanded = mask_token.unsqueeze(-1).expand(last_hidden_states.size()).float()
    sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_pooled_embeddings = sum_embeddings / sum_mask
    return mean_pooled_embeddings


def get_mask_and_last_hidden_states(batch_strings, tokenizer, model):
    embeddings = tokenizer(batch_strings, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    model_out = model.forward(**embeddings, output_hidden_states=True)
    # Layerwise pooling results
    last_hidden_states = model_out["hidden_states"][args.layer_num].detach().cpu()
    return last_hidden_states, embeddings["attention_mask"]


def retrieval_eval(emb1, emb2):
    sim = pairwise_distances_argmin(emb1, emb2, metric="cosine")
    acc = []
    for i, p in enumerate(sim):
        acc.append(i == p)
    return np.mean(acc) * 100, sim


def retrieve(pooler, batch_size):
    all_acc_jsonl = []
    all_acc = []
    data_dir = 'path/to/gpt3_embedding/data/aihub'
    language_pair = "enko"
    src_lang = "en"
    tgt_lang = "ko"

    print(src_lang + "->" + tgt_lang)
    source_file = f'{data_dir}.{language_pair}.{src_lang}'
    src_sentences = [line.strip() for line in open(source_file)]
    target_file = f'{data_dir}.{language_pair}.{tgt_lang}'
    tgt_sentences = [line.strip() for line in open(target_file)]

    for i in range(len(src_sentences)):
        src_sentences[i] = src_sentences[i] + " [EN2KO]"
    for i in range(len(tgt_sentences)):
        tgt_sentences[i] = tgt_sentences[i] + " [KO2EN]"
    
    # Make embedding
    all_src_embeddings = []
    all_tgt_embeddings = []
    length_sorted_idx = np.argsort([-len(sen) for sen in src_sentences])
    src_sentences_sorted = [src_sentences[idx] for idx in length_sorted_idx]
    tgt_sentences_sorted = [tgt_sentences[idx] for idx in length_sorted_idx]
    
    for b in trange(math.ceil(len(src_sentences_sorted) / batch_size)):
        src_batch_strings = src_sentences_sorted[b * batch_size: (b + 1) * batch_size]
        tgt_batch_strings = tgt_sentences_sorted[b * batch_size: (b + 1) * batch_size]

        with torch.no_grad():
            src_hidden_states, src_attention_mask = get_mask_and_last_hidden_states(src_batch_strings, tokenizer, model)
            tgt_hidden_states, tgt_attention_mask = get_mask_and_last_hidden_states(tgt_batch_strings, tokenizer, model)

            src_embeddings = mean_pooling_sentence_embeddings(src_hidden_states, src_attention_mask)
            tgt_embeddings = mean_pooling_sentence_embeddings(tgt_hidden_states, tgt_attention_mask)
        
        # Get semantic embedding with trained pooler
        src_embeddings = pooler(src_embeddings)[0]
        tgt_embeddings = pooler(tgt_embeddings)[0]
        all_src_embeddings.extend(src_embeddings.to("cpu"))
        all_tgt_embeddings.extend(tgt_embeddings.to("cpu"))

    all_src_embeddings = [all_src_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
    all_tgt_embeddings = [all_tgt_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
    all_src_embeddings = torch.stack(all_src_embeddings)
    all_tgt_embeddings = torch.stack(all_tgt_embeddings)

    all_src_embeddings = all_src_embeddings.cpu()
    all_tgt_embeddings = all_tgt_embeddings.cpu()

    all_src_embeddings = all_src_embeddings.detach().numpy()
    all_tgt_embeddings = all_tgt_embeddings.detach().numpy()

    acc, pred = retrieval_eval(all_src_embeddings, all_tgt_embeddings)
    
    assert len(src_sentences) == len(tgt_sentences)
    for i in range(len(src_sentences)):
        acc_jsonl = {
            "source_lang": "eng",
            "accuracy": acc,
            "source_text": src_sentences[i],
            "target_text": tgt_sentences[i],
            "prediction_idx": int(pred[i]),
            "prediction": tgt_sentences[pred[i]]
            }
        all_acc_jsonl.append(acc_jsonl)
    all_acc.append(acc)  
    print(f'{src_lang} -> {tgt_lang} average acc:', acc) 

    # Save results
    with open(f'layerwise_results/enko_tatoeba_l{str(args.layer_num)}.jsonl', 'w', encoding='utf-8') as f:
        for acc_jsonl in all_acc_jsonl:
            json.dump(acc_jsonl, f, ensure_ascii=False)
            f.write("\n")


def main():
    pooler = DREAM_MLP(embedding_size=2048, n_languages=2)
    # pooler = MEAT_MLP(embedding_size=2048)
    pooler.load_state_dict(torch.load(args.pooler_path))
    retrieve(pooler, args.batch_size)


if __name__ == "__main__":
    main()
