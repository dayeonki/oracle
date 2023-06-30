import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from time import time
import numpy as np
from tqdm import trange
import math

device = 2
batch_size = 2
model_path = "path/to/enko_gpt/checkpoints/enko_gpt"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path).eval().to("cuda:2")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
# Number of layer to include
include_layers = [0]

lang_pairs = ["enko"]
lang_num_dict = {"en": 0, "ko": 1}
valid_path = "path/to/enko_gpt/valid_enko.pt"


# Warmup
input_strings = [
    "Translate English to Korean",
    "Google Translate is machine translation service",
    "구글 번역은 기계 번역 서비스이다",
    "Naver is Korean portal service company",
    "네이버는 한국의 포털서비스 회사이다"
]
embeddings = tokenizer(input_strings, return_tensors="pt", padding=True, truncation=True).to(model.device)
model(**embeddings)


def get_mask_and_last_hidden_states(batch_strings, tokenizer, model):
    embeddings = tokenizer(batch_strings, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    model_out = model.forward(**embeddings, output_hidden_states=True, output_attentions=True)
    attention = model_out["attentions"]
    layerwise_attention = layerwise_pooling(attention, include_layers)
    attentions = layerwise_attention.tolist()
    # last layer hidden states
    last_hidden_states = model_out["hidden_states"][-1].detach().cpu()
    
    print(last_hidden_states.shape, embeddings["attention_mask"].shape)
    return last_hidden_states, embeddings["attention_mask"]


def mean_pooling_sentence_embeddings(last_hidden_states, mask_token):
    mask_token = mask_token.cpu()
    input_mask_expanded = mask_token.unsqueeze(-1).expand(last_hidden_states.size()).float()
    sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_pooled_embeddings = sum_embeddings / sum_mask
    return mean_pooled_embeddings


def layerwise_pooling(attention, layers=None, heads=None):
    if layers:
        attention = [attention[layer_index] for layer_index in layers]
    squeezed = []
    for layer_attention in attention:
        layer_attention = layer_attention.squeeze(0)
        if heads:
            layer_attention = layer_attention[heads]
        squeezed.append(layer_attention)
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)


def embedding_valid_data():
    for lang_pair in lang_pairs:
        src_lang = lang_pair[:2]
        tgt_lang = lang_pair[2:]

        with open(f"data/aihub/{lang_pair}.{src_lang}.valid", "r") as f:
            src_sentences = f.read().rstrip()
            src_sentences = src_sentences.split("\n")

        with open(f"data/aihub/{lang_pair}.{tgt_lang}.valid", "r") as f:
            tgt_sentences = f.read().rstrip()
            tgt_sentences = tgt_sentences.split("\n")
        assert len(src_sentences) == len(tgt_sentences)

        src_sentences = src_sentences[:10]
        tgt_sentences = tgt_sentences[:10]

        # Make embedding
        all_src_embeddings = []
        all_tgt_embeddings = []
        length_sorted_idx = np.argsort([-len(sen) for sen in src_sentences])
        src_sentences_sorted = [src_sentences[idx] for idx in length_sorted_idx]
        tgt_sentences_sorted = [tgt_sentences[idx] for idx in length_sorted_idx]
        
        elapsed_time = time()
        for b in trange(math.ceil(len(src_sentences_sorted) / batch_size)):
            src_batch_strings = src_sentences_sorted[b * batch_size: (b + 1) * batch_size]
            tgt_batch_strings = tgt_sentences_sorted[b * batch_size: (b + 1) * batch_size]

            with torch.no_grad():
                src_hidden_states, src_attention_mask = get_mask_and_last_hidden_states(src_batch_strings, tokenizer, model)
                tgt_hidden_states, tgt_attention_mask = get_mask_and_last_hidden_states(tgt_batch_strings, tokenizer, model)

                src_embeddings = mean_pooling_sentence_embeddings(src_hidden_states, src_attention_mask)
                tgt_embeddings = mean_pooling_sentence_embeddings(tgt_hidden_states, tgt_attention_mask)
            all_src_embeddings.extend(src_embeddings.to("cpu"))
            all_tgt_embeddings.extend(tgt_embeddings.to("cpu"))

        all_src_embeddings = [all_src_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        all_tgt_embeddings = [all_tgt_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        all_src_embeddings = torch.stack(all_src_embeddings)
        all_tgt_embeddings = torch.stack(all_tgt_embeddings)
        
        elapsed_time = time() - elapsed_time
        print("Elapsed time: ", elapsed_time)

        # Save to torch files
        torch.save(all_src_embeddings, f"path/to/enko_gpt/embedded/valid.{lang_pair}.{src_lang}.emb.pt")
        torch.save(all_tgt_embeddings, f"path/to/enko_gpt/embedded/valid.{lang_pair}.{tgt_lang}.emb.pt")


def main():
    embedding_valid_data()


if __name__ == "__main__":
    main()