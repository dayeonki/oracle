import torch
import argparse
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import trange


def loadlines(path):
    with open(path) as f:
        lines = [line.strip() for line in f] 
    return lines


def mean_pooling_sentence_embeddings(last_hidden_states, mask_token):
    input_mask_expanded = mask_token.unsqueeze(-1).expand(last_hidden_states.size()).float()
    sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_pooled_embeddings = sum_embeddings / sum_mask
    return mean_pooled_embeddings


def get_layerwise_mean_pooling(model, tokenizer, batch_strings):
    embeddings = tokenizer(batch_strings, return_tensors="pt", truncation=True, padding=True, max_length=512).to(model.device)
    model_out = model.forward(**embeddings, output_hidden_states=True, output_attentions=True)
    hidden_states = [each_layer for each_layer in model_out["hidden_states"]]
    attention_mask = embeddings["attention_mask"]
    mean_pooled_states = [mean_pooling_sentence_embeddings(each_layer, attention_mask).cpu() for each_layer in hidden_states]
    return mean_pooled_states


def get_batch_mean_layerwise_mean_pooling(model, tokenizer, strings, batch_size, n_layers=25):
    n_batches = math.ceil(len(strings) / batch_size)
    layerwise_mean_pooled_embeddings = [[] for _ in range(n_layers)]
    for b in trange(n_batches):
        batch_strings = strings[b * batch_size: (b + 1) * batch_size]
        with torch.no_grad():
            mean_pooled_embeddings = get_layerwise_mean_pooling(model, tokenizer, batch_strings)
            for layer_idx, each_embeddings in enumerate(mean_pooled_embeddings):
                layerwise_mean_pooled_embeddings[layer_idx].extend(each_embeddings)
    layerwise_mean_pooled_embeddings = [torch.stack(each_layer_embeddings) for each_layer_embeddings in layerwise_mean_pooled_embeddings]
    return layerwise_mean_pooled_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_input", required=True, type=str)
    parser.add_argument("-t", "--target_input", required=True, type=str)
    parser.add_argument("-sl", "--source_langid", required=True, type=int)
    parser.add_argument("-tl", "--target_langid", required=True, type=int)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-d", "--device", required=True, type=int)
    parser.add_argument("-b", "--batch_size", type=int, default=20)
    parser.add_argument("-m", "--model_path", type=str, default="path/to/enko_gpt")
    args = parser.parse_args()
    output_prefix = args.output
    if output_prefix.endswith(".pt"):
        output_prefix = output_prefix[:-3]

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path).eval().to(f"cuda:{args.device}")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    sources = loadlines(args.source_input)
    targets = loadlines(args.target_input)
    n_data = len(sources)
    assert len(sources) == len(targets)
    assert args.source_langid != args.target_langid

    source_layerwise_mean_pooled_embeddings = get_batch_mean_layerwise_mean_pooling(model, tokenizer, sources, args.batch_size)
    target_layerwise_mean_pooled_embeddings = get_batch_mean_layerwise_mean_pooling(model, tokenizer, targets, args.batch_size)
    source_langids = torch.LongTensor([args.source_langid] * n_data)
    target_langids = torch.LongTensor([args.target_langid] * n_data)

    n_layers = len(source_layerwise_mean_pooled_embeddings)
    for i_layer in range(n_layers):
        decomposer_train_data = {
            "src_emb": source_layerwise_mean_pooled_embeddings[i_layer],
            "tgt_emb": target_layerwise_mean_pooled_embeddings[i_layer],
            "src_lang": source_langids,
            "tgt_lang": target_langids,
        }
        torch.save(decomposer_train_data, f"{output_prefix}.l{i_layer}.pt")


if __name__ == "__main__":
    main()