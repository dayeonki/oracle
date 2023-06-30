import torch
import math
from tqdm import trange
from transformers import GPT2Tokenizer, GPT2LMHeadModel


device = 2
batch_size = 16
model_path = "path/to/enko_gpt/checkpoints/enko_gpt"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path).eval().to("cuda:2")
model.eval()
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

embedding_file = "path/to/enko_gpt/embedded/train.enko.en.emb.300000.pt"
data_file = "path/to/enko_gpt/data/split/enko.en.train.300000"


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
    last_hidden_states = model_out["hidden_states"][-1].detach().cpu()
    return last_hidden_states, embeddings["attention_mask"]


def save_norm():
    with open(f"{data_file}", "r") as f:
        sentences = f.read().rstrip()
        sentences = sentences.split("\n")
    
    outputs = []
    for b in trange(math.ceil(len(sentences) / batch_size)):
        sentence_batch = sentences[b * batch_size: (b + 1) * batch_size]
        if len(sentence_batch) == 16:
            hidden_states, attention_masks = get_mask_and_last_hidden_states(sentence_batch, tokenizer, model)
            embedding = mean_pooling_sentence_embeddings(hidden_states, attention_masks)
            outputs.append(embedding)
        else:
            pass
    
    all_embeddings = torch.stack(outputs)
    print(all_embeddings.shape)
    print(torch.norm(all_embeddings, p=2, dim=2))
    norm = torch.norm(all_embeddings, p=2, dim=2)

    # Save torch file
    torch.save(all_embeddings, f"enko_embeddings.pt")
    torch.save(norm, f"enko_embeddings_norm.pt")

   
if __name__ == "__main__":
    save_norm()