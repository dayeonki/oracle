import torch
import numpy as np
from tqdm import trange
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils import PaddingStrategy
from typing import Any, Dict, List, Optional, Union
from torch.nn.utils.rnn import pad_sequence


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, src_emb, tgt_emb, src_lang, tgt_lang):
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.src_emb)

    def __getitem__(self, idx):
        return {
            "src_emb": self.src_emb[idx],
            "tgt_emb": self.tgt_emb[idx],
            "src_lang": self.src_lang[idx],
            "tgt_lang": self.tgt_lang[idx],
        }


@dataclass
class LabeledBitextDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]):
        sources = []
        targets = []
        source_langs = []
        target_langs = []
        for feature in features:
            source_input = feature["src"]["input_ids"]
            target_input = feature["tgt"]["input_ids"]
            sources.append(torch.tensor(source_input).squeeze())
            targets.append(torch.tensor(target_input).squeeze())

            source_langs.append(feature["src_lang"])
            target_langs.append(feature["tgt_lang"])

        source_batch = pad_sequence(sources, batch_first=True, padding_value=0)
        target_batch = pad_sequence(targets, batch_first=True, padding_value=0)
        return {
            "src_ids": source_batch,
            "tgt_ids": target_batch,
            "src_lang_label": torch.tensor(source_langs),
            "tgt_lang_label": torch.tensor(target_langs),
        }


def load_dataset(train_path, valid_path):
    data_train = torch.load(train_path)
    dataset_train = TextDataset(
        data_train["src_emb"], data_train["tgt_emb"], data_train["src_lang"], data_train["tgt_lang"]
    )
    
    data_valid = torch.load(valid_path)
    dataset_valid = TextDataset(
        data_valid["src_emb"], data_valid["tgt_emb"], data_valid["src_lang"], data_valid["tgt_lang"]
    )
    return dataset_train, dataset_valid


def loadlines(path: str):
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    return lines


def cosine_distance_torch(x1, x2=None, eps=1e-6):
    x2 = x1 if x2 is None else x2
    w1 = x2.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x2 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def semantic_embedding(tokenizer, model, pooler, sentences, batch_size, device):
    semantic_embeddings = []
    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
    
    for i in trange(-(-len(sentences) // batch_size)):
        sentence_batch = sentences_sorted[i * batch_size : (i + 1) * batch_size]
        inputs = tokenizer(sentence_batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs.to(device)).pooler_output
        embeddings = pooler(outputs)[0]
        semantic_embeddings.extend(embeddings)

    semantic_embeddings = [semantic_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
    semantic_embeddings = torch.stack(semantic_embeddings)
    return semantic_embeddings


def language_embedding(tokenizer, model, pooler, sentences, batch_size, device):
    language_embeddings = []
    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
    
    for i in trange(-(-len(sentences) // batch_size)):
        sentence_batch = sentences_sorted[i * batch_size : (i + 1) * batch_size]
        inputs = tokenizer(sentence_batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs.to(device)).pooler_output
        embeddings = pooler(outputs)[1]
        language_embeddings.extend(embeddings)

    language_embeddings = [language_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
    language_embeddings = torch.stack(language_embeddings)
    return language_embeddings


def semantic_embedding_laser(laser, pooler, sentences, batch_size, lang, device):
    semantic_embeddings = []
    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
    
    for i in trange(-(-len(sentences) // batch_size)):
        sentence_batch = sentences_sorted[i * batch_size : (i + 1) * batch_size]

        languages = []
        for _ in range(len(sentence_batch)):
            languages.append(lang)
        
        outputs = laser.embed_sentences(sentence_batch, lang=languages)
        embeddings = pooler(torch.tensor(outputs).to(device))[0]
        semantic_embeddings.extend(embeddings)

    semantic_embeddings = [semantic_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
    semantic_embeddings = torch.stack(semantic_embeddings)
    return semantic_embeddings


def language_embedding_laser(laser, pooler, sentences, batch_size, lang, device):
    language_embeddings = []
    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
    
    for i in trange(-(-len(sentences) // batch_size)):
        sentence_batch = sentences_sorted[i * batch_size : (i + 1) * batch_size]
        languages = []
        for _ in range(len(sentence_batch)):
            languages.append(lang)
        
        outputs = laser.embed_sentences(sentence_batch, lang=languages)
        embeddings = pooler(torch.tensor(outputs).to(device))[1]
        language_embeddings.extend(embeddings)

    language_embeddings = [language_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
    language_embeddings = torch.stack(language_embeddings)
    return language_embeddings