import argparse
import transformers
import os
from datasets import Dataset
from typing import List
from transformers import AutoTokenizer
from utils import *


def tokenize_texts(sentences: List[str], tokenizer: transformers.PreTrainedTokenizer):
    tokenized_texts = []
    for text in sentences:
        tokenized_text = tokenizer(text, max_length=512, truncation=True, return_token_type_ids=False, return_special_tokens_mask=False)
        tokenized_texts.append(tokenized_text)
    return tokenized_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name_or_path", type=str, default="sentence-transformers/LaBSE")
    parser.add_argument("-o", "--output_dir", type=str, default="../data/tokenized/jako_test/")
    
    parser.add_argument("-sd", "--src_data_path", default="bitext/jako/jako.ja")
    parser.add_argument("-td", "--tgt_data_path", default="bitext/jako/jako.ko")
    parser.add_argument("-sl", "--src_lang", type=int, default=0)
    parser.add_argument("-tl", "--tgt_lang", type=int, default=1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Load bitext & language datasets
    sources = loadlines(args.src_data_path)
    targets = loadlines(args.tgt_data_path)
    assert len(sources) == len(targets)

    source_languages = []
    target_languages = []
    source_languages.append(args.src_lang for _ in range(len(sources)))
    target_languages.append(args.tgt_lang for _ in range(len(targets)))

    tokenized_sources = tokenize_texts(sources, tokenizer)
    tokenized_targets = tokenize_texts(targets, tokenizer)
    assert len(tokenized_sources) == len(tokenized_targets)
    
    # Save as Dataset format
    """
    Tokenized dataset format
        tokenized_dataset = {
            "src": [{'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                'input_ids': [0, 23713, 4034, 25069, 6, 4, 1880, 4858, 16844, 122, 141160, 3473, 67, 79315, 2464, 2388, 35601, 542, 4034, 6, 69418, 198276, 700, 142246, 18, 165, 18370, 491, 745, 160196, 98306, 6, 5, 2]}]
            "tgt": [{'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                'input_ids': [0, 4263, 4034, 83, 40907, 6, 4, 70, 456, 133266, 214, 1221, 9842, 99, 70, 4034, 619, 17121, 74, 927, 62816, 1295, 70, 3564, 111, 79315, 6, 5, 2]}],
            "src_lang": [1, 1, 1, ...],
            "tgt_lang": [2, 2, 2, ...]
    """
    tokenized_dataset = Dataset.from_dict(
        {
            "src": tokenized_sources,
            "tgt": tokenized_targets,
            "src_lang": source_languages,
            "tgt_lang": target_languages
        }
    )
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    tokenized_dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":
    main()