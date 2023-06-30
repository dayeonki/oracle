import argparse
import numpy as np
from tqdm import tqdm
from laser.LASER_light import SentenceEncoder, get_tokenizer


def l2_norm(embeddings):
	embeddings /= np.linalg.norm(embeddings, 
                              ord=2, 
                              axis=1, 
                              keepdims=True)
	return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="input text file")
    parser.add_argument("-o", "--output", type=str, required=True, help="output numpy file")
    parser.add_argument("-d", "--device", type=int, required=True, help="gpu device")
    args = parser.parse_args()

    with open(args.input) as f:
        sentences = [sentence.strip() for sentence in f]
    
    for i in range(0, len(sentences), 1000000):
        print("Start Index: ", str(i))
        print("End Index: ", str(i + 1000000))
        truncated_sentences = sentences[i : i+1000000]
        encoder = SentenceEncoder(device=args.device)
        tokenizer = get_tokenizer()
        
        pretokenized_sentences = [" ".join(tokenizer.encode_as_pieces(each)) for each in tqdm(truncated_sentences, total=len(truncated_sentences), desc="Tokenize")]
        embeddings = encoder.encode_sentences(pretokenized_sentences)
        embeddings = l2_norm(args, embeddings)
        np.save(f"{args.output}{str(i+1000000)}.npy", embeddings)


if __name__ == "__main__":
    main()
