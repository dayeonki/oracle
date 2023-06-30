import argparse
import numpy as np
from transformers import AutoModel, AutoTokenizer


def l2_norm(args, embeddings):
	embeddings /= np.linalg.norm(embeddings, 
                              ord=args.norm, 
                              axis=1, 
                              keepdims=True)
	return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name_or_path", type=str, required=True, help="Transformers' model name or path")
    parser.add_argument("-i", "--input", type=str, required=True, help="input text file")
    parser.add_argument("-o", "--output", type=str, required=True, help="output numpy file")
    parser.add_argument("-d", "--device", type=int, required=True, help="gpu device")
    
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true")
    parser.add_argument("-n", "--norm", type=int, default=2, help="normalization type")
    args = parser.parse_args()
        
    with open(args.input) as f:
        lines = [line.strip() for line in f]
        encoder = Encoder(
            model=AutoModel.from_pretrained(args.model_name_or_path),
            tokenizer=AutoTokenizer.from_pretrained(args.model_name_or_path),
            device=args.device
        )

        embeddings = encoder.encode(lines, batch_size=args.batch_size, verbose=args.verbose)
        embeddings = l2_norm(args, embeddings)
        np.save(f"{args.output}.npy", embeddings)


if __name__ == "__main__":
    main()