# Text Generation with checkpoints from LitGPT

# import torch

# from litgpt.model import TransformerDecoder


def generate_tokenisers(dataset_path="data/tinyshakespeare.txt"):
    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()
        chars = sorted(list(set(text)))

        # tokenisers
        stoi = {ch: i for i, ch in enumerate(chars)}  # string to int

        def encode(s):
            return [stoi[c] for c in s]  # encoder: maps strings to list of ints

        itos = {i: ch for i, ch in enumerate(chars)}  # int to string

        def decode(list_ints):
            return "".join(
                [itos[i] for i in list_ints]
            )  # decoder: maps list of ints to string

    return encode, decode


def main():

    # hparams = 0  # where to load these from?
    # decoder = TransformerDecoder(hparams)

    return 0


if __name__ == "__main__":
    main()
