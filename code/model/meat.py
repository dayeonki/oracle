import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, embedding_size: int, n_languages: int) -> None:
        super(MLP, self).__init__()
        self.meaning_embedding_layer = nn.Linear(embedding_size, embedding_size)
        self.language_embedding_layer = nn.Linear(embedding_size, embedding_size)

    def forward(self, input):
        meaning_embedding = self.meaning_embedding_layer(input)
        language_embedding = self.language_embedding_layer(input)
        return meaning_embedding, language_embedding


class Discriminator(nn.Module):
    def __init__(self, embedding_size: int, n_languages: int) -> None:
        super().__init__()
        # Change according to the number of languages (self.embedding_size, N)
        self.language_identification_layer = nn.Linear(embedding_size, n_languages)

    def forward(self, input):
        language_identification = self.language_identification_layer(input)
        return language_identification