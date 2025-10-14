class Tokenizer:
    def __init__(
            self,
            text: str
    ) -> None:
        self.vocab_size: int
        self.encoder: dict[str, int]
        self.decoder: dict[int, str]
        
        vocabulary = sorted(list(set(text)))
        self.vocab_size = len(vocabulary)

        self.encoder = dict()
        self.decoder = dict()
        for num, char in enumerate(vocabulary):
            self.encoder[char] = num
            self.decoder[num] = char

    def encode(
            self,
            chars: str
    ) -> list[int]:
        return [self.encoder[char] for char in chars]

    def decode(
            self,
            tokens: list[int]
    ) -> str:
        return "".join([self.decoder[token] for token in tokens])
