import string
import re
from collections import Counter
from typing import Iterable, List, Mapping, Optional


class CharTokenizer:
    """A small fixed-vocabulary tokenizer for English image captions."""

    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)
    DEFAULT_CHARACTERS = string.ascii_lowercase + string.digits + " .,!?'-"
    tokenizer_type = "char"

    def __init__(self, characters: Optional[Iterable[str]] = None):
        characters = self.DEFAULT_CHARACTERS if characters is None else characters
        # dict.fromkeys removes duplicates while preserving the configured order.
        vocabulary = list(self.SPECIAL_TOKENS) + list(dict.fromkeys(characters))
        self.stoi = {token: index for index, token in enumerate(vocabulary)}
        self.itos = {index: token for token, index in self.stoi.items()}

    @classmethod
    def from_vocab(cls, stoi: Mapping[str, int]):
        instance = cls.__new__(cls)
        instance._restore_vocab(stoi)
        return instance

    def _restore_vocab(self, stoi: Mapping[str, int]):
        ids = sorted(stoi.values())
        if ids != list(range(len(ids))):
            raise ValueError("token ids must be contiguous and start at zero")
        if any(token not in stoi for token in self.SPECIAL_TOKENS):
            raise ValueError("vocabulary is missing required special tokens")
        self.stoi = {str(token): int(index) for token, index in stoi.items()}
        self.itos = {index: token for token, index in self.stoi.items()}

    @classmethod
    def from_text(cls, text: str):
        """Build a tokenizer containing the default alphabet and text characters."""
        return cls(cls.DEFAULT_CHARACTERS + "".join(sorted(set(text.lower()))))

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def pad_token_id(self) -> int:
        return self.stoi[self.PAD_TOKEN]

    @property
    def bos_token_id(self) -> int:
        return self.stoi[self.BOS_TOKEN]

    @property
    def eos_token_id(self) -> int:
        return self.stoi[self.EOS_TOKEN]

    @property
    def unk_token_id(self) -> int:
        return self.stoi[self.UNK_TOKEN]

    def encode(self, text: str) -> List[int]:
        """Encode caption text without automatically adding BOS or EOS."""
        return [
            self.stoi.get(character, self.unk_token_id)
            for character in text.lower()
        ]

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        pieces = []
        for token_id in ids:
            token = self.itos.get(int(token_id), self.UNK_TOKEN)
            if skip_special_tokens and token == self.EOS_TOKEN:
                break
            if skip_special_tokens and token in {
                self.PAD_TOKEN,
                self.BOS_TOKEN,
            }:
                continue
            pieces.append(token)
        return "".join(pieces)


class WordTokenizer(CharTokenizer):
    """A vocabulary-backed tokenizer for English words and punctuation."""

    TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*|[^\w\s]", re.IGNORECASE)
    tokenizer_type = "word"

    def __init__(self, vocabulary: Optional[Iterable[str]] = None):
        vocabulary = () if vocabulary is None else vocabulary
        tokens = list(self.SPECIAL_TOKENS) + [
            token
            for token in dict.fromkeys(vocabulary)
            if token not in self.SPECIAL_TOKENS
        ]
        self.stoi = {token: index for index, token in enumerate(tokens)}
        self.itos = {index: token for token, index in self.stoi.items()}

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        min_frequency: int = 1,
        max_vocab_size: Optional[int] = None,
    ):
        if min_frequency < 1:
            raise ValueError("min_frequency must be positive")
        if max_vocab_size is not None and max_vocab_size < len(cls.SPECIAL_TOKENS):
            raise ValueError("max_vocab_size must include all special tokens")

        counts = Counter(
            token for text in texts for token in cls.tokenize_text(text)
        )
        vocabulary = [
            token
            for token, count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
            if count >= min_frequency
        ]
        if max_vocab_size is not None:
            vocabulary = vocabulary[: max_vocab_size - len(cls.SPECIAL_TOKENS)]
        return cls(vocabulary)

    @classmethod
    def tokenize_text(cls, text: str) -> List[str]:
        return cls.TOKEN_PATTERN.findall(text.lower())

    def encode(self, text: str) -> List[int]:
        return [
            self.stoi.get(token, self.unk_token_id)
            for token in self.tokenize_text(text)
        ]

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for token_id in ids:
            token = self.itos.get(int(token_id), self.UNK_TOKEN)
            if skip_special_tokens and token == self.EOS_TOKEN:
                break
            if skip_special_tokens and token in {self.PAD_TOKEN, self.BOS_TOKEN}:
                continue
            tokens.append(token)

        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?;:%)])", r"\1", text)
        text = re.sub(r"([(])\s+", r"\1", text)
        text = re.sub(r"\s*-\s*", "-", text)
        return text


def build_tokenizer(
    tokenizer_type: str = "char",
    texts: Optional[Iterable[str]] = None,
    stoi: Optional[Mapping[str, int]] = None,
    min_word_frequency: int = 1,
    max_vocab_size: Optional[int] = None,
):
    """Build a tokenizer for training or restore one from a checkpoint vocab."""
    tokenizer_classes = {"char": CharTokenizer, "word": WordTokenizer}
    try:
        tokenizer_class = tokenizer_classes[tokenizer_type]
    except KeyError as error:
        raise ValueError(f"unsupported tokenizer type: {tokenizer_type}") from error

    if stoi is not None:
        return tokenizer_class.from_vocab(stoi)
    if tokenizer_type == "word":
        if texts is None:
            raise ValueError("texts are required to build a word tokenizer")
        return WordTokenizer.from_texts(
            texts,
            min_frequency=min_word_frequency,
            max_vocab_size=max_vocab_size,
        )
    return CharTokenizer()


def tokenizer_from_checkpoint(checkpoint):
    """Restore tokenizer metadata, including legacy character checkpoints."""
    config = checkpoint.get("config", {})
    tokenizer_type = checkpoint.get(
        "tokenizer_type", config.get("tokenizer", "char")
    )
    return build_tokenizer(
        tokenizer_type=tokenizer_type,
        stoi=checkpoint.get("tokenizer_stoi"),
    )


def tokenizer(text):
    """Backward-compatible helper used by the earlier TinyGPT exercises."""
    instance = CharTokenizer.from_text(text)
    return (
        instance.encode,
        instance.decode,
        instance.vocab_size,
        instance.stoi,
        instance.itos,
    )
