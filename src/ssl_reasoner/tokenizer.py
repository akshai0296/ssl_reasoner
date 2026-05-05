from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CharTokenizer:
    """Small deterministic tokenizer for synthetic math strings."""

    chars: str
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"

    def __post_init__(self):
        special = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        vocab = special + list(dict.fromkeys(self.chars))
        object.__setattr__(self, "id_to_token", vocab)
        object.__setattr__(self, "token_to_id", {token: i for i, token in enumerate(vocab)})

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str, max_len: int, add_special: bool = True) -> list[int]:
        ids = []
        if add_special:
            ids.append(self.bos_id)
        ids.extend(self.token_to_id.get(ch, self.unk_id) for ch in text)
        if add_special:
            ids.append(self.eos_id)
        ids = ids[:max_len]
        ids.extend([self.pad_id] * (max_len - len(ids)))
        return ids

    def decode(self, ids: list[int]) -> str:
        out = []
        for idx in ids:
            if idx == self.eos_id:
                break
            if idx in (self.pad_id, self.bos_id):
                continue
            out.append(self.id_to_token[idx] if idx < len(self.id_to_token) else "")
        return "".join(out)


MATH_CHARS = "0123456789+-*()= ?abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,"


def build_math_tokenizer() -> CharTokenizer:
    return CharTokenizer(MATH_CHARS)
