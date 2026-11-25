import math
import os
import random

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers


# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------

REPO = "IWSLT/iwslt2017"
CONFIG = "iwslt2017-en-fr"

SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>"]

PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3

# Anchor paths to this file's location so the code works from any directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizers")


# ----------------------------------------------------------------
# Unit 1: Dataset loader
# ----------------------------------------------------------------

def load_split(split):
    """
    Load a dataset split ('train', 'validation' or 'test')
    and return (english_sentences, french_sentences).

    The IWSLT 2017 repo uses a loading script, which recent versions of
    `datasets` no longer run, so we download the Hub's auto-converted
    parquet files directly instead.
    """

    path = hf_hub_download(
        repo_id=REPO,
        repo_type="dataset",
        revision="refs/convert/parquet",
        filename=f"{CONFIG}/{split}/0000.parquet",
    )

    df = pd.read_parquet(path)

    en = [item["en"] for item in df["translation"]]
    fr = [item["fr"] for item in df["translation"]]

    assert len(en) == len(fr), (
        "English/French dataset sizes do not match."
    )

    return en, fr


# ----------------------------------------------------------------
# Unit 2: Tokenizer
# ----------------------------------------------------------------

def build_tokenizer(sentences, vocab_size=20000):
    """Train a word-level tokenizer on a list of sentences."""

    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))

    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS
    )

    tok.train_from_iterator(sentences, trainer)

    return tok


def verify_tokenizer(tok):
    """Ensure special tokens have the expected IDs."""

    expected = {
        "<pad>": PAD_ID,
        "<unk>": UNK_ID,
        "<sos>": SOS_ID,
        "<eos>": EOS_ID
    }

    for token, expected_id in expected.items():
        actual_id = tok.token_to_id(token)

        assert actual_id == expected_id, (
            f"Token {token} has ID {actual_id}, "
            f"expected {expected_id}"
        )


def get_tokenizers(
    en_train,
    fr_train,
    vocab_size=20000,
    folder=TOKENIZER_DIR
):
    """
    Load saved tokenizers if they exist.
    Otherwise, build and save them.

    Note: saved tokenizers are reused as-is. If you change vocab_size,
    delete the tokenizers/ folder so they get rebuilt.
    """

    en_path = os.path.join(folder, "en_tokenizer.json")
    fr_path = os.path.join(folder, "fr_tokenizer.json")

    if os.path.exists(en_path) and os.path.exists(fr_path):

        en_tok = Tokenizer.from_file(en_path)
        fr_tok = Tokenizer.from_file(fr_path)

    else:

        os.makedirs(folder, exist_ok=True)

        en_tok = build_tokenizer(en_train, vocab_size)
        fr_tok = build_tokenizer(fr_train, vocab_size)

        en_tok.save(en_path)
        fr_tok.save(fr_path)

    verify_tokenizer(en_tok)
    verify_tokenizer(fr_tok)

    return en_tok, fr_tok


# ----------------------------------------------------------------
# Unit 3: Batch data generator
# ----------------------------------------------------------------

def make_pairs(
    en_sents,
    fr_sents,
    en_tok,
    fr_tok,
    max_len=50
):
    """
    Convert sentences into (English IDs, French IDs).

    Discard pairs where either sentence is empty
    or exceeds max_len tokens.
    """

    assert len(en_sents) == len(fr_sents), (
        "English and French sentence counts do not match."
    )

    en_ids = [e.ids for e in en_tok.encode_batch(en_sents)]
    fr_ids = [f.ids for f in fr_tok.encode_batch(fr_sents)]

    pairs = [
        (e, f)
        for e, f in zip(en_ids, fr_ids)
        if 0 < len(e) <= max_len and 0 < len(f) <= max_len
    ]

    return pairs


class BatchGenerator:
    """
    Generates batches of:

    encoder_input  : English token IDs
    decoder_input  : <SOS> + French token IDs
    decoder_target : French token IDs + <EOS>

    Sequences are padded with PAD_ID.
    """

    def __init__(
        self,
        pairs,
        batch_size=128,
        shuffle=True,
        chunk_batches=50
    ):

        self.pairs = pairs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.chunk_size = batch_size * chunk_batches

    def __len__(self):
        return math.ceil(len(self.pairs) / self.batch_size)

    def __iter__(self):

        pairs = list(self.pairs)

        if self.shuffle:
            random.shuffle(pairs)

        batches = []

        for i in range(0, len(pairs), self.chunk_size):

            chunk = sorted(
                pairs[i:i + self.chunk_size],
                key=lambda p: len(p[0])
            )

            for j in range(0, len(chunk), self.batch_size):
                batches.append(chunk[j:j + self.batch_size])

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield self._collate(batch)

    @staticmethod
    def _collate(batch):

        n = len(batch)

        src_len = max(len(e) for e, _ in batch)
        tgt_len = max(len(f) for _, f in batch) + 1

        src = np.full(
            (n, src_len),
            PAD_ID,
            dtype=np.int32
        )

        dec_in = np.full(
            (n, tgt_len),
            PAD_ID,
            dtype=np.int32
        )

        dec_out = np.full(
            (n, tgt_len),
            PAD_ID,
            dtype=np.int32
        )

        for i, (e, f) in enumerate(batch):

            src[i, :len(e)] = e

            dec_in[i, 0] = SOS_ID
            dec_in[i, 1:len(f) + 1] = f

            dec_out[i, :len(f)] = f
            dec_out[i, len(f)] = EOS_ID

        return src, dec_in, dec_out


# ----------------------------------------------------------------
# Dataset preparation
# ----------------------------------------------------------------

def prepare_data(
    batch_size=128,
    vocab_size=20000,
    max_len=50
):
    """
    Load training and validation data,
    build tokenizers, encode sentences,
    and create batch generators.
    """

    # Load datasets
    en_train, fr_train = load_split("train")
    en_val, fr_val = load_split("validation")

    # Build/load training tokenizers
    en_tok, fr_tok = get_tokenizers(
        en_train,
        fr_train,
        vocab_size=vocab_size
    )

    # Convert sentences into token IDs
    train_pairs = make_pairs(
        en_train,
        fr_train,
        en_tok,
        fr_tok,
        max_len=max_len
    )

    val_pairs = make_pairs(
        en_val,
        fr_val,
        en_tok,
        fr_tok,
        max_len=max_len
    )

    print(
        f"Train pairs kept: {len(train_pairs)} of {len(en_train)} "
        f"({100 * len(train_pairs) / len(en_train):.1f}%)"
    )
    print(
        f"Validation pairs kept: {len(val_pairs)} of {len(en_val)} "
        f"({100 * len(val_pairs) / len(en_val):.1f}%)"
    )

    # Create batch generators
    train_gen = BatchGenerator(
        train_pairs,
        batch_size=batch_size,
        shuffle=True
    )

    val_gen = BatchGenerator(
        val_pairs,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        train_gen,
        val_gen,
        en_tok,
        fr_tok
    )


# ----------------------------------------------------------------
# Quick self-test
# ----------------------------------------------------------------

if __name__ == "__main__":

    train_gen, val_gen, en_tok, fr_tok = prepare_data(
        batch_size=64,
        vocab_size=20000,
        max_len=50
    )

    print("English vocab:", en_tok.get_vocab_size())
    print("French vocab:", fr_tok.get_vocab_size())

    print("Training batches:", len(train_gen))
    print("Validation batches:", len(val_gen))

    src, dec_in, dec_out = next(iter(train_gen))

    print("\nBatch shapes:")
    print("Encoder input :", src.shape)
    print("Decoder input :", dec_in.shape)
    print("Decoder target:", dec_out.shape)

    print("\nExample:")
    print("English:", en_tok.decode(src[0].tolist()))

    print("Decoder input:", fr_tok.decode(
        dec_in[0].tolist(),
        skip_special_tokens=False
    ))

    print("Decoder target:", fr_tok.decode(
        dec_out[0].tolist(),
        skip_special_tokens=False
    ))