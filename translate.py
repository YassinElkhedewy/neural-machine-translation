import argparse
import os
import random
import re

import numpy as np
import tensorflow as tf
from tokenizers import Tokenizer

from data import EOS_ID, SOS_ID, TOKENIZER_DIR, load_split
from model import build_models
from train import CHECKPOINT_DIR, RESULTS_DIR, load_checkpoint


DEMO_SENTENCES = [
    "Thank you very much for coming today.",
    "I would like to talk about climate change.",
    "What is your name?",
    "The weather is beautiful this morning.",
    "We need to work together to solve this problem.",
    "I love learning new languages.",
    "Where is the nearest train station?",
    "Technology is changing the way we live.",
]


# ----------------------------------------------------------------
# Load saved tokenizers and model
# ----------------------------------------------------------------

def load_translation_model(
    tokenizer_dir=TOKENIZER_DIR,
    checkpoint_dir=CHECKPOINT_DIR,
):
    """Load saved tokenizers and restore the trained Encoder and Decoder."""
    en_tok = Tokenizer.from_file(os.path.join(tokenizer_dir, "en_tokenizer.json"))
    fr_tok = Tokenizer.from_file(os.path.join(tokenizer_dir, "fr_tokenizer.json"))

    encoder, decoder = build_models(en_tok.get_vocab_size(), fr_tok.get_vocab_size())
    load_checkpoint(encoder, decoder, checkpoint_dir=checkpoint_dir)
    return encoder, decoder, en_tok, fr_tok


# ----------------------------------------------------------------
# Output cleanup
# ----------------------------------------------------------------

def clean_text(text):
    """Turn tokenized output ("j ' ai décidé .") into normal text ("J'ai décidé.")."""
    text = re.sub(r"\s*'\s*", "'", text)        # j ' ai      -> j'ai
    text = re.sub(r"\s*-\s*", "-", text)        # peut - être -> peut-être
    text = re.sub(r"\s+([,.)])", r"\1", text)   # mot .       -> mot.
    text = re.sub(r"\(\s+", "(", text)          # ( mot       -> (mot
    text = text.strip()
    return text[:1].upper() + text[1:]


# ----------------------------------------------------------------
# Unit 9a: Greedy decoding
# ----------------------------------------------------------------

def greedy_decode(sentence, encoder, decoder, en_tok, fr_tok, max_len=50):
    """Translate an English sentence to French by picking the top token at each step."""
    src_ids = en_tok.encode(sentence).ids
    if not src_ids:
        return ""

    src_tensor = tf.constant([src_ids], dtype=tf.int32)
    state_h, state_c = encoder(src_tensor, training=False)

    dec_input = tf.constant([[SOS_ID]], dtype=tf.int32)
    output_ids = []

    for _ in range(max_len):
        logits, state_h, state_c = decoder(
            dec_input,
            initial_state=[state_h, state_c],
            training=False,
        )
        next_id = int(tf.argmax(logits[0, 0]).numpy())

        if next_id == EOS_ID:
            break

        output_ids.append(next_id)
        dec_input = tf.constant([[next_id]], dtype=tf.int32)

    return clean_text(fr_tok.decode(output_ids, skip_special_tokens=True))


# ----------------------------------------------------------------
# Unit 9b: Beam search decoding (all beams processed as one batch)
# ----------------------------------------------------------------

def beam_search_decode(
    sentence,
    encoder,
    decoder,
    en_tok,
    fr_tok,
    beam_width=4,
    max_len=50,
    length_penalty=0.7,
):
    """Translate an English sentence to French using beam search."""
    src_ids = en_tok.encode(sentence).ids
    if not src_ids:
        return ""

    def norm(score, length):
        return score / (max(length, 1) ** length_penalty)

    src_tensor = tf.constant([src_ids], dtype=tf.int32)
    h, c = encoder(src_tensor, training=False)

    tokens = [[SOS_ID]]          # active hypotheses (each starts with <sos>)
    scores = np.zeros(1)         # their cumulative log-probabilities
    completed = []               # finished hypotheses: (score, token_ids)

    for _ in range(max_len):
        last = tf.constant([[t[-1]] for t in tokens], dtype=tf.int32)  # (k, 1)
        logits, h, c = decoder(last, initial_state=[h, c], training=False)
        log_probs = tf.nn.log_softmax(logits[:, 0, :], axis=-1).numpy()  # (k, vocab)

        total = scores[:, None] + log_probs
        vocab = total.shape[1]
        flat = total.reshape(-1)

        # Best 2*beam_width candidates over all beams and all words
        k = min(2 * beam_width, flat.size - 1)
        top = np.argpartition(-flat, k)[:k]
        top = top[np.argsort(-flat[top])]

        new_tokens, new_scores, keep = [], [], []
        for idx in top:
            beam, tok = divmod(int(idx), vocab)
            score = float(flat[idx])
            if tok == EOS_ID:
                completed.append((score, tokens[beam][1:]))
            else:
                new_tokens.append(tokens[beam] + [tok])
                new_scores.append(score)
                keep.append(beam)
                if len(new_tokens) == beam_width:
                    break

        if not new_tokens:
            break

        tokens, scores = new_tokens, np.array(new_scores)
        h = tf.gather(h, keep)
        c = tf.gather(c, keep)

        # Stop once the best finished hypothesis beats the best unfinished one
        if len(completed) >= beam_width:
            best_done = max(norm(s, len(t) + 1) for s, t in completed)
            if best_done >= norm(scores[0], len(tokens[0]) - 1):
                break

    if not completed:
        completed = [(float(s), t[1:]) for s, t in zip(scores, tokens)]

    _, best_tokens = max(completed, key=lambda x: norm(x[0], len(x[1]) + 1))
    return clean_text(fr_tok.decode(best_tokens, skip_special_tokens=True))


# ----------------------------------------------------------------
# Demo, BLEU evaluation, interactive mode
# ----------------------------------------------------------------

def run_demo(encoder, decoder, en_tok, fr_tok):
    """Translate the demo sentences and save them to results/example_translations.txt."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    lines = []
    for sent in DEMO_SENTENCES:
        greedy = greedy_decode(sent, encoder, decoder, en_tok, fr_tok)
        beam = beam_search_decode(sent, encoder, decoder, en_tok, fr_tok)
        block = f"EN    : {sent}\nGreedy: {greedy}\nBeam  : {beam}\n"
        print(block)
        lines.append(block)
    path = os.path.join(RESULTS_DIR, "example_translations.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved to {path}\n")


def evaluate_bleu(encoder, decoder, en_tok, fr_tok, num_sentences=300, max_len=50, seed=0):
    """Score greedy and beam search on a random sample of test sentences with BLEU."""
    import sacrebleu  # imported here so the rest of the file works without it

    en_test, fr_test = load_split("test")
    candidates = [
        i for i, s in enumerate(en_test)
        if 0 < len(en_tok.encode(s).ids) <= max_len
    ]
    chosen = random.Random(seed).sample(candidates, min(num_sentences, len(candidates)))
    refs = [fr_test[i] for i in chosen]

    greedy_hyps, beam_hyps = [], []
    for n, i in enumerate(chosen, start=1):
        greedy_hyps.append(greedy_decode(en_test[i], encoder, decoder, en_tok, fr_tok))
        beam_hyps.append(beam_search_decode(en_test[i], encoder, decoder, en_tok, fr_tok))
        if n % 100 == 0:
            print(f"  translated {n}/{len(chosen)}")

    greedy_bleu = sacrebleu.corpus_bleu(greedy_hyps, [refs], lowercase=True).score
    beam_bleu = sacrebleu.corpus_bleu(beam_hyps, [refs], lowercase=True).score

    summary = (
        f"BLEU on {len(chosen)} random test sentences (source length <= {max_len} tokens)\n"
        f"Greedy: {greedy_bleu:.2f}\n"
        f"Beam  : {beam_bleu:.2f}\n"
    )
    print(summary)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "bleu.txt"), "w") as f:
        f.write(summary)


def interactive(encoder, decoder, en_tok, fr_tok):
    print("--- English to French Translator ---")
    print("Type an English sentence and press Enter (type 'quit' or 'exit' to stop).\n")
    while True:
        try:
            sent = input("EN: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not sent:
            continue
        if sent.lower() in ("quit", "exit"):
            break
        print(f"Greedy: {greedy_decode(sent, encoder, decoder, en_tok, fr_tok)}")
        print(f"Beam  : {beam_search_decode(sent, encoder, decoder, en_tok, fr_tok)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="English to French translation")
    parser.add_argument("--bleu", type=int, default=0,
                        help="score this many random test sentences with BLEU, then exit")
    parser.add_argument("--no-interactive", action="store_true",
                        help="skip the interactive prompt")
    args = parser.parse_args()

    encoder, decoder, en_tok, fr_tok = load_translation_model()

    if args.bleu:
        evaluate_bleu(encoder, decoder, en_tok, fr_tok, num_sentences=args.bleu)
    else:
        run_demo(encoder, decoder, en_tok, fr_tok)
        if not args.no_interactive:
            interactive(encoder, decoder, en_tok, fr_tok)