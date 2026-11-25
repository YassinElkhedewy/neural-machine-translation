# English-French Neural Machine Translation (LSTM encoder-decoder)

An LSTM encoder-decoder translation model (English to French), trained on the
IWSLT 2017 en-fr dataset using TensorFlow/Keras.

## Files
- `data.py`: dataset loader, word-level tokenizers (20k vocab per language), batch generator
- `model.py`: LSTM encoder and LSTM decoder
- `train.py`: training loop, validation, checkpoint saving/loading
- `translate.py`: greedy and beam search decoding, interactive translator, BLEU evaluation
- `tokenizers/`: saved tokenizers (must match the trained weights)
- `results/`: example translations and BLEU score

## Setup
Python 3.12 is recommended (TensorFlow does not support newer versions yet).
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
The dataset downloads automatically on first run.

## Usage
```bash
python train.py                      # train (about 2 min/epoch on an RTX 5060)
python translate.py                  # example translations, then an interactive prompt
python translate.py --bleu 300       # BLEU on 300 random test sentences
```
Training writes `checkpoints/encoder.weights.h5` and `checkpoints/decoder.weights.h5`.
These weight files are too large for a browser upload, so run `python train.py`
to regenerate them before using `translate.py`.

## Model
- Encoder: embedding (256) + LSTM (512 units); its final hidden and cell states initialize the decoder
- Decoder: embedding (256) + LSTM (512 units) + dense layer over the French vocabulary
- Dropout 0.2, Adam (lr 1e-3, halved when validation loss stops improving), gradient clipping,
  padding-masked loss, early stopping. Sentences longer than 50 tokens are filtered out.

## Results
- Best validation loss: 2.685 (perplexity 14.7)
- BLEU on 300 random test sentences: greedy 12.37, beam search 13.07
- Example translations are in `results/example_translations.txt`

This is a plain encoder-decoder without attention, so long or rare-word sentences
are translated less reliably.
