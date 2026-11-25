import tensorflow as tf


# ----------------------------------------------------------------
# Model hyperparameters (shared by train.py and translate.py)
# ----------------------------------------------------------------

EMBED_DIM = 256
LSTM_UNITS = 512
DROPOUT = 0.2
PAD_ID = 0


# ----------------------------------------------------------------
# Unit 4: LSTM Encoder
# ----------------------------------------------------------------

class Encoder(tf.keras.Model):
    """
    Embeds English token IDs and encodes the sentence
    into final LSTM hidden and cell states (state_h, state_c).

    Padding is masked explicitly, so the returned state is the one
    after the last real token, not after the padding.
    """

    def __init__(self, vocab_size, embed_dim=EMBED_DIM, units=LSTM_UNITS, dropout=DROPOUT):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=True,  # Ignore PAD_ID (0)
        )
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.lstm = tf.keras.layers.LSTM(
            units=units,
            return_state=True,
        )

    def call(self, x, training=False):
        mask = tf.not_equal(x, PAD_ID)
        x = self.embedding(x)
        x = self.dropout(x, training=training)
        _, state_h, state_c = self.lstm(x, mask=mask, training=training)
        return state_h, state_c


# ----------------------------------------------------------------
# Unit 5: LSTM Decoder
# ----------------------------------------------------------------

class Decoder(tf.keras.Model):
    """
    Embeds French token IDs, runs an LSTM initialized with the
    encoder's final state, and predicts logits over the French vocabulary.
    """

    def __init__(self, vocab_size, embed_dim=EMBED_DIM, units=LSTM_UNITS, dropout=DROPOUT):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=True,  # Ignore PAD_ID (0)
        )
        self.emb_dropout = tf.keras.layers.Dropout(dropout)
        self.lstm = tf.keras.layers.LSTM(
            units=units,
            return_sequences=True,
            return_state=True,
        )
        self.out_dropout = tf.keras.layers.Dropout(dropout)
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, initial_state, training=False):
        mask = tf.not_equal(x, PAD_ID)
        x = self.embedding(x)
        x = self.emb_dropout(x, training=training)
        outputs, state_h, state_c = self.lstm(
            x,
            mask=mask,
            initial_state=initial_state,
            training=training,
        )
        outputs = self.out_dropout(outputs, training=training)
        logits = self.fc(outputs)
        return logits, state_h, state_c


# ----------------------------------------------------------------
# Helper: build both models so their weights exist
# ----------------------------------------------------------------

def build_models(src_vocab_size, tgt_vocab_size):
    """
    Create the encoder and decoder and run one dummy forward pass.
    Keras subclassed models have no weights until they are called once,
    so this must happen before loading a checkpoint into them.
    """
    encoder = Encoder(src_vocab_size)
    decoder = Decoder(tgt_vocab_size)

    dummy = tf.ones((1, 2), dtype=tf.int32)
    state_h, state_c = encoder(dummy, training=False)
    decoder(dummy, initial_state=[state_h, state_c], training=False)

    return encoder, decoder


# ----------------------------------------------------------------
# Quick self-test
# ----------------------------------------------------------------

if __name__ == "__main__":
    vocab_size = 20000
    batch_size = 64
    src_len = 14
    tgt_len = 22

    encoder, decoder = build_models(vocab_size, vocab_size)

    # Dummy batch matching the shapes from data.py
    dummy_src = tf.random.uniform((batch_size, src_len), minval=1, maxval=vocab_size, dtype=tf.int32)
    dummy_dec_in = tf.random.uniform((batch_size, tgt_len), minval=1, maxval=vocab_size, dtype=tf.int32)

    state_h, state_c = encoder(dummy_src, training=False)
    logits, dec_h, dec_c = decoder(dummy_dec_in, initial_state=[state_h, state_c], training=False)

    print("Encoder state_h:", state_h.shape)
    print("Encoder state_c:", state_c.shape)
    print("Decoder logits :", logits.shape)

    # Padding test: the same sentence with and without trailing padding
    # must produce (almost) identical encoder states.
    short = tf.constant([[5, 6, 7]], dtype=tf.int32)
    padded = tf.constant([[5, 6, 7, 0, 0, 0]], dtype=tf.int32)
    h1, c1 = encoder(short, training=False)
    h2, c2 = encoder(padded, training=False)
    max_diff = float(tf.reduce_max(tf.abs(h1 - h2)))
    print("Padding test max difference:", max_diff)
    assert max_diff < 1e-4, "Encoder state changes with padding. Masking is not working."
    print("Padding test passed.")

    print("Parameters:",
          f"encoder {encoder.count_params():,} | decoder {decoder.count_params():,}")