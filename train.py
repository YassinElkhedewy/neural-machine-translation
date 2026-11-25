import csv
import math
import os
import time

import tensorflow as tf

from data import BASE_DIR, PAD_ID, prepare_data
from model import build_models


# ----------------------------------------------------------------
# Training hyperparameters
# ----------------------------------------------------------------

EPOCHS = 12
BATCH_SIZE = 64
VOCAB_SIZE = 20000
MAX_LEN = 50
LEARNING_RATE = 1e-3
PATIENCE = 3  # stop early after this many epochs without improvement

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ENCODER_FILE = "encoder.weights.h5"
DECODER_FILE = "decoder.weights.h5"


# ----------------------------------------------------------------
# Masked loss (ignores PAD_ID tokens)
# ----------------------------------------------------------------

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction="none",
)


def masked_loss(real, pred):
    """Compute cross-entropy loss while ignoring <pad> positions."""
    loss = loss_object(real, pred)
    mask = tf.cast(tf.not_equal(real, PAD_ID), dtype=tf.float32)
    loss *= mask
    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1.0)


# ----------------------------------------------------------------
# Unit 8: Checkpoint saving and loading
# ----------------------------------------------------------------

def save_checkpoint(encoder, decoder, optimizer=None, checkpoint_dir=CHECKPOINT_DIR):
    """
    Save encoder and decoder weights (overwriting the previous ones).

    The optimizer argument is kept for compatibility but is not saved:
    weights only keeps the files small enough to push to GitHub.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    encoder.save_weights(os.path.join(checkpoint_dir, ENCODER_FILE))
    decoder.save_weights(os.path.join(checkpoint_dir, DECODER_FILE))
    return checkpoint_dir


def load_checkpoint(encoder, decoder, optimizer=None, checkpoint_dir=CHECKPOINT_DIR):
    """Load the saved weights into encoder and decoder."""
    enc_path = os.path.join(checkpoint_dir, ENCODER_FILE)
    dec_path = os.path.join(checkpoint_dir, DECODER_FILE)
    if not (os.path.exists(enc_path) and os.path.exists(dec_path)):
        raise FileNotFoundError(f"No checkpoint found in '{checkpoint_dir}'.")

    # Keras models have no weights until called once, so build them first.
    if not (encoder.built and decoder.built):
        dummy = tf.ones((1, 2), dtype=tf.int32)
        state_h, state_c = encoder(dummy, training=False)
        decoder(dummy, initial_state=[state_h, state_c], training=False)

    encoder.load_weights(enc_path)
    decoder.load_weights(dec_path)
    print(f"Loaded checkpoint from: {checkpoint_dir}")
    return checkpoint_dir


# ----------------------------------------------------------------
# Unit 6 & 7: Compiled Training and Validation steps
# ----------------------------------------------------------------

def make_train_step(encoder, decoder, optimizer):
    """Create a compiled tf.function training step."""

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None], dtype=tf.int32),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        ]
    )
    def train_step(src, dec_in, dec_out):
        with tf.GradientTape() as tape:
            state_h, state_c = encoder(src, training=True)
            logits, _, _ = decoder(
                dec_in,
                initial_state=[state_h, state_c],
                training=True,
            )
            loss = masked_loss(dec_out, logits)

        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss

    return train_step


def make_val_step(encoder, decoder):
    """Create a compiled tf.function validation step."""

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None], dtype=tf.int32),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        ]
    )
    def val_step(src, dec_in, dec_out):
        state_h, state_c = encoder(src, training=False)
        logits, _, _ = decoder(
            dec_in,
            initial_state=[state_h, state_c],
            training=False,
        )
        return masked_loss(dec_out, logits)

    return val_step


# ----------------------------------------------------------------
# Unit 7: Validation
# ----------------------------------------------------------------

def validate(val_gen, val_step):
    """Compute average translation loss over the validation split."""
    total_loss = 0.0
    num_batches = 0

    for src, dec_in, dec_out in val_gen:
        loss = val_step(
            tf.convert_to_tensor(src, dtype=tf.int32),
            tf.convert_to_tensor(dec_in, dtype=tf.int32),
            tf.convert_to_tensor(dec_out, dtype=tf.int32),
        )
        total_loss += float(loss)
        num_batches += 1

    return total_loss / max(num_batches, 1)


# ----------------------------------------------------------------
# Unit 6: Training loop
# ----------------------------------------------------------------

def train(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    vocab_size=VOCAB_SIZE,
    max_len=MAX_LEN,
    learning_rate=LEARNING_RATE,
    patience=PATIENCE,
    checkpoint_dir=CHECKPOINT_DIR,
    results_dir=RESULTS_DIR,
    max_steps=None,
):
    """
    Full training and validation loop with best-checkpoint saving.

    max_steps limits the number of training batches per epoch.
    Use it only for quick smoke tests, e.g. train(epochs=1, max_steps=100).
    """
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    print("Preparing data...")
    train_gen, val_gen, en_tok, fr_tok = prepare_data(
        batch_size=batch_size,
        vocab_size=vocab_size,
        max_len=max_len,
    )

    encoder, decoder = build_models(
        en_tok.get_vocab_size(),
        fr_tok.get_vocab_size(),
    )
    print(
        f"Parameters: encoder {encoder.count_params():,} | "
        f"decoder {decoder.count_params():,}"
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        global_clipnorm=1.0,
    )

    train_step = make_train_step(encoder, decoder, optimizer)
    val_step = make_val_step(encoder, decoder)

    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "val_perplexity", "learning_rate", "seconds"]
        )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    current_lr = learning_rate

    total_batches = len(train_gen)
    if max_steps:
        total_batches = min(total_batches, max_steps)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        total_loss = tf.constant(0.0)  # accumulated on the GPU, read only when printing

        for step, (src, dec_in, dec_out) in enumerate(train_gen, start=1):
            if max_steps and step > max_steps:
                break

            loss = train_step(
                tf.convert_to_tensor(src, dtype=tf.int32),
                tf.convert_to_tensor(dec_in, dtype=tf.int32),
                tf.convert_to_tensor(dec_out, dtype=tf.int32),
            )
            total_loss += loss

            if step % 500 == 0 or step == total_batches:
                avg_so_far = float(total_loss) / step
                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"Step {step}/{total_batches} | "
                    f"Train Loss: {avg_so_far:.4f}"
                )

        train_loss = float(total_loss) / total_batches
        val_loss = validate(val_gen, val_step)
        val_ppl = math.exp(min(val_loss, 20))
        elapsed = time.time() - start_time

        print(
            f"--> Epoch {epoch}/{epochs} finished in {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Perplexity: {val_ppl:.1f}"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.4f}", f"{val_loss:.4f}",
                 f"{val_ppl:.2f}", f"{current_lr:.6f}", f"{elapsed:.1f}"]
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(encoder, decoder, optimizer, checkpoint_dir)
            print(f"    New best model saved to: {checkpoint_dir}")
        else:
            epochs_without_improvement += 1
            current_lr = max(current_lr * 0.5, 1e-5)
            optimizer.learning_rate.assign(current_lr)
            print(f"    Val loss did not improve. Lowered learning rate to {current_lr:.6f}")

            if epochs_without_improvement >= patience:
                print(f"    No improvement for {patience} epochs. Stopping early.")
                break

    print(f"Training finished. Best validation loss: {best_val_loss:.4f}")
    return encoder, decoder


if __name__ == "__main__":
    train()