"""Train DeepSpeech2 for a few steps on random features of varying length, without data.

Builds the model from configs/vivos.yml, with a synthetic character vocabulary in place of the
config's absolute vocabulary path. Each step feeds random (batch, T, 80, 1) features, for ten
values of T, through a training step (the model's CtcModel.call with training=True, the package's
ctc_loss, and Adam) and then a training=False forward pass, so both are called at varying T.

Two things are done here rather than in the package:

* `ctc_decoders`, a compiled extension imported at the top of tiramisu_asr/models/ctc.py for greedy
  and beam-search decoding, is replaced by a stub whose functions raise if called. Nothing on the
  training or forward path calls them.
* The training step is written here rather than taken from tiramisu_asr.runners, whose modules
  import tensorflow.keras.mixed_precision.experimental, which TensorFlow 2.9 no longer has.
"""
import os
import sys
import tempfile
import time
import types

import numpy as np
import tensorflow as tf
import yaml


def _stub_ctc_decoders():
  def unavailable(*args, **kwargs):
    raise NotImplementedError("ctc_decoders is stubbed; decoding is not used by this driver")
  module = types.ModuleType("ctc_decoders")
  module.ctc_greedy_decoder = unavailable
  module.ctc_beam_search_decoder = unavailable
  sys.modules.setdefault("ctc_decoders", module)


_stub_ctc_decoders()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tiramisu_asr.featurizers.speech_featurizers import TFSpeechFeaturizer  # noqa: E402
from tiramisu_asr.featurizers.text_featurizers import TextFeaturizer  # noqa: E402
from tiramisu_asr.losses.ctc_losses import ctc_loss  # noqa: E402
from model import DeepSpeech2  # noqa: E402

BATCH = 2
LENGTHS = [50, 83, 120, 157, 200, 233, 270, 311, 350, 400]
REPEATS = 2


def main():
  with open(os.path.join(HERE, "configs", "vivos.yml")) as handle:
    config = yaml.safe_load(handle)

  with tempfile.TemporaryDirectory() as directory:
    vocabulary = os.path.join(directory, "vocabulary.txt")
    with open(vocabulary, "w") as handle:
      handle.write("\n".join(list(" abcdefghijklmnopqrstuvwxyz'")) + "\n")
    config["decoder_config"]["vocabulary"] = vocabulary
    config["decoder_config"]["lm_config"] = None
    speech_featurizer = TFSpeechFeaturizer(config["speech_config"])
    text_featurizer = TextFeaturizer(config["decoder_config"])

  feature_dim, channels = speech_featurizer.compute_feature_dim()
  tf.random.set_seed(2020)
  rng = np.random.RandomState(2020)
  model = DeepSpeech2(input_shape=[None, feature_dim, channels],
                      arch_config=config["model_config"],
                      num_classes=text_featurizer.num_classes, name="deepspeech2")
  model._build([1, 50, feature_dim, channels])
  optimizer = tf.keras.optimizers.Adam(1e-4)
  reduction = model.time_reduction_factor

  start = time.perf_counter()
  for _ in range(REPEATS):
    for length in LENGTHS:
      features = tf.random.normal([BATCH, length, feature_dim, channels])
      input_length = tf.fill([BATCH], length)
      label_length = rng.randint(1, max(2, -(-length // reduction) // 4), size=BATCH)
      labels = np.zeros([BATCH, label_length.max()], dtype=np.int32)
      for b in range(BATCH):
        labels[b, :label_length[b]] = rng.randint(0, text_featurizer.num_classes - 1,
                                                  size=label_length[b])
      with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        per_example = ctc_loss(y_true=labels, y_pred=predictions,
                               input_length=input_length // reduction,
                               label_length=label_length, blank=text_featurizer.blank)
        loss = tf.nn.compute_average_loss(per_example, global_batch_size=BATCH)
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      model(features, training=False)
      print("T=%d logits=%s loss=%.3f" % (length, tuple(predictions.shape), float(loss)))
  print("CtcModel.call traces: %d ; ctc_loss traces: %d ; elapsed %.1fs"
        % (model.call.experimental_get_tracing_count(), ctc_loss.experimental_get_tracing_count(),
           time.perf_counter() - start))


if __name__ == "__main__":
  main()
