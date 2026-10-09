#!/bin/bash
#
# Prepares the LibriSpeech transcripts examples/conformer/train.py trains on, with this project's own
# script, scripts/create_librispeech_trans.py, as tensorflow_asr/datasets/README.md directs, and writes
# a copy of examples/conformer/config.yml whose data paths point at them. Run from the project root
# ($PYTHON may name the interpreter to use).
#
# Inputs: the three LibriSpeech archives from https://www.openslr.org/12 (train-clean-100, dev-clean,
# test-clean) in $LIBRISPEECH_DIR.
# Outputs, all under $LIBRISPEECH_DIR and none in this checkout:
#   LibriSpeech/<subset>/            each archive, extracted
#   LibriSpeech/<subset>/transcripts.tsv   create_librispeech_trans.py's output for that subset
#   conformer-config.yml             examples/conformer/config.yml with its H:/ dataset prefix replaced
#                                    by the extracted tree and its D:/ output prefix by ./my_train
#                                    (in this project's .gitignore), and its training-time writes
#                                    switched off: no checkpoint (save_freq past any run's length),
#                                    and TensorBoard without histograms, graph, images or profiling
# Idempotent: an extracted subset and a written transcript are kept and not redone.

set -euo pipefail

PYTHON="${PYTHON:-python3.10}"
LIBRISPEECH_DIR="${LIBRISPEECH_DIR:-$HOME/.cache/python-subjects/assets/librispeech}"
SUBSETS="train-clean-100 dev-clean test-clean"

for subset in $SUBSETS; do
	marker="$LIBRISPEECH_DIR/.extracted-$subset"
	if [[ ! -e $marker ]]; then
		tar -xzf "$LIBRISPEECH_DIR/$subset.tar.gz" -C "$LIBRISPEECH_DIR"
		touch "$marker"
	fi

	out="$LIBRISPEECH_DIR/LibriSpeech/$subset/transcripts.tsv"
	if [[ ! -s $out ]]; then
		# Written beside its final name and renamed, so an interrupted run leaves nothing that reads as
		# done. The script imports tensorflow_asr from this checkout, hence the path; it uses no GPU.
		CUDA_VISIBLE_DEVICES= PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}" \
			"$PYTHON" scripts/create_librispeech_trans.py --dir "$LIBRISPEECH_DIR/LibriSpeech/$subset" "$out.partial"
		mv "$out.partial" "$out"
	fi
done

# Regenerated every time from the checked-out config, so it never goes stale against it. A textual
# substitution, not a YAML round trip: the loader's own float resolver (file_util.load_yaml) reads
# values such as `1e-9` that a re-dump would quote, so the rest of the file stays byte for byte.
config="$LIBRISPEECH_DIR/conformer-config.yml"
sed -e "s|H:/MLDL/Datasets/ASR/Raw/LibriSpeech/|$LIBRISPEECH_DIR/LibriSpeech/|" \
	-e "s|D:/Models/local/conformer/|./my_train/conformer/|" \
	-e "s|^\(      save_freq:\) epoch$|\1 1000000000|" \
	-e "s|^\(      histogram_freq:\) 1$|\1 0|" \
	-e "s|^\(      write_graph:\) True$|\1 False|" \
	-e "s|^\(      write_images:\) True$|\1 False|" \
	-e "s|^\(      profile_batch:\) 2$|\1 0|" \
	examples/conformer/config.yml > "$config.partial"
if grep -qE '(H|D):/' "$config.partial"; then
	echo "setup.sh: examples/conformer/config.yml has a Windows path this script does not map." >&2
	exit 1
fi
if grep -qE '^ +(save_freq: epoch|histogram_freq: [1-9]|write_graph: True|write_images: True|profile_batch: [1-9])' "$config.partial"; then
	echo "setup.sh: examples/conformer/config.yml has a training-time write this script does not switch off." >&2
	exit 1
fi
mv "$config.partial" "$config"
