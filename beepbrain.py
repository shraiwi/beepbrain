import tensorflow as tf
import numpy as np
from tensorflow.keras import *
import keras_nlp
import song2vec, note2vec
import json
from pathlib import Path
from song2vec import Song

class BeepBrain(Model):
	def __init__(self, name="BeepBrain", d_model=32, d_seq=1024, d_out=(12+8)*4+2, d_ff=1024, num_heads=8, num_decoders=4, dropout=0.0):
		layer_input_seq = layers.Input(name="sequence", shape=(d_seq, d_model))
		layer_input_mask = layers.Input(name="mask", shape=(d_seq,))

		layer_decoder = layer_input_seq
		for i in range(num_decoders):
			layer_decoder = keras_nlp.layers.TransformerDecoder(
				name=f"decoder{i}", intermediate_dim=d_ff, num_heads=num_heads)(
					layer_decoder, decoder_padding_mask=layer_input_mask
				)

		layer_linear = layers.Dense(name="linear", units=d_out)(layer_decoder)

		super().__init__(name=name, 
			inputs=[layer_input_seq, layer_input_mask],
			outputs=[layer_linear])

		self.d_model = d_model
		self.d_seq = d_seq
		self.d_out = d_out
		self.d_ff = d_ff
		self.num_heads = num_heads
		self.num_decoders = num_decoders
		self.dropout = dropout

class NoteTokenLoss(losses.Loss):
	def __init__(self, tokenizer, from_logits=True):
		super().__init__()
		self.tokenizer = tokenizer
		self.cross_loss = losses.CategoricalCrossentropy(from_logits=from_logits)
		self.binary_loss = losses.BinaryCrossentropy(from_logits=from_logits)

	def call(self, y_true, y_pred):
		true_metadata = y_true[:Song.metadata_size()]
		pred_metadata = y_pred[:Song.metadata_size()]

		true_is_occupied, true_semitones, true_octaves = self.tokenizer.split(y_true[Song.metadata_size():, ...])
		pred_is_occupied, pred_semitones, pred_octaves = self.tokenizer.split(y_pred[Song.metadata_size():, ...])

		loss_metadata = self.binary_loss(true_metadata, pred_metadata)
		loss_is_occupied = self.binary_loss(true_is_occupied, pred_is_occupied)
		loss_semitones = self.cross_loss(true_semitones, pred_semitones)
		loss_octaves = self.cross_loss(true_octaves, pred_octaves)

		return loss_metadata + loss_is_occupied + loss_semitones + loss_octaves

# use tf.padded_batch ....

def render_dir(folder_dir, tokenizer):
	"""
	loads songs, and yields (dense, sparse) rendered pairs per permutation.
	"""
	for json_path in Path(folder_dir).glob("*.json"):
		with json_path.open() as f_json:
			song_json = json.load(f_json)
			song = Song(song_json, tokenizer=tokenizer)

			yield song.render("dense"), song.render("sparse")

def masked_sliding_window(x, d_seq, stride, pad_value=0):
	"""
	splits x into chunks of (d_seq, ...) size, stepping by stride every time.

	automatically pads chunks to size d_seq, and returns a mask indicating where the chunk contains padded data.
	"""

	x = np.asarray(x)

	for start_idx in range(0, x.shape[0], stride):
		x_window = x[start_idx:start_idx+d_seq, ...]

		chunk = np.full_like(x, pad_value, shape=(d_seq, *x_window.shape[1:]))
		mask = np.ones(d_seq, dtype=bool)

		chunk[:x_window.shape[0], ...] = x_window
		mask[x_window.shape[0]:] = False

		yield mask, chunk

def make_song_dataset(folder_dir, tokenizer, d_seq):
	"""
	"""
	def _generator_func(folder_dir, tokenizer, d_seq):
		sliding_window_args = {
			"d_seq": d_seq,
			"stride": d_seq // 2
		}
		for dense_render, sparse_render in render_dir(folder_dir, tokenizer):
			for (mask, x), (_, y) in zip(masked_sliding_window(dense_render, **sliding_window_args), 
				masked_sliding_window(sparse_render, **sliding_window_args)):

				yield (
					(tf.convert_to_tensor(x), tf.convert_to_tensor(mask)), 
					tf.convert_to_tensor(y))

	return tf.data.Dataset.from_generator(
		lambda: _generator_func(folder_dir, tokenizer, d_seq),
		output_signature=(
			(
				tf.TensorSpec(name="x", 
					shape=(d_seq, tokenizer.token_size["dense"] + Song.metadata_size()), 
					dtype=tf.float32),
				tf.TensorSpec(name="mask", 
					shape=(d_seq), dtype=tf.bool),
			),
			tf.TensorSpec(name="y", 
				shape=(d_seq, tokenizer.token_size["sparse"] + Song.metadata_size()), 
				dtype=tf.float32), )
		)

if __name__ == "__main__":
	import matplotlib.pyplot as plt

	tokenizer = note2vec.NoteTokenizer()


	dataset = make_song_dataset("parsed_archive", tokenizer, 1024)

	for (x, mask), y in dataset.take(10).as_numpy_iterator():
		if tf.math.reduce_all(mask) == False:
			plt.imshow(y.T)
			plt.show()
		print(mask.shape, x.shape, y.shape)

	print(*masked_sliding_window([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], 4, 2))

	loss = NoteTokenLoss(tokenizer, from_logits=False)

	true_song = [
		(1, 2),
		(5, -1),
		(4, 3),
	]

	pred_song = [
		(3, 2),
		(5, -1),
		(4, 3),
	]

	print(loss.call(
		tokenizer.encode(true_song, method="sparse"), 
		tokenizer.encode(pred_song, method="sparse"))
	)

	model = BeepBrain()

	model.summary()
