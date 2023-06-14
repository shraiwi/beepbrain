import tensorflow as tf
from tensorflow.keras import *
import keras_nlp
import song2vec

class BeepBrain(Sequential):
	def __init__(self, d_model=32, d_context=1024, d_out=(12+8)*4+2, dff=1024, num_heads=8, num_decoders=4, dropout=0.0):
		super().__init__(
			layers=(
				layers.Input(name="Input", shape=(d_context, d_model)),
				*(keras_nlp.layers.TransformerDecoder(name=f"Decoder{i}", intermediate_dim=dff, num_heads=num_heads) for i in range(num_decoders)),
				layers.Dense(name="Linear", units=d_out),
			),
			name="BeepBrain",
		)

if __name__ == "__main__":
	model = BeepBrain()

	model.summary()