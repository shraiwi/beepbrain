import numpy as np
import warnings

SEMITONE_COUNT = 12 # oop looks like we can do microtones lol

ftype = np.float32
itype = np.int32

def embed_octaves(xvals, octaves_norm, ratio=0.4):
	"""
	generate an embedding for an octave using the function 
		sin(2 * pi * (x - octave_norm * 0.5)) * ratio + (1.0 - ratio)
	"""
	return np.sin(2.0 * np.pi * (xvals - octaves_norm[..., None] * 0.5)) * ratio + (1.0 - ratio)

def embed_semitones(xvals, semitones_norm):
	"""
	generate an embedding for a semitone using the function 
		cos(2 * pi * (x - semitone_norm))
	"""
	return np.cos(2.0 * np.pi * (xvals - semitones_norm[..., None]))

def embed_pitches(pitches, method="dense", size=12, octave_range=8, dtype=ftype):
	"""
	generates an embedding for an array of pitches based on the note and semitone embeddings.
	"""
	pitches = np.asarray(pitches)

	octaves, semitones = np.divmod(pitches, SEMITONE_COUNT)

	if method == "dense": # continuous embedding of pitches
		octaves_norm, semitones_norm = np.clip((octaves / octave_range, semitones / SEMITONE_COUNT), 0.0, 1.0)
		xvals, dx = np.linspace(0.0, 1.0, size, dtype=dtype, retstep=True)

		return embed_octaves(xvals, octaves_norm) * embed_semitones(xvals, semitones_norm)
	elif method == "sparse": # one-hot embed pitches. this will ignore the "size" parameter

		token_size = octave_range + SEMITONE_COUNT + 1

		out = np.zeros((pitches.shape[0], token_size), dtype=dtype)

		idxs = np.arange(pitches.size, dtype=itype)

		# is_occupied, semitones on top, octaves on bottom.

		out[idxs, 0] = 1.
		out[idxs, 1 + semitones] = 1.
		out[idxs, 1 + SEMITONE_COUNT + octaves] = 1.

		return out

class NoteTokenizer():
	def __init__(self, chord_size=4, pitch_size=6, octave_range=8):
		self.chord_size = chord_size
		self.octave_range = octave_range
		self.num_pitches = self.octave_range * SEMITONE_COUNT

		pitch_range = np.arange(self.num_pitches, dtype=itype)

		# Generate dense and sparse look-up tables
		self.lut = {
			"dense": embed_pitches(pitch_range, method="dense", size=pitch_size, octave_range=self.octave_range),
			"sparse": embed_pitches(pitch_range, method="sparse", octave_range=self.octave_range)
		}
		self.token_size = {}

		for method in self.lut:

			null_pitch = np.zeros_like(self.lut[method][0])

			self.token_size[method] = null_pitch.size * self.chord_size
			self.lut[method] = np.concatenate((self.lut[method], null_pitch[None, ...]))

	def cast_pitches(self, pitches):
		"""
		clamps and pads a pitch, chord, or array of chords into a valid range.

		ignores negative pitches.
		"""

		pitches = np.asarray(pitches).astype(itype)

		if len(pitches.shape) == 0:
			# automatically convert a note into a chord containing just the note.
			pitches = np.pad(pitches[..., None], (0, self.chord_size - 1), constant_values=-1)
		elif pitches.shape[-1] > self.chord_size:
			warnings.warn(f"limiting chord note count to {self.chord_size} (from {pitches.shape[-1]})")
			pitches = pitches[..., :self.chord_size]
		elif pitches.shape[-1] < self.chord_size:
			warnings.warn(f"padding chord note count to {self.chord_size} (from {pitches.shape[-1]})")
			pitches = np.pad(pitches, (0, self.chord_size - pitches.shape[-1]), constant_values=-1)

		# try to intelligently clamp pitches by retaining their semitone while clamping the octave to the max available one.
		# not perfect, but good enough as it retains the relationship between pitches.
		oob_pitches_mask = pitches >= self.num_pitches
		oob_octave, _ = np.divmod(pitches, SEMITONE_COUNT, where=oob_pitches_mask, out=(np.zeros_like(pitches), pitches))
		pitches += np.minimum(oob_octave, self.octave_range - 1) * SEMITONE_COUNT

		return pitches

	def split(self, tokens, method="sparse"):
		"""
		splits tokens into its constituent parts
		for dense tokens, this will be the pitches that it is constructed from.
		for sparse token this will be the pitches in a (is_occupied, semitone, octave) array

		(pitch_0, pitch_1, ...)
		"""

		for pitch_idx in range(self.chord_size):
			if method == "dense":
				raise NotImplementedError()
			elif method == "sparse":
				pitch_offset = self.token_size[method] * pitch_idx
				yield (
					tokens[..., pitch_offset + 0 : pitch_offset + 1], 
					tokens[..., pitch_offset + 1 : pitch_offset + 1 + SEMITONE_COUNT], 
					tokens[..., pitch_offset + 1 + SEMITONE_COUNT : pitch_offset + self.octave_range]
				)

	def encode(self, pitches, relative=True, sort=True, method="dense"):
		"""
		turns a pitch, chord or array of chords into a token or an array of tokens.
		to represent an unplayed key, use any negative pitch value.

		Parameters
		----------
		pitches
			a single pitch, chord, or array of chords to turn into tokens
		relative
			whether or not to subtract the lowest pitch from all other pitches in the chord.
		sort
			whether or not to sort the pitches.
		method
			the lookup table to use for encoding ("dense" or "sparse").
		"""
		pitches = self.cast_pitches(pitches)

		# set null pitches to null index
		null_entries = pitches < 0
		pitches[null_entries] = self.num_pitches

		if sort:
			pitches.sort()

		if relative:
			pitch_children = pitches[..., 1:]
			np.subtract(pitch_children, pitches[..., 0, None], where=pitch_children != self.num_pitches,
				out=pitch_children)  # determine child offset from root note

		return self.lut[method][pitches].reshape((*pitches.shape[:-1], self.token_size[method]))

if __name__ == "__main__":
	import matplotlib.pyplot as plt

	tokenizer = NoteTokenizer(method="sparse")

	print(tokenizer.lut.shape)

	print(tokenizer.encode(-1))

	dots = np.dot(tokenizer.lut, tokenizer.lut[..., None])

	plt.imshow(dots)
	plt.show()
	plt.clf()

	print(tokenizer.encode([
		(1, 1, 5, 4),
		(0, 5, -1, 7),
		(0, 5, -1, 22),
		(1, 2, 3, -1),
		(0, -1, -1, -1),
	]))

	print(tokenizer.encode([
		(1, 1, 5, 4, 5),
	]))

	print(tokenizer.encode([
		(1, 3),
	]))

	print(tokenizer.encode([1, 2, 3]))

	print(tokenizer.encode([1, 2, 3, 4]))

	plt.imshow(tokenizer.encode([ (1, 1, 2, 3), (3, 4, 5, 6), (0, 1, -1, -1), (25, 27, 29, 30) ]).T)
	plt.show()


