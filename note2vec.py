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

def embed(pitches, size=12, octave_range=8, dtype=ftype):
	"""
	generates an embedding for a set of pitches based on the note and semitone embeddings
	"""
	pitches = np.asarray(pitches)

	output_shape = (*pitches.shape, size)
	output_size = np.prod(output_shape)

	octaves, semitones = np.divmod(pitches, SEMITONE_COUNT)

	octaves_norm, semitones_norm = np.clip((octaves / octave_range, semitones / SEMITONE_COUNT), 0.0, 1.0)

	xvals, dx = np.linspace(0.0, 1.0, size, dtype=dtype, retstep=True)

	return embed_octaves(xvals, octaves_norm) * embed_semitones(xvals, semitones_norm)

class NoteTokenizer():
	def __init__(self, chord_size=4, pitch_size=6, octave_range=8):
		self.chord_size = chord_size
		self.pitch_size = pitch_size
		self.octave_range = octave_range

		self.token_size = self.chord_size * self.pitch_size

		if True:
			# normal embeddings
			self.lut = embed(np.arange(self.octave_range * SEMITONE_COUNT, dtype=itype), size=self.pitch_size)
			self.null_pitch = np.zeros_like(self.lut[0])
		else:
			warnings.warn("using debug embeddings")
			# debug embeddings
			self.lut = np.repeat(np.arange(self.octave_range * SEMITONE_COUNT, dtype=itype)[..., None], self.pitch_size, axis=-1)
			self.null_pitch = np.array([-1] * self.pitch_size)

		self.lut = np.concatenate((self.lut, self.null_pitch[None, ...]))
		self.null_idx = self.lut.shape[0] - 1

	def encode(self, pitches, debias=True, sort=True, clip=True):
		"""
		turns a pitch, chord or array of chords into a token or an array of tokens. 
		to represent a unplayed key, use any negative pitch value.

		Parameters
		----------
		pitches
			a single pitch, chord, or array of chords to turn into tokens
		debias
			whether or not to subtract the lowest pitch from all other pitches in the chord.
		sort
			whether or not to sort the pitches.
		clip
			whether or not to rescale the pitches to fit within the tokenizer's octave range.
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
		oob_pitches_mask = (pitches >= self.lut.shape[0]) & (pitches >= 0)
		oob_octave, _ = np.divmod(pitches, SEMITONE_COUNT, where=oob_pitches_mask, out=(np.zeros_like(pitches), pitches))
		pitches += np.minimum(oob_octave, self.octave_range - 1) * SEMITONE_COUNT

		# set null pitches to null index
		null_entries = pitches < 0
		pitches[null_entries] = self.null_idx

		if sort: pitches.sort()

		if debias:
			pitch_children = pitches[..., 1:]
			np.subtract(pitch_children, pitches[..., 0, None], where=pitch_children != self.null_idx, out=pitch_children) # determine child offset from root note

		return self.lut[pitches].reshape((*pitches.shape[:-1], self.token_size))

if __name__ == "__main__":
	import matplotlib.pyplot as plt

	tokenizer = NoteTokenizer()

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


