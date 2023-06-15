import json
import pandas as pd
import numpy as np
from pathlib import Path
from pprint import pprint
import timeit

import note2vec
from note2vec import NoteTokenizer

import itertools

# your goal is to "render" each channel into a single track.

"""
beepbox songs:

note timing is in "parts":
	there are 24 parts in a beat. there are 3-16 beats in a bar.

in this music system, there is always 48 ticks in a bar. this allows for pretty good subdivision and lower quantization error.
"""

"""
there are 3 different ways we can generate variations of a song:
- shuffling channels: choose 1-i note channels and permute, choose 0-i drum channels and permute, and render.
- 
"""

def pat2chords(pat, parts_per_tick, out=None, ticks_per_pattern=48, tokenizer=note2vec.NoteTokenizer()):
	"""
	convert a pattern into chords
	"""
	chords_shape = (ticks_per_pattern, tokenizer.chord_size)

	if out is None:
		out = np.full(chords_shape, -1, dtype=note2vec.itype) # init chords to all empty
	
	assert out.shape == chords_shape, f"out is of invalid shape (expected {chords_shape}, got {out.shape})"

	for note_start, note_end, note_pitches in pat:
		note_start_ticks, note_end_ticks = note_start // parts_per_tick, note_end // parts_per_tick

		#print(note_start_ticks, note_end_ticks, note_pitches)

		out[note_start_ticks:note_end_ticks, :len(note_pitches)] = note_pitches

	return out

def bit_count(n):
	return int(np.log2(n) + 0.5)

class Song:

	MAX_CHANNELS = 15

	def metadata_size():
		return bit_count(Song.MAX_CHANNELS) + 1

	def __init__(self, song_data, tokenizer=NoteTokenizer(), ticks_per_pattern=48, include_metadata=True):

		self.ticks_per_pattern = ticks_per_pattern
		self.tokenizer = tokenizer

		self.parts_per_tick = song_data["beatsPerBar"] * 24 // ticks_per_pattern;

		self.rendered_channels = { "sparse": [], "dense": [], }
		self.channel_types = { "note": [], "drum": [], }

		self.bar_indices = [] 
		self.rendered_channel_chords = []

		# render channels into chords
		for chan_idx, chan in enumerate(song_data["channels"]):
			channel_chords = np.full(
				(len(chan["patterns"]) + 1, self.ticks_per_pattern, self.tokenizer.chord_size), 
				-1, 
				dtype=note2vec.itype)
			channel_bars = [0 if pat_idx is None else (pat_idx + 1) for pat_idx in chan["bars"]]

			is_relative = { "note": True, "drum": False }[chan["instrumentType"]]

			for pat_idx, pat in enumerate(chan["patterns"]):
				pat2chords(pat, self.parts_per_tick, out=channel_chords[pat_idx + 1], tokenizer=self.tokenizer)

				# clamp chords
				channel_chords[pat_idx + 1] = self.tokenizer.cast_pitches(channel_chords[pat_idx + 1])

			rendered_channel_chords = np.concatenate(tuple(channel_chords[channel_bars]), axis=0)

			if include_metadata:
				channel_metadata = np.zeros(Song.metadata_size(), dtype=note2vec.ftype)

				channel_metadata[0] = float(is_relative)
				for bit_num in range(bit_count(Song.MAX_CHANNELS)):
					channel_metadata[1 + bit_num] = float(bool(chan_idx & (1 << bit_num)))
			else:
				channel_metadata = np.zeros(0)

			for method in ("sparse", "dense"):
				rendered_channel = self.tokenizer.encode(
					rendered_channel_chords, 
					method=method,
					relative=is_relative,
				)

				metadata_broadcast = np.broadcast_to(channel_metadata, (rendered_channel.shape[0], channel_metadata.size))

				rendered_channel = np.concatenate((metadata_broadcast, rendered_channel), axis=-1)
				self.rendered_channels[method].append(rendered_channel)

			self.channel_types[chan["instrumentType"]].append(chan_idx)
			self.bar_indices.append(channel_bars)
			self.rendered_channel_chords.append(rendered_channel_chords)

		for method in self.rendered_channels:
			self.rendered_channels[method] = np.array(self.rendered_channels[method])


		self.bar_indices = np.array(self.bar_indices, dtype=np.uint8)
		self.bar_size = self.bar_indices.shape[-1]
		self.bar_occupancy = self.bar_indices.astype(bool)

	def render(self, method="dense", out=None):
		return SongPermutation(self, tuple(range(len(self.rendered_channels)))).render(method=method, out=out)

	def permutate(self, n_note, n_drum=None):
		"""
		yield song permutations containing n_note note channels and n_drum drum channels.
		this will never return empty songs.
		"""

		note_combos = list(itertools.permutations(self.channel_types["note"], n_note))
		drum_combos = list(itertools.permutations(self.channel_types["drum"], n_drum))

		if len(note_combos) == 0: note_combos.append(tuple())
		if len(drum_combos) == 0: drum_combos.append(tuple())

		for combo in itertools.product(note_combos, drum_combos):
			combo = tuple(itertools.chain.from_iterable(combo))

			combo_bar_occupancy = self.bar_occupancy[combo, ...]
			combo_occupancy = np.logical_and.reduce(combo_bar_occupancy, axis=0)

			if not combo_occupancy.any(): continue # skip the combo if there are no notes.

			yield SongPermutation(self, combo)


class SongPermutation:
	def __init__(self, parent, channel_indices):
		self.parent = parent
		self.channel_indices = channel_indices
	
	def render(self, method="dense", out=None):
		"""
		interleave the channels using numpy.

		(tick, channel_num), interleaved
		"""

		rendered_channel = self.parent.rendered_channels[method]

		shape = (rendered_channel.shape[-2] * len(self.channel_indices), 
			rendered_channel.shape[-1])

		if out is None:
			out = np.empty(shape, dtype=rendered_channel.dtype)
		else:
			assert out.shape == shape, f"out is of invalid shape (expected {shape}, got {out.shape})"

		for chan_idx, parent_chan_idx in enumerate(self.channel_indices):
			out[chan_idx::len(self.channel_indices), ...] = rendered_channel[parent_chan_idx]

		"""
		indices = np.tile(
			np.arange(len(self.channel_indices), dtype=note2vec.itype), 
			(2, rendered_channel.shape[-2])
		).T
		indices[..., 0] = np.repeat(np.arange(
			rendered_channel.shape[-2], dtype=note2vec.itype), 
			len(self.channel_indices)
		)
		"""

		return out



if __name__ == "__main__":
	import matplotlib.pyplot as plt

	with open("parsed_archive/15208.0.json") as f:
		song_json = json.load(f)

		"""
		note: pitches are stored irrespective to key. i.e. 0 will always mean the key.
		"""

		#pprint(song_json)

		song = Song(song_json)

		pprint(song.bar_occupancy)
		pprint(song.channel_types)

		#plt.imshow(song.rendered_channels["sparse"][1].T)
		#plt.imshow(song.tokenizer.lut["sparse"].T)
		#plt.show()

		for perm in song.permutate(n_note=None):
			indices, interleaved = perm.render(method="dense")

			plt.imshow(interleaved[:40].T)
			plt.show()