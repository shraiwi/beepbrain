import json
import pandas as pd
import numpy as np
from pathlib import Path
from pprint import pprint

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
	
	assert out.shape == chords_shape, f"out is of invalid size (expected {chords_shape}, got {out.size})"

	for note_start, note_end, note_pitches in pat:
		note_start_ticks, note_end_ticks = note_start // parts_per_tick, note_end // parts_per_tick

		print(note_start_ticks, note_end_ticks, note_pitches)

		out[note_start_ticks:note_end_ticks, :len(note_pitches)] = note_pitches

	return out

class Song:
	def __init__(self, song_data, tokenizer=NoteTokenizer(), ticks_per_pattern=48):

		self.ticks_per_pattern = ticks_per_pattern
		self.tokenizer = tokenizer

		self.parts_per_tick = song_data["beatsPerBar"] * 24 // ticks_per_pattern;

		self.channels = []
		self.channel_types = {
			"note": [],
			"drum": [],
		}

		# render channels into chords
		for chan in song_data["channels"]:
			channel_chords = np.full(
				(len(chan["patterns"]), self.ticks_per_pattern, self.tokenizer.chord_size), 
				-1, 
				dtype=note2vec.itype)

			for pat_idx, pat in enumerate(chan["patterns"]):
				pat2chords(pat, self.parts_per_tick, out=channel_chords[pat_idx], tokenizer=tokenizer)

			self.channel_types[chan["instrumentType"]].append(len(self.channels))
			self.channels.append(channel_chords)


class SongPermutation:
	def __init__(self, parent, channel_indices):
		self.parent = parent
		self.channel_indices = channel_indices

	def __iter__(self):
		"""
		interleave the channels and yield tokens
		"""

		for token_idx, (chan_idx, chan) in enumerate(itertools.repeat(enumerate(self.channel_indices), self.parent.ticks_per_pattern)):
			pass

			

with open("parsed-archive/0.0.json") as f:
	song_json = json.load(f)

	"""
	note: pitches are stored irrespective to key. i.e. 0 will always mean the key.
	"""

	pprint(song_json)

	song = Song(song_json)

	pprint([*map(len, song.channels)])