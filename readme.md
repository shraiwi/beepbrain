# BeepBrain

Yet another music-generating transformer, but it's my baby.

## General Training Methodology

1. Download BeepBox Archive and parse songs.
2. Train embedding model to convert song ticks into 

## Model Architecture

A 100M parameter decoder-only transformer model, with a 512-token context window. I draw upon the 170M parameter GPT-2 model to determine model sizes.

The model is designed to generate its output in an interleaved manner. For example, if it is instructed to generate 3 channels the tokens will have the following pattern: abcabcabc...

The model will receive the following "static" inputs:
- rhythm (one-hot) 5D
- note channel count (one-hot) 10D
- drum channel count (one-hot) 5D




# Old Notes
```
# the task is to extract patterns from songs and "render"
# them into longer sequences

# general structure
"""
ai song structure

beats_per_bar beats per bar
24 parts per beat
2 ticks per part

each beepbox beat has 24 parts, which there are 2 ticks per part
12 ai ticks per beepbox beat

could we turn notes into distance-field structures?
that means that the dot product of two patterns could yield important data.

the goals are to capture:
	- relative positioning data
	- chord similarity
	- pattern data
in a single metric

11 pitches in drum chan, 37 pitches in note 

12 semitones per octave, 3 octaves + 1 note (37 semitones) range per octave.

ok, so:
1. "render" patterns into 96x64 vectors

256d vector represents ~7 patterns. mix all note channels and all drum channels into a single sdf. invert signs of note and drum channels.

consider how addition, multiplication, and dot product can help.

dot should indicate something, and sum of differences should indicate something else.

metadata
	tempo
	key
	rhythm
	bar_count
	beats_per_bar
channel[]
	instrument_type (note|drum)
		[!] all channels after channel[song.pitchChannelCount] are drum channels.
	octave
	pattern[]
		note
			start
			duration
			pitch
	bars[]
		pattern_index


ok, so byte pair tokenise the songs. define special tokens for the following:

other
	<|songend|>

metadata
	<|metadata.start|> M
	<|metadata.end|> m

	<|metadata.tempo|> t
	<|metadata.key|> k
	<|metadata.bars_per_song|> a
	<|metadata.beats_per_bar|> e

channel
	<|channel.start|> C
	<|channel.end|> c

pattern
	<|pattern.start|> P
	<|pattern.end|> p

bar
	<|bar.start|> B
	<|bar.end|> b

note
	start, duration, pitch

for example, here's a simple song. ignore the whitespace. all numbers are encoded as their bytes.
M
	t 120
	k 0
	r 0
	a 
m

C
	n
	3
	P
		30 120 4
		30 20 2
		
	p
	P
		120 1 4
		130 3 5
	p
	B
		1 1 2 3 4 2 1 0 0 1
	b
c
```