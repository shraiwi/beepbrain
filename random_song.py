import json, random, webbrowser
from pathlib import Path

json_paths = tuple(Path("parsed-archive").glob("*.json"))

random_song = random.choice(json_paths)

with random_song.open() as f:

	print(random_song)

	song_hash = json.load(f)["hash"]

	webbrowser.open(f"www.beepbox.co/{song_hash}")