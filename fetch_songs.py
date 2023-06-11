import concurrent.futures, requests, time, json, re, ast, os
from pathlib import Path
from operator import itemgetter

CONNECTIONS = 8
TIMEOUT = 5
ALLOWED_FORK = "www.beepbox.co"

print("fetching beepbox archive webpage...")


with requests.get(r"https://twitter-archive.beepbox.co/") as req:
    archive_html = req.content.decode("utf-8")

"""
with open("archive.html", "r") as f_archive:
    archive_html = f_archive.read()
"""

def extract_value(key, html):
    expr = f"(?s){key}\\s*=\\s*(\\[.+?\\]);?\\n"
    match = re.search(expr, html).group(1)
    #print(match)
    try: return ast.literal_eval(match)
    except: return json.loads(match)

all_domains = extract_value("allDomains", archive_html)
all_dates = extract_value("allDates", archive_html)
short_tweets = extract_value("shortTweets", archive_html)

allowed_fork_idx = all_domains.index(ALLOWED_FORK)

filtered_songs = [] # url, song_idxs
num_songs = 0

for tweet_idx, (name_idx, date_idx, rt_count, like_count, song_fork_idxs) in enumerate(short_tweets):
    url = f"http://twitter-archive.beepbox.co/tweets/{tweet_idx}.json"

    song_idxs = tuple(i for i, fork_idx in enumerate(song_fork_idxs) if fork_idx == allowed_fork_idx)
    num_songs += len(song_idxs)

    if song_idxs:
        filtered_songs.append((url, song_idxs))

# remake, remade, cover, from, transcribe, transcribed, copy, 
#filtered_songs = filtered_songs[:10]

print(f"found {num_songs} songs from {ALLOWED_FORK}")

archive_path = Path("archive")
filtered_songs_path = Path("filtered_songs.json")
urls_path = Path("urls.txt")

archive_path.mkdir(exist_ok=True)

with filtered_songs_path.open("w+") as f_filtered, urls_path.open("w+") as f_urls:
    json.dump(filtered_songs, f_filtered)
    f_urls.writelines(f"{song[0]}\n" for song in filtered_songs)

DL_CMD = f"cd {archive_path.resolve()} && cat {urls_path.resolve()} | pv -p -t -e -s $(stat -c %s {urls_path.resolve()}) | xargs -n20 -P8 wget -c -nc -r -nH -nd -nv --"

print(f"run the following command to download the songs:")
print(f"{DL_CMD}")