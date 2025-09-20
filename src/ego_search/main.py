import argparse
import asyncio
import json
from pathlib import Path
import matplotlib.pyplot as plt

from .prompt import ego_search, format_results, format_segment_text, prompt
from .chunk import make_segments_text
from .extract import make_embeddings
from .settings import Settings


def cli_ingest():
	parser = argparse.ArgumentParser()
	parser.add_argument("filename", nargs="+")
	parser.add_argument("--dataset", "-d", required=True)
	args = parser.parse_args()

	settings = Settings()

	all_segments = []
	for f in args.filename:
		f = Path(f)
		print(f"Reading {f}")
		with open(f, "r") as fp:
			data = fp.read()
		
		print("Splitting...")
		segments = make_segments_text(data, {"document": str(f)})

		print("Aggregate")
		all_segments += segments

	facts, tags = zip(*all_segments)

	# for s in [format_segment_text(f, t) for f, t in all_segments]:
	# 	print(s)
	# exit(0)

	
	print("Process embeddings")
	embeddings = make_embeddings(facts, settings.embedding_model)
	
	out_file = Path(settings.data_directory) / f"{args.dataset}.json"
	print(f"Writing to {out_file}")
	dataset = {
		"facts": facts,
		"tags": tags,
		"embeddings": embeddings,
		"embedding_model": settings.embedding_model,
	}
	
	out_file.parent.mkdir(exist_ok=True)
	with open(out_file, "w") as fp:
		json.dump(dataset, fp, indent=2)
	
	print("Done!")


def cli_prompt():
	parser = argparse.ArgumentParser()
	parser.add_argument("prompt")
	parser.add_argument("--dataset", "-d", required=True)
	args = parser.parse_args()

	settings = Settings()

	in_file = Path(settings.data_directory) / f"{args.dataset}.json"	
	print(f"Reading from {in_file}")

	with open(in_file, "r") as fp:
		dataset = json.load(fp)
	
	sts, ens, scrs = ego_search(args.prompt, dataset["facts"], dataset["embedding_model"])
	r = format_results(sts, ens, scrs, dataset["facts"], dataset["tags"])
	plt.show()
	plt.clf()

	print(f"Narrow on '{r[0]}'")
	segments = r[0].split()
	print("made", segments, f"({len(segments)})")
	sts, ens, scrs = ego_search(args.prompt, segments, dataset["embedding_model"], 6)
	format_results(sts, ens, scrs, segments, [])
	plt.show()
