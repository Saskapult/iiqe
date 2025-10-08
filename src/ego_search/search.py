from pathlib import Path
import dspy
import litellm
import numpy as np
from scipy.signal import chirp, find_peaks, peak_widths, find_peaks_cwt
import matplotlib.pyplot as plt

from ego_search.extract import make_embeddings


# def fuzzy_subset_match(statements: list[str], filtered: list[str]) -> list[int]:
# 	filtered_sources = []
# 	for f in filtered:
# 		sims = [fuzz.ratio(f, s) for s in statements]
# 		i = np.argmax(sims)
# 		if f != statements[i]:
# 			logger.warning(f"Filtered statement '{f}' has no exact match but is most similar to input statement '{statements[i]}', using that to prevent source loss")
# 		filtered_sources.append(i)
# 	return filtered_sources


def format_segment_text(statement, tags) -> str:
	tags_str = f"{Path(tags["document"]).name} line {tags["st_line"]+1} columns {tags["st_column"]+1}-{tags["en_column"]+1}"
	return f"{statement.strip()} - {tags_str}"


def smooth_it(data, k: int = 2):
	smooth = []
	for i in range(0, len(data)):
		st = max(0, i - k)
		en = min(i + k, len(data))
		smooth.append(np.mean(data[st:en]))
	return np.array(smooth)


def make_overlapping(segments: list[str], overlap_by: int = 2) -> list[list[str]]:
	overlaps = []
	for i in range(0, len(segments)-overlap_by):
		overlaps.append(segments[i:i+overlap_by])
	return overlaps


# https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
def find_runs(x):
	"""Find runs of consecutive items in an array."""

	# ensure array
	x = np.asanyarray(x)
	if x.ndim != 1:
		raise ValueError('only 1D array supported')
	n = x.shape[0]

	# handle empty array
	if n == 0:
		return np.array([]), np.array([]), np.array([])

	else:
		# find run starts
		loc_run_start = np.empty(n, dtype=bool)
		loc_run_start[0] = True
		np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
		run_starts = np.nonzero(loc_run_start)[0]

		# find run values
		run_values = x[loc_run_start]

		# find run lengths
		run_lengths = np.diff(np.append(run_starts, n))

		return run_values, run_starts, run_lengths


def ego_search(
	text: str,
	segments: list[str],
	embedding_model: str,
	overlap: int = 4,
	strictness: float = 0.65,
):
	# Need overlap for individual words 
	overlapping = make_overlapping(segments, overlap_by=overlap)
	overlapping = ["".join(t) for t in overlapping]

	print("Make embeddings...")
	embeddings = make_embeddings([text] + overlapping, embedding_model)
	text_embedding = np.array(embeddings[0])
	embeddings = np.array([np.array(e) for e in embeddings[1:]])

	# Get sims 
	print("Compute similarities...")
	sims = np.abs(np.dot(embeddings, text_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(text_embedding)))
	# plt.plot(sims)

	# Re-expand into score curve (using rolling average)
	scores = np.zeros(len(segments))
	for i, sim in enumerate(sims):
		for j in range(i, i + overlap):
			scores[j] += sim
	scores = scores / overlap
	plt.plot(scores, label="score")

	# plt.hlines(np.mean(scores), 0, len(segments))
	# plt.hlines(np.percentile(scores, 75), 0, len(segments))
	# What if select largest contiguous region over mean? 
	# Or over mean with highest peak 

	# Find regions above some quantile 
	thres = np.quantile(scores, strictness)
	vs, sts, rlen = find_runs(scores >= thres)
	sts = np.array([i for i, v in zip(sts, vs) if v])
	rlen = np.array([i for i, v in zip(rlen, vs) if v])
	ens = sts + rlen
	print(sts)
	print(ens)
	plt.hlines(np.full(sts.shape, thres), sts, ens, color="C2", label="regions")

	# Score them (by highest value)
	r_scores = np.array([s + np.argmax(scores[s:e]) for s, e in zip(sts, ens)])
	plt.plot(r_scores, scores[r_scores], "x", label="peak value")
	r_scores = scores[r_scores]

	plt.legend() 

	# # Find peaks 
	# peaks, _ = find_peaks(scores, height=0.0, width=0.0)
	# heights, wmin, wmax = peak_widths(scores, peaks, rel_height=prominence)[1:]
	# plt.plot(peaks, scores[peaks], "x")
	# plt.hlines(heights, wmin, wmax)

	# # Map back 
	# sts = np.floor(wmin).astype(int)
	# ens = np.ceil(wmax).astype(int)

	order = np.argsort(r_scores)[::-1]
	
	return sts[order], ens[order], r_scores[order]


def format_results(
	sts,
	ens,
	scores,
	segments: list[str],
	tags: list[dict],
):
	print(f"Found {len(scores)} sections")
	r = []
	for st, en, sc in zip(sts, ens, scores):
		print(f" - {st}:{en} ({sc})")
		g = " ".join(segments[st:en])
		print("\t", g)
		r.append(g)
	return r



def prompt(
	text: str, 
	facts: list[str],
	tags: list[dict],
	# Make optional?
	embeddings: list[list[float]],
	embedding_model: str,
	query_model: str,
): 
	embeddings = np.array([np.array(e) for e in embeddings])
	text_embedding = litellm.embedding(embedding_model, input=[text]).data[0].embedding

	print("Do the searchy")
	cosine = np.abs(np.dot(embeddings, text_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(text_embedding)))
	sims = np.argsort(cosine)[::-1]

	print("Locate peaks")
	k = 2
	smoothed = smooth_it(cosine, k)
	peaks, _ = find_peaks(smoothed, height=0.0, width=0.0)
	results_half = peak_widths(smoothed, peaks, rel_height=0.5)
	results_full = peak_widths(smoothed, peaks, rel_height=1)
	# heights = smoothed[peaks]
	heights, wmin, wmax = results_full[1:]
	print("peaks", peaks)
	print("rh", results_half)
	print("heights", heights)
	# print("widths", widths)

	# peaks2 = find_peaks_cwt(cosine, np.arange(1,10))

	print("Tie to text")
	print(f"Base {len(cosine)}")
	print(f"Smoothed {len(smoothed)}")
	# Round to index ranges
	sts = np.floor(wmin).astype(int)
	ens = np.ceil(wmax).astype(int)

	order = np.argsort(heights)[::-1]
	print(sts)
	print(ens)

	# If another thing overlaps it, it should be boosted with that score
	adj_heights = np.array(heights)
	for i, (st, en, h) in enumerate(zip(sts, ens, heights)):
		for j, (ost, oen, oh) in enumerate(zip(sts, ens, heights)):
			if i != j and st >= ost and en <= oen:
				adj_heights[i] += np.abs(oh - h)

	# # Bar plot for these? 
	# scores = np.zeros(len(facts))

	order = np.argsort(adj_heights)[::-1]
	print(sts)
	print(ens)

	print(f"Found {len(order)} sections")
	for i in order:
		print(f" - {sts[i]}:{ens[i]} ({adj_heights[i]})")
		for j in range(sts[i], ens[i]):
			print("\t", facts[j])
	
	# print("scores", scores)
	
	# new_order = np.argsort(scores)[::-1]

	# print(f"Found {len(new_order)} sections")
	# for i in new_order:
	# 	print(f" - {sts[i]}:{ens[i]} ({scores[i]})")
	# 	for j in range(sts[i], ens[i]):
	# 		print("\t", facts[j])

	# segments = "The yams are nicely textured and add a lot to the experience. I like to make a potato and lentil stew in the fall.".split()
	# ego_search(text, segments, [], embedding_model)

	# Similarities 
	plt.plot(cosine)
	# Smoothed 
	plt.plot(smoothed)
	# plt.plot(peaks, smoothed[peaks], "x")
	plt.plot(peaks, heights, "x")
	# plt.plot(peaks2, cosine[peaks2], "x")
	plt.hlines(heights, wmin, wmax)
	plt.hlines(adj_heights, wmin, wmax)
	plt.plot(peaks, heights, "x")
	# plt.bar(np.array([i for i in range(0, len(facts))]), scores, align="edge")
	# plt.hlines(*results_half[1:], color="C2")
	# plt.hlines(*results_full[1:], color="C3")
	plt.show()
	exit(0)

	



	k = 4
	candidates = sims[:k]

	extracts = [format_segment_text(facts[i], tags[i]) for i in candidates]

	class NavigationPrompt(dspy.Signature):
		""" 
		Please help me locate information related to my query. 
		Here are some things I have written and where I wrote about them. 
		Direct me to where I can find relevant information. 
		"""
		extracts: list[str] = dspy.InputField()
		response: str = dspy.OutputField()
	direction = dspy.Predict(NavigationPrompt)

	print("Make some direction")
	with dspy.context(lm=dspy.LM(query_model)):
		response = direction(extracts=extracts).response

	print("===")
	print(response)
	print("---")
	for e in extracts:
		print(f"  {e}")
