from typing import Any, Generator
from pydantic import BaseModel
import regex as re


# class SegmentAggregator:
# 	def __init__(
# 		self,
# 		# The limits after which a new chunk is formed
# 		chunk_size_chars=200,
# 		chunk_size_words=200, 
# 		# Chunks will try to overlap by this many segments 
# 		overlap=0,
# 	):
# 		self.threshold_chars: int = chunk_size_chars
# 		self.threshold_words: int = chunk_size_words
# 		self.overlap: int = overlap

# 		self.buffer: list[str] = []

# 		# Tags are addressed on a segment level 
# 		self.tags: list[tuple[int, dict]] = []
# 		self.segment_i: int = 0

# 	def avaiable_segments(self) -> tuple[int, bool]:
# 		""" 
# 		The number of segments that will be pulled into a chunk if a chunk is formed. 
# 		Also a flag for whether a threshold was reached. 
# 		"""

# 		if len(self.buffer) == 0:
# 			return 0, False

# 		# Find the number of segments we can include in this chunk 
# 		# This will be >= 1
# 		segment_count = 1
# 		cur_len_words = len(self.buffer[0].split(" "))
# 		cur_len_chars = len(self.buffer[0])
# 		threshold_reached = False
# 		while segment_count < len(self.buffer): # Keep adding if another is available and will be below thresholds 
# 			next_segment = self.buffer[segment_count]
# 			next_len_words = len(next_segment.split(" "))
# 			next_len_chars = len(next_segment)
# 			if cur_len_words + next_len_words >= self.threshold_words or cur_len_chars >= next_len_chars >= self.threshold_chars:
# 				# Do not add bc we're done here
# 				# print("Threshold would be excceded, stopping")
# 				threshold_reached = True
# 				break
# 			else:
# 				# print(f"Add segment {self.segment_i + segment_count + 1}")
# 				segment_count += 1
# 				cur_len_words += next_len_words
# 				cur_len_chars += next_len_chars
# 		return segment_count, threshold_reached

# 	def _make_group(self) -> tuple[list[str], list[dict]]:
# 		if len(self.buffer) == 0:
# 			# print("buffer empty!")
# 			return [], []

# 		# Take at least one segment, try to overlap with others 
# 		# Could just specify htis in the arguments
# 		segment_count, _ = self.avaiable_segments()

# 		segments = self.buffer[:segment_count]

# 		# Remove at least one segment but try to overlap them if there are more
# 		to_remove = max(1, segment_count - self.overlap)
# 		self.buffer = self.buffer[to_remove:]
# 		self.segment_i += to_remove

# 		# Take applicable tags, filter tags buffer
# 		tags = [tag for i, tag in self.tags]
# 		self.tags = [(i, tag) for i, tag in self.tags if i > self.segment_i]

# 		return segments, tags

# 	def feed(self, segments: list[str], tags: dict) -> Generator[tuple[list[str], list[dict[Any, Any]]], Any, None]:
# 		# Tags are applicable to anything below or equal to the new index
# 		self.buffer += segments
# 		self.tags.append((self.segment_i + len(self.buffer), tags))

# 		while self.avaiable_segments()[1]:
# 			yield self._make_group()

# 	# Forces the builder to use the data in the chunk buffer regardless of the buffer's size
# 	# This should produce only one element
# 	def finish(self) -> tuple[list[str], list[dict]]:
# 		return self._make_group()


def to_sentences(text: str) -> list[str]:
	# Split but also keep the characters 
	# apparently not very common! 

	sentences = []
	lines = re.split(r"((?<=[!.?]\s)\n*)", text)
	# for i, line in enumerate(lines):
	# 	print(f"line '{line.replace("\n", "/n")}'")

	# exit(0)
	return lines


def to_words(line):
	words = re.split(r"(\w+\s+)", line)
	# for w in words:
	# 	print(f"word '{w}'")
	return [w for w in words if w != ""]


# class Chunk(BaseModel):
# 	embedding: list[float]
# 	contents: str


def make_segments_text(text: str, tags: dict) -> list[tuple[str, dict]]:
	# We will use line and column numbers for addressing

	sentences = to_sentences(text)

	segments = []
	line = 0
	column = 0
	for sentence in sentences:
		en_line = line + sentence.count("\n")

		print(f"'{sentence}'")
		
		i = sentence[::-1].rfind("\n")
		# print(f"newline at {i}")
		if i == -1:
			i = 0 
			en_column = column + len(sentence[i:])
		else:
			print("OOo a newline")
			en_column = len(sentence[i:])
		# print(f"len is {len(sentence[i:])} ({len(sentence)})")
		# en_column = column + len(sentence[i:])
		if len(sentence.strip()) != 0:
			segments.append((sentence, {
				**tags,
				"st_line": line,
				"st_column": column,
				"length": len(sentence),
				"en_line": en_line, # Not actually needed bc we split on a newline
				"en_column": en_column,
			}))

		line = en_line
		column = en_column

	for seg, t in segments:
		if len(seg) < 6:
			print(seg)
			print(t)
			print("------")

	return segments
