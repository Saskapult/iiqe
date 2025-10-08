import litellm


def make_embeddings(data: list[str], embedding_model) -> list[list[float]]:
	result = litellm.embedding(embedding_model, input=data).data
	embeddings = [d.embedding for d in result]
	# model = SentenceTransformer(embedding_model)
	# embeddings = model.encode_document(data)
	return embeddings


# async def make_facts(chunks: list[Chunk]) -> list[str]:
# 	class ExtractFacts(dspy.Signature):
# 		""" Extract facts from the text, please be thorough. """
# 		source_text: str = dspy.InputField()
# 		facts: list[str] = dspy.OutputField()
# 	extract_facts = dspy.Predict(ExtractFacts)

# 	limiter = aiolimiter.AsyncLimiter(3, 1)

# 	async def extract_facts(text: str) -> list[str]:
# 		await limiter.acquire()
# 		return (await extract_facts.acall(source_text=text)).facts

# 	print(f"Extract facts from {len(chunks)} chunks")
# 	tasks = [extract_facts(c.contents) for c in chunks]
# 	results = asyncio.gather(*tasks)

# 	results = [f for group in results for f in group]

# 	# print(f"Running NER")
# 	# nlp = spacy.load("en_core_web_sm")
# 	# # Parallelize? 
# 	# # Use you a map reduce 
# 	# results = {}
# 	# for r in results:
# 	# 	for fact in r:
# 	# 		print(fact, end="")
# 	# 		ents = nlp(fact).ents
# 	# 		print(f" has {len(ents)} entities")
# 	# 		for e in ents:
# 	# 			print(f"- {e.text}")
# 	# 			if e.lower() in 

# 	# Extract facts 
# 	# NER for entities 
# 	# fact -> list[entity]

# 	# MINE score wrt graph size 
	
# 	return results

