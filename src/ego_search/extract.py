# from chunk import Chunk
# from litellm import aembedding
import litellm
# import networkx as nx
# import aiolimiter
# import dspy
# import asyncio
# from sentence_transformers import SentenceTransformer


# def append_to_kg_nx(g: nx.MultiDiGraph, graph: Graph, tags: list[dict]):
# 	def find_node_by_property_id(g, id) -> int | None:
# 		found = [(n, d) for n, d in g.nodes.items() if d["properties"]["id"] == id]
# 		assert len(found) <= 1
# 		return found[0][0] if len(found) > 0 else None
	
# 	assert all([isinstance(k, int) for k in g.nodes.keys()])
# 	largest_idx = max(g.nodes.keys()) if len(g.nodes) > 0 else 0
# 	for i, entity in enumerate(graph.entities):
# 		# print(f"Write entity {i+1}/{len(graph.entities)} ({entity})")
# 		e_name = entity
# 		e_id = e_name.upper()

# 		if idx := find_node_by_property_id(g, e_id):
# 			idx = idx
# 		else:
# 			idx = largest_idx+1
# 			g.add_node(idx, properties={
# 				"id": e_id, 
# 				"name": e_name,
# 				"references": [],
# 				"tags": [],
# 			})
# 			largest_idx = idx
		
# 		g.nodes[idx]["properties"]["references"].append(graph.chunk_id)
# 		g.nodes[idx]["properties"]["tags"] += tags

# 	for i, (a, r, b) in enumerate(graph.relations):
# 		# print(f"Write relation {i+1}/{len(graph.relations)} ({a} ~ {r} ~ {b})")
# 		r_name = to_db_repr(r)
# 		a_id = to_db_repr(a).upper()
# 		b_id = to_db_repr(b).upper()
# 		r_id = r_name.upper()

# 		a_idx = find_node_by_property_id(g, a_id)
# 		assert not (a_idx is None)
# 		b_idx = find_node_by_property_id(g, b_id)
# 		assert not (b_idx is None)

# 		if not g.has_edge(a_idx, b_idx, r_id):
# 			g.add_edge(a_idx, b_idx, r_id, properties={
# 				"id": r_id,
# 				"name": r_name,
# 				"references": [],
# 				"tags": [],
# 			})
		
# 		g[a_idx][b_idx][r_id]["properties"]["references"].append(graph.chunk_id)
# 		g[a_idx][b_idx][r_id]["properties"]["tags"] += tags



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

