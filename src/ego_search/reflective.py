from __future__ import annotations

import litellm
from sqlmodel import select
from .models import (
	PromptResponse,
	Citation,
	CitationSubgraphResult,
	CitationLocation,
)
import dspy
from fuzzywuzzy import fuzz
import numpy as np
import asyncio



class ReflectiveStatement:
	relationship: tuple[str, str, str]
	source_chunks: list[UUID]


class ReflectiveResponse:
	response: str
	statements: list[ReflectiveStatement]


async def get_relevant_subjects(question: str, model: str) -> list[str]:
	class RetrievalPlanSignature(dspy.Signature):
		"""
		I am planning to retrieve information in order to answer a question.
		What subjects should I research in answering this question?
		"""
		question: str = dspy.InputField()
		subjects: list[str] = dspy.OutputField()
	retrieval_plan = dspy.Predict(RetrievalPlanSignature)
	with dspy.context(lm=dspy.LM(model)):
		initial_subjects = (await retrieval_plan.acall(question=question)).subjects
	return initial_subjects


async def k_most_relevant_entities(text: str, embedding_model: str, k: int, project_id: UUID) -> list[str]:
	# Get embedding 
	embedding = (await litellm.aembedding(embedding_model, input=[text])).data[0].embedding
	
	# Query database 
	async with session() as db:
		res = await db.execute(
			select(NodeEmbedding)
			.where(NodeEmbedding.project_id == project_id)
			.order_by(NodeEmbedding.embedding.cosine_distance(embedding)
		).limit(k))
		nearest = [e.node_key for e in res.scalars().all()]
	if len(nearest) == 0:
		logger.error(f"No embeddings found for model '{embedding_model}'")
	return nearest


async def map_to_relevant_entities(entity_set: list[str], embedding_model: str, project_id: UUID) -> list[str]:
	# Simply map each entity to the four most relevant entities in the kg
	# Could try having a threshold or something more complicated here 
	return list(set([e for subject in entity_set for e in await k_most_relevant_entities(subject, embedding_model, 4, project_id)]))


async def _get_project_source_ids(project_id: UUID) -> list[str]:
    """
    Authoritative scoping: fetch ingestion source IDs for the project from SQL.
    Returns canonical UUID strings to compare against r.source_ids in Memgraph.
    """
    async with session() as s:
        res = await s.execute(
            text("SELECT id::text FROM ingestion_sources WHERE project_id = :pid"),
            {"pid": str(project_id)},
        )
        return [row[0] for row in res.fetchall()]


async def neighbour_based_subgraph(entities: list[str], project_id: UUID) -> tuple[list[tuple[str, str, str]], list[list[UUID]]]:
	read = await run_read(
		"""
		MATCH (a:Entity)-[r:Relation]-(b:Entity)
		WHERE a.key IN $keys AND any(s IN r.source_ids WHERE s IN $srcs)
		RETURN DISTINCT
			r.key AS rkey, r.predicate_key AS pkey, coalesce(r.predicate_name, r.name) AS rname,
			a.key AS skey, a.name AS sname,
			b.key AS ekey, b.name AS oname,
			r.source_ids AS source_ids, r.source_chunks AS source_chunks
		""",
		{"keys": [default_keygen(e) for e in entities], "srcs": await _get_project_source_ids(project_id)},
	)

	# The triplet data structure allows for only one source id for some reason 
	# I will not use it for this so that I do not have to lack functionality
	# Once it has that then we can use it here 

	# Dedup 
	# subject is start 
	# predicate is r or predicate 
	# object is end or object 
	# TODO: make terminology consistent (end is not "end" in all grammar systems)
	relationships = [(row["sname"], row["rname"], row["oname"]) for row in read]
	references = [row["source_chunks"] for row in read]
	relationships_references = {rel: ref for rel, ref in zip(relationships, references)}
	relationships = list(set(relationships))
	references = [relationships_references[rel] for rel in relationships]

	return relationships, references


def fuzzy_subset_match(statements: list[str], filtered: list[str]) -> list[int]:
	""" Maps some statements to their corresponding sources using string similarity. """

	filtered_sources = []
	for f in filtered:
		sims = [fuzz.ratio(f, s) for s in statements]
		i = np.argmax(sims)
		if f != statements[i]:
			logger.warning(f"Filtered statement '{f}' has no exact match but is most similar to input statement '{statements[i]}', using that to prevent source loss")
		filtered_sources.append(i)
	return filtered_sources


async def relevancy_filter(
	question: str, 
	kb: list[tuple[str, str, str]], 
	sources: list[list[UUID]],
	model: str,
) -> tuple[list[tuple[str, str, str]], list[list[UUID]]]:
	class SelectionPlanSignature(dspy.Signature):
		""" 
		I have retrieved some relationships from a knowledge graph that I might use to answer a question. 
		Which of these are relevant in answering the question?
		"""
		question: str = dspy.InputField()
		available_relationships: list[str] = dspy.InputField()
		relevant_relationships: list[str] = dspy.OutputField()
	selection_plan = dspy.Predict(SelectionPlanSignature)

	kb_str = [" ".join(t) for t in kb]
	with dspy.context(lm=dspy.LM(model)):
		result = await selection_plan.acall(question=question, available_relationships=kb_str)
	indices = fuzzy_subset_match(kb_str, result.relevant_relationships)
	return [kb[i] for i in indices], [sources[i] for i in indices]


async def can_resolve(prompt: str, kb: list[tuple[str, str, str]], model: str):
	class FinishingPlanSignature(dspy.Signature):
		""" 
		I have retrieved some relationships from a knowledge graph in order to answer a question. 
		Do I have enough information to provide an accurate and complete answer to the question?
		"""
		question: str = dspy.InputField()
		retrieved_relationships: list[tuple[str, str, str]] = dspy.InputField()
		enough_information: bool = dspy.OutputField()
	finishing_plan = dspy.Predict(FinishingPlanSignature)
	with dspy.context(lm=dspy.LM(model)):
		return (await finishing_plan.acall(question=prompt, retrieved_relationships=kb)).enough_information


async def get_elaboration_subjects(prompt: str, kb: list[tuple[str, str, str]], model: str) -> list[str]:
	class ElaborationPlanSignature(dspy.Signature):
		""" 
		I have retrieved some relationships from a knowledge graph in order to answer a question. 
		What further subjects would be relevant to answering this question?
		"""
		question: str = dspy.InputField()
		retrieved_subjects: list[tuple[str, str, str]] = dspy.InputField()
		subjects: list[str] = dspy.OutputField()
	elaboration_plan = dspy.Predict(ElaborationPlanSignature)
	with dspy.context(lm=dspy.LM(model)):
		return (await elaboration_plan.acall(question=prompt, retrieved_subjects=kb)).subjects


async def citation_str(relationship: tuple[str, str, str], references: list[UUID], chunk_too=False) -> str:
	async with session() as db:
		chunks = await get_chunks_by_ids(db, references)
	# It's weird that chunks are 1:1 with a page number 
	# That's different than how I did it 
	# They should be able to span multiple pages 
	# I guess that's okay for now
	chunks_pages = sorted(list(set([p for c in chunks for p in [c.page_number] if not p is None])))
	
	pages_str = f"page{"s" if len(chunks_pages) > 1 else ""} {", ".join([str(p) for p in chunks_pages])}"
	chunks_str = f"chunk{"s" if len(references) > 1 else ""} {", ".join([str(r) for r in references])}"
	
	return " ".join(relationship) + f" - {pages_str}" + f" ({chunks_str})" if chunk_too else ""


async def create_response(prompt: str, triples: list[tuple[str, str, str]], references: list[list[UUID]], model: str) -> tuple[str, list[str]]:
	class AnswerResultSignature(dspy.Signature):
		""" 
		You know nothing about anything and rely only on the information explicitly provided to you.
		Using the provided information, answer the question and direct the user to the relevant sources so that they may verify your answer. 
		"""
		question: str = dspy.InputField()
		# I like cats - page 42
		knowledge: list[str] = dspy.InputField()
		answer: str = dspy.OutputField()
	answer_result = dspy.Predict(AnswerResultSignature)
	
	facts = [await citation_str(rel, ref) for rel, ref in zip(triples, references)]

	with dspy.context(lm=dspy.LM(model)):
		answer = (await answer_result.acall(question=prompt, knowledge=facts)).answer
	return answer, facts


async def validate_answer_statements(response: str, kb: list[tuple[str, str, str]], model: str, rate_limit: int = 3) -> tuple[list[str], list[str]]:
	class ExtractSignature(dspy.Signature):
		""" List the facts presented in the text. """
		text: str = dspy.InputField()
		statements: list[str] = dspy.OutputField()
	
	class ValidationSignature(dspy.Signature):
		""" Is the given statement supported by the fact list? """
		statement: list[str] = dspy.InputField()
		fact_list: list[str] = dspy.InputField()
		statement_is_in_list: bool = dspy.OutputField()
	
	extract = dspy.Predict(ExtractSignature)
	validate = dspy.Predict(ValidationSignature)

	limiter = AsyncLimiter(rate_limit, time_period=1)
	
	async def is_supported(statement: str, kb: list[str]) -> bool:
		await limiter.acquire()
		return (await validate.acall(statement=statement, fact_list=kb)).statement_is_in_list

	supported = []
	unsupported = []
	
	with dspy.context(lm=dspy.LM(model)):
		answer_statements = (await extract.acall(text=response)).statements

		tasks = [is_supported(s, [" ".join(triple) for triple in kb]) for s in answer_statements]
		statement_is_supported = await asyncio.gather(*tasks)

		for statement, s in zip(answer_statements, statement_is_supported):
			if s:
				supported.append(statement)
			else:
				unsupported.append(statement)

	return supported, unsupported


async def whittle_response(response: str, context: list[str], model: str, debug_response: list[tuple[str, list[str]]] = []) -> tuple[str, list[str]]:

	class WhittleSignature(dspy.Signature):
		""" Rewrite the given answer to not include the invalid statements. If the answer cannot be rewritten without the invalid statements, leave the new answer empty. """
		old_answer: str = dspy.InputField()
		invalid_statements: list[str] = dspy.InputField(desc="Statements that should not be included in the rewritten answer")
		rewritten_answer: str = dspy.OutputField()
	whittle = dspy.Predict(WhittleSignature)

	async def checks_or_error(text, kb, model):
		try:
			checks_results = await validate_answer_statements(text, kb, model)
		except Exception as e:
			checks_results = e
		return checks_results

	unsubstatiated = []
	while True:
		checks_results = await checks_or_error(response, context, model)
		if isinstance(checks_results, Exception):
			logger.error("Error while validating response for whittling")
			raise checks_results
		supported, unsupported = checks_results
		if len(unsupported) == 0: 
			break
		logger.debug(f"Whittle response to not include {unsupported}")
		unsubstatiated += unsupported
		with dspy.context(lm=dspy.LM(model)):
			response = (await whittle.acall(old_answer=response, invalid_statements=unsupported)).rewritten_answer
		debug_response.append((response, unsupported))
		logger.debug(f"Response is now '{response}'")
		if response.strip() == "":
			logger.debug(f"Whittled to nothing, terminating")
			response = "Could not construct a well-sourced answer to the given question, please try again."
			break
	
	return response, unsubstatiated


async def reduce_references(response: str, kb: list[tuple[str, str, str]], references: list[list[UUID]], model: str, rate_limit: int = 3) -> tuple[list[tuple[str, str, str]], list[list[UUID]], list[tuple[str, str, str]], list[list[UUID]]]:
	class SourceReductionSignature(dspy.Signature):
		""" Is this information included in the text? """
		information: str = dspy.InputField()
		text: str = dspy.InputField()
		included: bool = dspy.OutputField()
	source_reduction = dspy.Predict(SourceReductionSignature)
	new_relationships = []
	new_sources = []
	discard_relationships = []
	discard_sources = []

	limiter = AsyncLimiter(rate_limit, time_period=1)
	
	async def is_relevant(information: str, response: str) -> bool:
		await limiter.acquire()
		return (await source_reduction.acall(information=information, text=response)).included
	
	with dspy.context(lm=dspy.LM(model)):
		tasks = [is_relevant(" ".join(t), response) for t in kb]
		relevant = await asyncio.gather(*tasks)

		for t, ref, relevant in zip(kb, references, relevant):
			if relevant:
				new_relationships.append(t)
				new_sources.append(ref)
			else:
				logger.debug(f"Statement '{" ".join(t)}' is irrelevant to the response")
				discard_relationships.append(t)
				discard_sources.append(ref)
	logger.debug(f"Reduced to {len(new_relationships)} sources (previously {len(kb)})")
	return new_relationships, new_sources, discard_relationships, discard_sources


async def concept_subgraph(entities: list[str], project_id: UUID) -> CitationSubgraphResult:
	targets: set[str] = set([default_keygen(e) for e in entities])
	if not targets:
		return CitationSubgraphResult(node_keys=[], edge_keys=[])
	
	params = {
		"pid": str(project_id),
		"targets": list(targets),
		"max_hops": 20,
	}

	# Nodes on any BFS-shortest path root -> target (typed edges only)
	node_rows = await run_read(
		f"""
		UNWIND $targets AS tkey
		MATCH (root:CGRoot {{project_id:$pid}}), (e:Entity {{key:tkey}})
		MATCH p = (root)-[:HAS_COMMUNITY|:MEMBER_OF *BFS..{params["max_hops"]}]-(e)
		UNWIND nodes(p) AS n
		RETURN DISTINCT n.key AS node_key
		""",
		params,
	)

	# Edges on those paths; synthesize key if missing
	edge_rows = await run_read(
		f"""
		UNWIND $targets AS tkey
		MATCH (root:CGRoot {{project_id:$pid}}), (e:Entity {{key:tkey}})
		MATCH p = (root)-[rels:HAS_COMMUNITY|:MEMBER_OF *BFS..{params["max_hops"]}]-(e)
		UNWIND rels AS r
		WITH r, startNode(r) AS a, endNode(r) AS b, type(r) AS t
		RETURN DISTINCT coalesce(r.key, a.key + '--' + t + '--' + b.key) AS edge_key
		""",
		params,
	)

	node_keys = [row["node_key"] for row in node_rows]
	edge_keys = [row["edge_key"] for row in edge_rows]

	logger.debug(f"[subgrapher]: (computed) node keys{node_keys}\nedge keys: {edge_keys}")
	return CitationSubgraphResult(
		node_keys=node_keys,
		edge_keys=edge_keys,
	)


async def prompt_reflective(prompt_model: str, embedding_model: str, project_id: UUID, prompt: str, max_retries=5) -> tuple[PromptResponse, dict]:
	logger.debug(f"Received prompt request for project_id={project_id} with prompt='{prompt}'")

	debug_items = {}
	# Add args to debug items
	debug_items["prompt_model"] = prompt_model
	debug_items["embedding_model"] = embedding_model
	debug_items["prompt"] = prompt
	debug_items["project_id"] = project_id
	debug_items["max_retries"] = max_retries

	# Extract subjects to research  
	initial_subjects = await get_relevant_subjects(prompt, prompt_model)
	logger.debug(f"Found {len(initial_subjects)} research subjects: {initial_subjects}")
	
	# Subject entities 
	subject_entities = await map_to_relevant_entities(initial_subjects, embedding_model, project_id)
	logger.debug(f"Mapped {len(subject_entities)} subject entities: {subject_entities}")

	# Question entities 
	prompt_entities = await map_to_relevant_entities([prompt], embedding_model, project_id)
	logger.debug(f"Mapped {len(prompt_entities)} prompt entities: {prompt_entities}")

	# Add entity extraction? 

	# Aggregate those 
	seed_entities = list(set(subject_entities + prompt_entities))
	logger.debug(f"Aggregated to {len(seed_entities)} seed entities: {seed_entities}")

	# We have up to max_retries rounds of discoveries 
	debug_items["rounds"] = []
	debug_items["rounds"].append({})
	round = debug_items["rounds"][len(debug_items["rounds"])-1]

	# This first one has some special fields 
	round["subject_entities"] = subject_entities
	round["prompt_entities"] = prompt_entities
	# But its fields represent a superset of those in subsequent rounds 
	round["subjects"] = initial_subjects
	round["seed_entities"] = seed_entities

	# Get neighbours 
	relationships, references = await neighbour_based_subgraph(seed_entities, project_id)
	logger.debug(f"Found {len(relationships)} statements: {relationships}")
	round["relationships_kg"] = relationships

	# Filter 
	relationships, references = await relevancy_filter(prompt, relationships, references, prompt_model)
	logger.debug(f"Relevancy filtered to {len(relationships)} statements: {relationships}")
	round["relationships_filtered"] = relationships
	round["relationships_new"] = relationships
	round["relationships_sum"] = relationships

	debug_items["iter_limit_exceeded"] = False

	kb = relationships
	kb_references = references
	i = 0
	while not await can_resolve(prompt, kb, prompt_model):
		logger.debug(f"Not enough information to generate response!")
		if not (i < max_retries):
			logger.debug(f"Iteration limit exceeded, attempting response anyway")
			debug_items["iter_limit_exceeded"] = True
			break
		i += 1

		debug_items["rounds"].append({})
		round = debug_items["rounds"][len(debug_items["rounds"])-1]

		# Extract subjects to research  
		elaboration_subjects = await get_elaboration_subjects(prompt, kb, prompt_model)
		logger.debug(f"Found {len(elaboration_subjects)} research subjects: {elaboration_subjects}")
		round["subjects"] = elaboration_subjects

		# Map to similar entities 
		seed_entities = await map_to_relevant_entities(elaboration_subjects, embedding_model, project_id)
		logger.debug(f"Mapped {len(seed_entities)} seed entities: {seed_entities}")
		round["seed_entities"] = seed_entities

		# Get neighbours 
		relationships, references = await neighbour_based_subgraph(seed_entities, project_id)
		logger.debug(f"Found {len(relationships)} statements: {relationships}")
		round["relationships_kg"] = relationships

		# Filter 
		relationships, references = await relevancy_filter(prompt, relationships, references, prompt_model)
		logger.debug(f"Relevancy filtered to {len(relationships)} statements: {relationships}")
		round["relationships_filtered"] = relationships

		# Merge 
		new_indices = [i for i, r in enumerate(relationships) if not r in kb]
		new_relationships = [relationships[i] for i in new_indices]
		new_references = [references[i] for i in new_indices]
		logger.debug(f"Found {len(new_indices)} new statements: {new_relationships}")
		kb += new_relationships
		kb_references += new_references	
		round["relationships_new"] = new_relationships
		round["relationships_sum"] = kb
	
	# Generate text for a response 
	logger.debug("Make response")
	response, context = await create_response(prompt, kb, kb_references, prompt_model)
	# Record (response, removed info)
	debug_items["response"] = [(response, [])]

	# Remove unsubstatiated parts of the response 
	logger.debug("Whittle response")
	response, unsubstatiated = await whittle_response(response, context, prompt_model, debug_items["response"])

	# Could retry if failed 
	# Maybe pull full chunks and retry 

	# Filter unused kb information
	logger.debug("Reduce references")
	kb, kb_references, d_kb, d_refs = await reduce_references(response, kb, kb_references, prompt_model)
	debug_items["references"] = kb
	debug_items["references_sources"] = kb_references	
	debug_items["references_unused"] = d_kb
	debug_items["references_sources_unused"] = d_refs

	# References -> citations 
	logger.debug("Convert references to citations")
	# In the ported query implementaion, citations seem to no longer 
	# correspond to a given statement (they are a blob now) 
	# I can't change the output type for this so I will just go with it 
	# for now
	references_flattened_dedup = list(set([ref for refs in kb_references for ref in refs]))
	async with session() as db:
		chunks = await get_chunks_by_ids(db, references_flattened_dedup)
	citations = [
		Citation(
			source_id=ch.source_id, 
			location=CitationLocation(page_number=ch.page_number), 
			text=ch.text,
		) for ch in chunks 
	]

	# Deduplication is skipped becuase I assume that we won't have two 
	# chunks with the same text 

	# 5) Create Highlight Data (node/edge key list from reaching subgraph)
	logger.debug("Fetch citation subgraph")
	all_entities = [e for a, _, b, in kb for e in [a, b]]
	citation_subgraph = await concept_subgraph(all_entities, project_id)

	# We're not returning the relations anymore? 
	# That is odd and not great for transparency 
	# TODO: Return another output object that the orchestrator can transform 
	# into a PromptResponse
	return PromptResponse(
		answer=response, citations=citations, citation_subgraph=citation_subgraph,
	), debug_items


async def setup_then(f):
	await init_postgres()
	await init_graphdb()
	return await f


def print_debug_items(debug_items):
	print("prompt_model:", debug_items["prompt_model"])
	print("embedding_model:", debug_items["embedding_model"])
	print("project_id:", debug_items["project_id"])
	print("max_retries:", debug_items["max_retries"])
	print("prompt:", debug_items["prompt"])

	for i, round in enumerate(debug_items["rounds"]):
		print(f"Round {i}:")
		print(f"Subjects: {round["subjects"]}")
		if i == 0:
			print("\tsubject_entities: ", round["subject_entities"])
			print("\tprompt_entities: ", round["prompt_entities"])
		print("\tseed_entities:", round["seed_entities"])
		print(f"\tFound {len(round["relationships_kg"])} kg relationships ({len(round["relationships_filtered"])} relevant)")
		for a, r, b in round["relationships_kg"]:
			print(f"\t\t{" ".join((a, r, b))}")
		print(f"\tAdded {len(round["relationships_new"])} new relationships")
		for a, r, b in round["relationships_new"]:
			print(f"\t\t{" ".join((a, r, b))}")
	if debug_items["iter_limit_exceeded"]:
		print("Exceeded iteration limit")
	else:
		print("Sufficient info acquired")

	print("===")
	for i, (response, removed) in enumerate(debug_items["response"]):
		if i != 0:
			print("Removing")
			for s in removed:
				print("-", s)
			print("---")
		print(response)
		print("===")
	
	print("Citations:")
	for r, s in zip(debug_items["references"], debug_items["references_sources"]):
		print("-", asyncio.run(citation_str(r, s, chunk_too=True)))
	print("Citations (unused):")
	for r, s in zip(debug_items["references_unused"], debug_items["references_sources_unused"]):
		print("-", asyncio.run(citation_str(r, s, chunk_too=True)))


def main():
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("prompt")
	parser.add_argument("--query_model", default="bedrock/us.amazon.nova-pro-v1:0")
	parser.add_argument("--embedding_model", default="bedrock/amazon.titan-embed-text-v2:0")
	parser.add_argument("--max_retries", default=5)
	parser.add_argument("--project_id") # user must have this copied
	args = parser.parse_args()

	print(f"Running manual prompt '{args.prompt}'")

	response, debug_items = asyncio.run(setup_then(prompt_reflective(args.query_model, args.embedding_model, args.project_id, args.prompt, max_retries=args.max_retries)))
	print_debug_items(debug_items)


if __name__ == "__main__":
	main()
