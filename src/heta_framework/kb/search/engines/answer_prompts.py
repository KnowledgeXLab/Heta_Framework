"""Answer prompts used by built-in query engines."""

from __future__ import annotations

from heta_framework.kb.search.engines._language import numbered_context
from heta_framework.kb.search.types import QueryResult


def vector_answer_prompt(query: str, results: tuple[QueryResult, ...]) -> str:
    """Return the answer prompt for chunk vector search."""
    return _prompt(
        instruction=(
            "Answer the user question using only the retrieved text chunks. "
            "Use inline citation markers like [1] when referring to evidence. "
            "If the evidence is insufficient, say so."
        ),
        context_label="Retrieved evidence",
        query=query,
        results=results,
    )


def keyword_answer_prompt(query: str, results: tuple[QueryResult, ...]) -> str:
    """Return the answer prompt for keyword search."""
    return _prompt(
        instruction=(
            "Answer the user question using only the keyword-matched text chunks. "
            "Use inline citation markers like [1] when referring to evidence. "
            "If the evidence is insufficient, say so."
        ),
        context_label="Retrieved evidence",
        query=query,
        results=results,
    )


def graph_answer_prompt(query: str, results: tuple[QueryResult, ...]) -> str:
    """Return the answer prompt for Heta graph search."""
    return _prompt(
        instruction=(
            "Answer the user question using only the graph facts and their evidence. "
            "Explain relevant entities and relations when they matter. "
            "Use inline citation markers like [1] when referring to evidence. "
            "If the graph evidence is insufficient, say so."
        ),
        context_label="Retrieved graph evidence",
        query=query,
        results=results,
    )


def graph_rag_local_answer_prompt(local_context: str, response_type: str) -> str:
    """Return the answer prompt for GraphRAG local search."""
    return f"""---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}


---Data tables---

{local_context}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.

If you don't know the answer, just say so. Do not make anything up.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""


def graph_rag_global_map_prompt(context_data: str) -> str:
    """Return the map prompt for GraphRAG global community reports."""
    return f"""---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1...", "score": score_value}},
        {{"description": "Description of point 2...", "score": score_value}}
    ]
}}

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".
Do not include information where the supporting evidence for it is not provided.


---Data tables---

{context_data}

---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".
Do not include information where the supporting evidence for it is not provided.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1", "score": score_value}},
        {{"description": "Description of point 2", "score": score_value}}
    ]
}}
"""


def graph_rag_global_reduce_prompt(
    *,
    query: str,
    points_context: str,
    response_type: str,
) -> str:
    """Return the reduce prompt for GraphRAG global support points."""
    return  f"""---Role---

You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}


---Analyst Reports---

{points_context}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""



def hybrid_answer_prompt(query: str, results: tuple[QueryResult, ...]) -> str:
    """Return the answer prompt for hybrid vector and graph search."""
    return _prompt(
        instruction=(
            "Answer the user question using only the retrieved chunk evidence and graph facts. "
            "Prefer evidence that is directly relevant and reconcile overlapping evidence. "
            "Use inline citation markers like [1] when referring to evidence. "
            "If the evidence is insufficient, say so."
        ),
        context_label="Retrieved evidence",
        query=query,
        results=results,
    )


def rerank_answer_prompt(query: str, results: tuple[QueryResult, ...]) -> str:
    """Return the answer prompt for reranked evidence search."""
    return _prompt(
        instruction=(
            "Answer the user question using only the reranked evidence. "
            "Prioritize higher-ranked evidence and use inline citation markers like [1]. "
            "If the evidence is insufficient, say so."
        ),
        context_label="Retrieved evidence",
        query=query,
        results=results,
    )


def rewrite_answer_prompt(query: str, results: tuple[QueryResult, ...]) -> str:
    """Return the answer prompt for query rewrite search."""
    return _prompt(
        instruction=(
            "Answer the user question using only the evidence retrieved from rewritten "
            "query variants. Synthesize the evidence without mentioning internal query "
            "rewriting. Use inline citation markers like [1] when referring to evidence. "
            "If the evidence is insufficient, say so."
        ),
        context_label="Retrieved evidence",
        query=query,
        results=results,
    )


def _prompt(
    *,
    instruction: str,
    context_label: str,
    query: str,
    results: tuple[QueryResult, ...],
) -> str:
    return (
        f"{instruction}\n\n"
        f"{context_label}:\n{numbered_context(results)}\n\n"
        f"User question: {query}"
    )
