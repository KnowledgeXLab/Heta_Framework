"""Prompts for graph-building steps."""

ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are a precise knowledge graph entity extraction engine.
Return only valid JSON. Do not include markdown, explanations, or extra text."""

ENTITY_EXTRACTION_PROMPT = """Extract knowledge graph entities from the chunk below.

Rules:
- Return a JSON object with exactly one top-level key: "entities".
- "entities" must be an array.
- Each entity must include: name, type, subtype, description, attributes.
- name must be a specific named entity, not a pronoun or vague phrase.
- type must be a concise entity category.
- subtype may be null when no reliable subtype is available.
- description must be a short factual description grounded in the chunk.
- attributes must be a JSON object of string keys and string values.
- Do not invent facts that are not supported by the chunk.
- If no reliable entities exist, return {{"entities": []}}.

Chunk metadata:
- chunk_id: {chunk_id}
- document_id: {document_id}
- source: {source_name}

Chunk text:
{chunk_text}
"""

ENTITY_EXTRACTION_RETRY_PROMPT = """The previous entity extraction response was invalid.

Validation error:
{error}

Return the corrected result for the same chunk. Return only valid JSON with this shape:
{{"entities":[{{"name":"...","type":"...","subtype":null,"description":"...","attributes":{{}}}}]}}

Chunk metadata:
- chunk_id: {chunk_id}
- document_id: {document_id}
- source: {source_name}

Chunk text:
{chunk_text}
"""

RELATION_EXTRACTION_SYSTEM_PROMPT = """You are a precise knowledge graph relation extraction engine.
Return only valid JSON. Do not include markdown, explanations, or extra text."""

RELATION_EXTRACTION_PROMPT = """Extract knowledge graph relations from the chunk below.

Rules:
- Return a JSON object with exactly one top-level key: "relations".
- "relations" must be an array.
- Each relation must include: source, target, type, name, description, attributes.
- source and target must exactly match names from the provided entities.
- Do not create new entities.
- Do not create self-relations.
- type must be a concise relation category.
- name must be the specific relation name.
- description must be a short factual description grounded in the chunk.
- attributes must be a JSON object of string keys and string values.
- Do not invent facts that are not supported by the chunk.
- If no reliable relations exist, return {{"relations": []}}.

Chunk metadata:
- chunk_id: {chunk_id}
- document_id: {document_id}
- source: {source_name}

Entities:
{entities_json}

Chunk text:
{chunk_text}
"""

RELATION_EXTRACTION_RETRY_PROMPT = """The previous relation extraction response was invalid.

Validation error:
{error}

Return the corrected result for the same chunk. Return only valid JSON with this shape:
{{"relations":[{{"source":"...","target":"...","type":"...","name":"...","description":"...","attributes":{{}}}}]}}

Remember:
- source and target must exactly match names from the provided entities.
- Do not create new entities.
- Do not create self-relations.

Chunk metadata:
- chunk_id: {chunk_id}
- document_id: {document_id}
- source: {source_name}

Entities:
{entities_json}

Chunk text:
{chunk_text}
"""

HETA_ENTITY_EXTRACTION_PROMPT = ENTITY_EXTRACTION_PROMPT
HETA_RELATION_EXTRACTION_PROMPT = RELATION_EXTRACTION_PROMPT

ONTOLOGY_CONSTRAINT_SYSTEM_PROMPT = """You are a precise ontology constraint engine.
Return only valid JSON. Do not include markdown, explanations, or extra text."""

ONTOLOGY_ENTITY_CONSTRAINT_PROMPT = """Constrain extracted knowledge graph entities to the ontology schema.

Rules:
- Return a JSON object with exactly one top-level key: "entities".
- Each returned entity must keep the same JSON shape as the input entity.
- Keep only entities grounded in the input chunk and compatible with the schema.
- You may map entity.type to an allowed schema type when the evidence supports it.
- You may improve descriptions only when grounded in the input chunk.
- Delete entities that are irrelevant, unsupported, or cannot be mapped to the schema.

Ontology schema:
{schema_json}

Entities:
{entities_json}

Chunk text:
{chunk_text}
"""

ONTOLOGY_RELATION_CONSTRAINT_PROMPT = """Constrain extracted knowledge graph relations to the ontology schema.

Rules:
- Return a JSON object with exactly one top-level key: "relations".
- Each returned relation must keep the same JSON shape as the input relation.
- Keep only relations grounded in the input chunk and compatible with the schema.
- source and target must refer to entities from the provided constrained entity list.
- You may map relation.type to an allowed schema type when the evidence supports it.
- You may improve relation names and descriptions only when grounded in the input chunk.
- Delete relations whose endpoints do not exist or whose endpoint types violate the schema.

Ontology schema:
{schema_json}

Constrained entities:
{entities_json}

Relations:
{relations_json}

Chunk text:
{chunk_text}
"""

ONTOLOGY_ENTITY_CONSTRAINT_RETRY_PROMPT = """The previous ontology entity constraint response was invalid.

Validation error:
{error}

Return corrected JSON with this shape:
{{"entities":[{{"entity_id":"...","chunk_id":"...","document_id":"...","name":"...","type":"...","subtype":null,"description":"...","attributes":{{}},"source_chunk_ids":["..."]}}]}}

Ontology schema:
{schema_json}

Entities:
{entities_json}

Chunk text:
{chunk_text}
"""

ONTOLOGY_RELATION_CONSTRAINT_RETRY_PROMPT = """The previous ontology relation constraint response was invalid.

Validation error:
{error}

Return corrected JSON with this shape:
{{"relations":[{{"relation_id":"...","chunk_id":"...","document_id":"...","source_entity_id":"...","target_entity_id":"...","source_entity_name":"...","target_entity_name":"...","type":"...","name":"...","description":"...","attributes":{{}},"source_chunk_ids":["..."]}}]}}

Ontology schema:
{schema_json}

Constrained entities:
{entities_json}

Relations:
{relations_json}

Chunk text:
{chunk_text}
"""

LIGHTRAG_RELATION_KEYWORDS_PROMPT = """Extract concise LightRAG relationship keywords.

Rules:
- Return a JSON object with exactly one top-level key: "keywords".
- "keywords" must be a comma-separated string.
- Use short topical phrases grounded in the relation and chunk.
- Do not invent facts.

Relation:
{relation_json}

Chunk text:
{chunk_text}
"""

ENTITY_DEDUPLICATION_SYSTEM_PROMPT = """You are a precise knowledge graph entity deduplication engine.
Return only valid JSON. Do not include markdown, explanations, or extra text."""

ENTITY_DEDUPLICATION_PROMPT = """Merge duplicate knowledge graph entities.

Rules:
- Return a JSON object with exactly one top-level key: "entity".
- The entity must include: name, type, subtype, description, attributes.
- Preserve only facts supported by the input entities.
- Prefer the clearest canonical name.
- description must be concise but include the useful facts from all duplicates.
- attributes must be a JSON object of string keys and string values.

Entities:
{entities_json}
"""

ENTITY_DEDUPLICATION_RETRY_PROMPT = """The previous entity deduplication response was invalid.

Validation error:
{error}

Return the corrected result for the same entities. Return only valid JSON with this shape:
{{"entity":{{"name":"...","type":"...","subtype":null,"description":"...","attributes":{{}}}}}}

Entities:
{entities_json}
"""

RELATION_DEDUPLICATION_SYSTEM_PROMPT = """You are a precise knowledge graph relation deduplication engine.
Return only valid JSON. Do not include markdown, explanations, or extra text."""

RELATION_DEDUPLICATION_PROMPT = """Merge duplicate knowledge graph relations.

Rules:
- Return a JSON object with exactly one top-level key: "relation".
- The relation must include: type, name, description, attributes.
- Preserve only facts supported by the input relations.
- description must be concise but include the useful facts from all duplicates.
- attributes must be a JSON object of string keys and string values.
- Do not change the relation endpoints.

Relations:
{relations_json}
"""

RELATION_DEDUPLICATION_RETRY_PROMPT = """The previous relation deduplication response was invalid.

Validation error:
{error}

Return the corrected result for the same relations. Return only valid JSON with this shape:
{{"relation":{{"type":"...","name":"...","description":"...","attributes":{{}}}}}}

Relations:
{relations_json}
"""

GRAPH_SUMMARY_PROMPT = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""


GRAPH_RAG_ENTITY_EXTRACTION_PROMPT="""-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}9){completion_delimiter}
#############################
Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}9){completion_delimiter}
#############################
Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}7){completion_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""
GRAPH_RAG_ENTITY_CONTINUE_EXTRACTION_PROMPT="""MANY entities were missed in the last extraction.  Add them below using the same format:
"""
GRAPH_RAG_ENTITY_IF_LOOP_EXTRACTION_PROMPT= """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""
GRAPH_RAG_COMMUNITY_REPORT_PROMPT = """You are an AI assistant that helps a human analyst to perform general information discovery.
Information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., organizations and individuals) within a network.

# Goal
Write a comprehensive report of a community, given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact. The content of this report includes an overview of the community's key entities, their legal compliance, technical capabilities, reputation, and noteworthy claims.

# Report Structure

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
            ...
        ]
    }}

# Grounding Rules
Do not include information where the supporting evidence for it is not provided.


# Example Input
-----------
Text:
```
Entities:
```csv
id,entity,type,description
5,VERDANT OASIS PLAZA,geo,Verdant Oasis Plaza is the location of the Unity March
6,HARMONY ASSEMBLY,organization,Harmony Assembly is an organization that is holding a march at Verdant Oasis Plaza
```
Relationships:
```csv
id,source,target,description
37,VERDANT OASIS PLAZA,UNITY MARCH,Verdant Oasis Plaza is the location of the Unity March
38,VERDANT OASIS PLAZA,HARMONY ASSEMBLY,Harmony Assembly is holding a march at Verdant Oasis Plaza
39,VERDANT OASIS PLAZA,UNITY MARCH,The Unity March is taking place at Verdant Oasis Plaza
40,VERDANT OASIS PLAZA,TRIBUNE SPOTLIGHT,Tribune Spotlight is reporting on the Unity march taking place at Verdant Oasis Plaza
41,VERDANT OASIS PLAZA,BAILEY ASADI,Bailey Asadi is speaking at Verdant Oasis Plaza about the march
43,HARMONY ASSEMBLY,UNITY MARCH,Harmony Assembly is organizing the Unity March
```
```
Output:
{{
    "title": "Verdant Oasis Plaza and Unity March",
    "summary": "The community revolves around the Verdant Oasis Plaza, which is the location of the Unity March. The plaza has relationships with the Harmony Assembly, Unity March, and Tribune Spotlight, all of which are associated with the march event.",
    "rating": 5.0,
    "rating_explanation": "The impact severity rating is moderate due to the potential for unrest or conflict during the Unity March.",
    "findings": [
        {{
            "summary": "Verdant Oasis Plaza as the central location",
            "explanation": "Verdant Oasis Plaza is the central entity in this community, serving as the location for the Unity March. This plaza is the common link between all other entities, suggesting its significance in the community. The plaza's association with the march could potentially lead to issues such as public disorder or conflict, depending on the nature of the march and the reactions it provokes."
        }},
        {{
            "summary": "Harmony Assembly's role in the community",
            "explanation": "Harmony Assembly is another key entity in this community, being the organizer of the march at Verdant Oasis Plaza. The nature of Harmony Assembly and its march could be a potential source of threat, depending on their objectives and the reactions they provoke. The relationship between Harmony Assembly and the plaza is crucial in understanding the dynamics of this community."
        }},
        {{
            "summary": "Unity March as a significant event",
            "explanation": "The Unity March is a significant event taking place at Verdant Oasis Plaza. This event is a key factor in the community's dynamics and could be a potential source of threat, depending on the nature of the march and the reactions it provokes. The relationship between the march and the plaza is crucial in understanding the dynamics of this community."
        }},
        {{
            "summary": "Role of Tribune Spotlight",
            "explanation": "Tribune Spotlight is reporting on the Unity March taking place in Verdant Oasis Plaza. This suggests that the event has attracted media attention, which could amplify its impact on the community. The role of Tribune Spotlight could be significant in shaping public perception of the event and the entities involved."
        }}
    ]
}}


# Real Data

Use the following text for your answer. Do not make anything up in your answer.

Text:
```
{input_text}
```

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
            ...
        ]
    }}

# Grounding Rules
Do not include information where the supporting evidence for it is not provided.

Output:
"""

LIGHTRAG_PROMPTS = {
    "default_entity_types_guidance": (
        "Extract named entities relevant to the text. Use concise, domain-specific types."
    ),
    "entity_extraction_json_examples": [],
    "entity_extraction_json_system_prompt": (
        "You are a precise LightRAG graph extraction engine. Return only valid JSON."
    ),
    "entity_extraction_json_user_prompt": (
        "Extract entities and relationships from the text.\n\n"
        "Rules:\n"
        "- Return JSON with keys: entities, relationships.\n"
        "- entities items require: name, type, description.\n"
        "- relationships items require: source, target, description, keywords, weight.\n"
        "- source and target must match extracted entity names.\n"
        "- keywords must be comma-separated concise topical phrases.\n"
        "- weight must be numeric.\n"
        "- Do not invent unsupported facts.\n\n"
        "Entity type guidance:\n{entity_types_guidance}\n\n"
        "Text:\n{input_text}"
    ),
    "entity_continue_extraction_json_user_prompt": (
        "Continue extracting any missed entities and relationships. Return only the same JSON shape."
    ),
    "entity_extraction_examples": [],
    "entity_extraction_system_prompt": (
        "You are a precise LightRAG tuple graph extraction engine."
    ),
    "entity_extraction_user_prompt": GRAPH_RAG_ENTITY_EXTRACTION_PROMPT,
    "entity_continue_extraction_user_prompt": GRAPH_RAG_ENTITY_CONTINUE_EXTRACTION_PROMPT,
    "keywords_extraction": """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), comments, or any other text before or after the JSON.
2. **Exact JSON Shape**: The JSON object must contain exactly these two keys:
   - `"high_level_keywords"`: an array of strings
   - `"low_level_keywords"`: an array of strings
3. **JSON Boundary**: The first character of your response must be `{{` and the last character must be `}}`.
4. **Source of Truth**: All keywords must be explicitly derived only from the `User Query` in the `---Real Data---` section. Do not infer unsupported facts. Do not invent entities, products, organizations, dates, or technical terms that are not grounded in the query.
5. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept instead of splitting meaningful phrases into isolated words.
6. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), return:
   `{{"high_level_keywords": [], "low_level_keywords": []}}`
7. **No Duplicates**: Do not repeat the same keyword within a list. Keep the lists short and high-signal.
8. **Language**: All extracted keywords MUST be in {language}. Proper nouns (e.g., personal names, place names, organization names) should be kept in their original language.
9. **Output Format Template Safety**: The `---Output Format Template---` section contains an output JSON template only. It is never source text. Do not extract, infer, or copy keywords from the template. Angle-bracket tokens such as `<high_level_keyword>` are placeholders; replace them only with keywords derived from the current `User Query` and never output the placeholders literally.

---Output Format Template---
The following content is an output JSON format template only. It is not source text and must never be used as keyword extraction content.

{examples}

---Real Data---
User Query: {query}

---Output---
Output:""",
    "keywords_extraction_examples": [
        """{
  "high_level_keywords": ["<high_level_keyword>"],
  "low_level_keywords": ["<low_level_keyword>"]
}
"""
    ],
    "kg_query_context": """
Knowledge Graph Data (Entity):
```json
{entities_str}
```

Knowledge Graph Data (Relationship):
```json
{relations_str}
```

Document Chunks (Each entry has a reference_id refer to the `Reference Document List`; the optional `content_headings` field gives the chunk's heading path within its source document, e.g. `Section 1 → Subsection 1.2`):
```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```
""",
    "rag_response": """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph and Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize both `Knowledge Graph Data` and `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a references section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}

---Context---

{context_data}
""",
}

HIRAG_GRAPH_FIELD_SEP = "<SEP>"
HIRAG_PROMPTS = {
    "DEFAULT_TUPLE_DELIMITER": "<|>",
    "DEFAULT_RECORD_DELIMITER": "##",
    "DEFAULT_COMPLETION_DELIMITER": "<|COMPLETE|>",
    "META_ENTITY_TYPES": ["organization", "person", "location", "event"],
    "hi_entity_extraction": GRAPH_RAG_ENTITY_EXTRACTION_PROMPT,
    "hi_relation_extraction": (
        "Given the extracted entities and the source text, extract relationships among "
        "the entities only.\n\nEntities:\n{entities}\n\nText:\n{input_text}"
    ),
    "entiti_continue_extraction": GRAPH_RAG_ENTITY_CONTINUE_EXTRACTION_PROMPT,
    "entiti_if_loop_extraction": GRAPH_RAG_ENTITY_IF_LOOP_EXTRACTION_PROMPT,
    "summary_clusters": (
        "You are given a list of related entity descriptions. Synthesize one concise "
        "higher-level entity that represents the shared concept.\n\n"
        "Entity description list: {entity_description_list}"
    ),
    "summarize_entity_descriptions": GRAPH_SUMMARY_PROMPT,
    "community_report": GRAPH_RAG_COMMUNITY_REPORT_PROMPT,
    "local_rag_response": (
        "You are a helpful assistant answering questions from HiRAG retrieval context.\n\n"
        "Use only the context data below to answer. If the context does not contain "
        "enough information, say that the available context is insufficient.\n"
        "Do not invent facts. Cite source identifiers from the context when useful.\n\n"
        "Response type: {response_type}\n\n"
        "Context data:\n"
        "{context_data}"
    ),
}

HIRAG_COMMUNITY_SUMMARY_PROMPT = GRAPH_RAG_COMMUNITY_REPORT_PROMPT
HIRAG_ENTITY_SUMMARY_PROMPT = GRAPH_SUMMARY_PROMPT

LEANRAG_AGGREGATE_ENTITY_PROMPT = """
# Role: Entity Aggregation Analyst

## Profile
- author: LangGPT
- version: 1.0
- language: English
- description: You are an expert in concept synthesis. Your task is to identify a meaningful aggregate entity from a set of related entities and extract structured insights based solely on provided evidence.

## Skills
- Abstraction and naming of collective concepts based on entity types
- Structured summarization and typology recognition
- Comparative analysis across multiple entities
- Strict grounding to provided data (no hallucinated content)

## Goals
- Derive a meaningful aggregate entity that broadly represents the given entity set
- The aggregate entity name must not match any single entity in the set
- Provide an accurate and concise description of the aggregate entity reflecting shared characteristics
- Extract 5–10 structured findings about the entity set based on grounded evidence

## OutputFormat
Format:
Input:
{input_text}

Output:
{{
      "entity_name": "<name>",
      "entity_description": "<brief description summarizing the shared traits and structure>",
      "findings": [
        {{
          "summary": "<summary_1>",
          "explanation": "<explanation_1>"
        }},
        {{
          "summary": "<summary_2>",
          "explanation": "<explanation_2>"
        }}
        // ...
      ]
    }}

## Rules
- Grounding Rule: All content must be based solely on the provided entity set — no external assumptions
- Naming Rule: The aggregate entity name must not be identical to any single entity; it should reflect a composite structure, function, or theme
- Each finding must include a concise summary and a detailed explanation
- Avoid adding speculative or unsupported interpretations

## Workflows
1. Review the list of entities, focusing on types, descriptions, and relational structure
2. Synthesize a generalized name that best represents the full entity set
3. Write a clear, evidence-based description of the aggregate entity
4. Extract and elaborate on key findings, emphasizing structure, purpose, and interconnections
"""

LEANRAG_AGGREGATE_RELATION_PROMPT = """
# Role: Inter-Aggregation Relationship Analyst

## Profile
- author: LangGPT
- version: 1.1
- language: English
- description: You specialize in analyzing relationships between two aggregation entities. Your goal is to synthesize one high-level, abstract summary sentence describing how two named aggregations are connected, based solely on their descriptions and sub-entity relationships.

## Skills
- Aggregated reasoning across entity groups
- Abstraction of cross-entity relationships
- Formal summarization under strict constraints
- Strong grounding without repetition or speculation

## Goals
- Produce a single-sentence summary (≤{tokens} words) explaining the nature of the relationship between two aggregation entities
- Avoid reproducing individual sub-entity relationships
- Emphasize structural, functional, or thematic connections at the group level

---

## InputFormat
Aggregation A Name: {entity_a}
Aggregation A Description: {entity_a_description}

Aggregation B Name: {entity_b}
Aggregation B Description: {entity_b_description}

Sub-Entity Relationships:
{relation_information}
---

## OutputFormat
<Single-sentence explanation (≤{tokens} words) summarizing the relationship between Aggregation A and Aggregation B. Use abstract group-level language and do not include names or specific node-level relationships.>

---

## Rules

- DO NOT output `relationship<|>` lines or copy sub-entity relationship descriptions
- DO NOT name specific sub-entities (e.g., individuals)
- DO NOT use the term “community”; always refer to “aggregation,” “group,” “collection,” or thematic equivalents
- DO use collective terms (e.g., “external reviewers,” “trade policy actors”)
- The sentence must be ≤{tokens} words, factual, grounded, and in formal English
- The relationship must reflect an **aggregation-level abstraction**, such as:
  - support/collaboration
  - review/feedback
  - functional alignment
  - domain linkage (e.g., one produces work, the other evaluates it)

## Example

### Input:
Aggregation A Name: WTO External Contributors
Aggregation A Description: A group of economists and trade policy experts who provided feedback on early drafts of WTO reports.

Aggregation B Name: WTO Flagship Reports
Aggregation B Description: Core analytical publications from the WTO addressing international trade issues.

Sub-Entity Relationships:
- Person A → early drafts of WTO report → gave feedback
- Person B → early drafts → reviewed document

### Output:
WTO External Contributors played an advisory role to the WTO Flagship Reports aggregation by offering critical expert feedback on preliminary drafts, strengthening the analytical rigor and credibility of the final publications.
"""

LEANRAG_PROMPTS = {
    "aggregate_entities": LEANRAG_AGGREGATE_ENTITY_PROMPT,
    "cluster_cluster_relation": LEANRAG_AGGREGATE_RELATION_PROMPT,
    "rag_response": """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

Multiple Paragraphs


---Data tables---

{context_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.

If you don't know the answer, just say so. Do not make anything up.

Do not include information where the supporting evidence for it is not provided.


Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
""",
}
