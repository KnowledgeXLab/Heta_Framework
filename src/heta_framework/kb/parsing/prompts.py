"""Default prompts used by knowledge base parsers."""

IMAGE_DESCRIPTION_GUIDELINES = """Describe the image for a knowledge base.
Write a detailed, retrieval-friendly description. Cover the following when visible:
- All readable text, labels, captions, numbers, tables, chart legends, and UI text.
- Main objects, people, diagrams, screenshots, layout, spatial relationships, and visual hierarchy.
- The likely document context or intent of the image.
- Important facts that a user may later search for.

Do not invent details that are not visible. If text is unclear, say it is unclear.
Preserve the language used in the image when transcribing visible text."""

DEFAULT_HTML_IMAGE_DESCRIPTION_PROMPT = f"""Describe this image from an HTML page.

{IMAGE_DESCRIPTION_GUIDELINES}

Use the surrounding webpage context, existing caption, alt text, and image URL only as hints.
If those hints conflict with the visible image, trust the visible image."""

DEFAULT_IMAGE_DESCRIPTION_PROMPT = f"""Describe this standalone image.

{IMAGE_DESCRIPTION_GUIDELINES}"""

DEFAULT_TABLE_DESCRIPTION_PROMPT = (
    "Describe this table for knowledge base retrieval. "
    "Use a concise title and summarize the columns and visible sample rows."
)
