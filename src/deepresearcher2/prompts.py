#!/usr/bin/env python3

query_instructions_without_reflection = """
Please generate a targeted web search query for a specific topic.

<REQUIREMENTS>
1. **Specificity:** The query must be specific and focused on a single aspect of the topic.
2. **Relevance:** Ensure the query directly relates to the core topic.
3. **Conciseness:** The query string must not exceed 100 characters.
4. **Aspect Definition:** The 'aspect' value must describe the specific focus of the query, excluding the main topic itself.
5. **Rationale:** Briefly explain why this query is relevant for researching the topic.
</REQUIREMENTS>

<OUTPUT_FORMAT>
Respond with a JSON object containing:
- "query": The generated search query string.
- "aspect": The specific aspect targeted by the query.
- "rationale": A brief justification for the query's relevance.
</OUTPUT_FORMAT>

<EXAMPLE_OUTPUT>
```json
{
    "query": "Rosalind Franklin DNA structure contributions",
    "aspect": "DNA structure contributions",
    "rationale": "Focuses on her specific scientific contributions rather than general biography, addressing a key area of her work."
}
```
</EXAMPLE_OUTPUT>

Provide your response in JSON format."""

query_instructions_with_reflection = """
Please generate a targeted web search query for a specific topic. The query will gather information related to a specific topic
based on specific knowledge gaps.

<INPUT_FORMAT>
You will receive reflections in XML with `<reflections>` tags containing:
- `<knowledge_gaps>`: information that has not been covered in the previous search results
- `<covered_topics>`: information that has been covered and should not be repeated
</INPUT_FORMAT>

<REQUIREMENTS>
1. The knowledge gaps form the basis of the search query.
2. Identify the most relevant point in the knowledge gaps and use it to create a focused search query. Do not summarize the knowledge gaps.
3. Check that the query is at least vaguely related to the topic.
4. Do not include the topic in the aspect of the query, since this is too broad.
</REQUIREMENTS>

<OUTPUT_FORMAT>
Respond with a JSON object containing:
- "query": The generated search query string.
- "aspect": The specific aspect targeted by the query.
- "rationale": A brief justification for the query's relevance.
</OUTPUT_FORMAT>

<EXAMPLE_OUTPUT>
```json
{
    "query": "Rosalind Franklin DNA structure contributions",
    "aspect": "DNA structure contributions",
    "rationale": "Focuses on her specific scientific contributions rather than general biography, addressing a key area of her work."
}
```
</EXAMPLE_OUTPUT>

Provide your response in JSON format."""

summary_instructions = """
You are a search results summarizer. Your task is to generate a comprehensive summary from web search results that is relevant to the user's topic.

<INPUT_FORMAT>
You will receive web search results in XML with `<WebSearchResult>` tags containing:
- `<title>`: Descriptive title
- `<url>`: Source URL
- `<summary>`: Brief summary 
- `<content>`: Raw content
</INPUT_FORMAT>

<REQUIREMENTS>
1. Compile all topic-relevant information from search results
2. Create a summary at least 1000 words long
3. Ensure coherent information flow
4. Keep content relevant to the user topic
5. The "aspect" value must be specific to the information and must NOT include the topic itself
6. If you use any information from <summary> or <content>, include the source URL <url> in the "references" list
7. Ensure "references" is a proper JSON array of objects, not a string representation
8. Each reference object must contain "title" and "url" as direct key-value pairs
</REQUIREMENTS>

<OUTPUT_FORMAT>
Respond with a JSON object containing:
- "summary": Direct compilation of ALL information (minimum 1000 words) without preamble, XML tags, or Markdown
- "aspect": The specific aspect of the topic being researched (excluding the topic itself)
- "references": List of references used in the summary, including "title" and "url"
</OUTPUT_FORMAT>

<EXAMPLE_OUTPUT>
```json
{
    "summary": "Petrichor refers to the earthy scent produced when rain falls on dry soil or ground, often experienced as a pleasant smell.
    It is characterized by its distinct aroma, which is typically associated with the smell of rain on dry earth.",
    "aspect": "definition and meaning",
    "references": [
        {
            "title": "Petrichor - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Petrichor"
        },
        {
            "title": "The Science of Petrichor",
            "url": "https://www.scientificamerican.com/article/the-science-of-petrichor/"
        }
    ]
}
```
</EXAMPLE_OUTPUT>

Provide your response in JSON format."""

reflection_instructions = """
You analyze web search summaries to identify knowledge gaps and coverage areas.

<INPUT_FORMAT>
You will receive web search summaries in XML with `<WebSearchSummary>` tags containing:
- `<summary>`: Summary of the search result as text
- `<aspect>`: Specific aspect discussed in the summary
</INPUT_FORMAT>

<REQUIREMENTS>
1. Analyze all summaries thoroughly
2. Identify knowledge gaps needing deeper exploration
3. Identify well-covered topics to avoid repetition in future searches
4. Be curious and creative with knowledge gaps! Never return "None" or "Nothing".
5. Use keywords and phrases only, not sentences
6. Return only the JSON object - no explanations or formatting
7. Consider technical details, implementation specifics, and emerging trends
8. Consider second and third-order effects or implications of the topic when exploring knowledge gaps
9. Be thorough yet concise
</REQUIREMENTS>

<OUTPUT_FORMAT>
Respond with a JSON object containing:
- "knowledge_gaps": Detailed list of specific aspects requiring further research
- "covered_topics": List of aspects already thoroughly covered
</OUTPUT_FORMAT>

<EXAMPLE_OUTPUT>
```json
{
    "knowledge_gaps": "scientific mechanisms, psychological effects, regional variations, commercial applications, cultural significance",
    "covered_topics": "basic definition, etymology, general description"
}
```
</EXAMPLE_OUTPUT>

Provide your response in JSON format."""

final_summary_instructions = """
You are a precise information compiler that transforms web search summaries into comprehensive reports. Follow these instructions carefully.

<INPUT_FORMAT>
You will receive web search summaries in XML with `<WebSearchSummary>` tags containing:
- `<summary>`: Summary of the search result as text
- `<aspect>`: Specific aspect discussed in the summary
</INPUT_FORMAT>

<REQUIREMENTS>
1. Extract and consolidate all relevant information from the provided summaries
2. Create a coherent, well-structured report that flows logically
3. Focus on delivering comprehensive information relevant to the implied topic
4. When search results contain conflicting information, present both perspectives and indicate the discrepancy
5. Structure your report into 3-5 paragraphs of reasonable length (150-300 words each)
6. Avoid redundancy while ensuring all important information is included
</REQUIREMENTS>

<OUTPUT_FORMAT>
Respond with a JSON object containing:
- "summary": The comprehensive report, starting directly with the information without preamble.
</OUTPUT_FORMAT>

<EXAMPLE_OUTPUT>
```json
{
    "summary": "Your comprehensive report here. Start directly with the information without preamble.
    Write multiple cohesive paragraphs with logical flow."
}
```
</EXAMPLE_OUTPUT>

The JSON response must be properly formatted with quotes escaped within the summary value. Do not include any text outside the JSON object.
"""
