#!/usr/bin/env python3

query_instructions_without_reflection = """
Your goal is to generate a targeted web search query.
The query will gather information related to a specific topic.

<FORMAT>
Format your response as a JSON object with ALL three of these exact keys:
   - "query": The actual search query string
   - "aspect": The specific aspect of the topic being researched
   - "rationale": Brief explanation of why this query is relevant
</FORMAT>

<EXAMPLE>
Example output:
{{
    "query": "Rosalind Franklin biography",
    "aspect": "biography",
    "rationale": "The user is looking for information about Rosalind Franklin, so a search query about her biography is most relevant."
}}
</EXAMPLE>

Provide your response in JSON format."""

query_instructions_with_reflection = """
<GOAL>
Your goal is to generate a targeted web search query.
The query will gather information related to a specific topic based on specific knowledge gaps.
</GOAL>

<INPUT_FORMAT>
You will receive the knowledge gaps in XML format. Here is an example.

<reflection>
  <knowledge_gaps>impact of her work on modern molecular biology, her personal life and struggles, detailed analysis of Photograph 51</knowledge_gaps>
  <knowledge_coverage>biography and contributions</knowledge_coverage>
</reflection>

</INPUT_FORMAT>

<REQUIREMENTS>
When generating the web search query:
1. The knowledge gaps form the basis of the search query.
2. Identify the most relevant point in the knowledge gaps and use it to create a focused search query. Do not summarize the knowledge gaps.
3. Check that the query is at least vaguely related to the topic.
4. Do not include the topic in the aspect of the query, since this is too broad.
</REQUIREMENTS>

<FORMAT>
Format your response as a JSON object with ALL three of these exact keys:
   - "query": The actual search query string
   - "aspect": The specific aspect of the topic and knowledge gaps being researched. The aspect should not include the topic itself.
   - "rationale": Brief explanation of why this query is relevant
</FORMAT>

<EXAMPLE>
Example output:
{{
    "query": "Rosalind Franklin biography",
    "aspect": "biography",
    "rationale": "The user is looking for information about Rosalind Franklin, so a search query about her biography is most relevant."
}}
</EXAMPLE>

Provide your response in JSON format.

"""

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
</REQUIREMENTS>

<OUTPUT_FORMAT>
Respond with a JSON object containing:
- "summary": Direct compilation of ALL information (minimum 1000 words) without preamble, XML tags, or Markdown
- "aspect": The specific aspect of the topic being researched (excluding the topic itself)
</OUTPUT_FORMAT>

<EXAMPLE_OUTPUT>
```json
{
    "summary": "Petrichor refers to the earthy scent produced when rain falls on dry soil or ground, often experienced as a pleasant smell.
    It is characterized by its distinct aroma, which is typically associated with the smell of rain on dry earth.",
    "aspect": "definition and meaning"
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
3. Identify well-covered knowledge areas
4. Be curious and creative with knowledge gaps! Never return "None" or "Nothing".
5. Use keywords and phrases only, not sentences
6. Return only the JSON object - no explanations or formatting
7. Consider technical details, implementation specifics, and emerging trends
8. Be thorough yet concise
</REQUIREMENTS>

<OUTPUT_FORMAT>
Respond with a JSON object containing:
- "knowledge_gaps": Detailed list of specific aspects requiring further research
- "knowledge_coverage": List of aspects already thoroughly covered
</OUTPUT_FORMAT>

<EXAMPLE_OUTPUT>
```json
{
    "knowledge_gaps": "scientific mechanisms, psychological effects, regional variations, commercial applications, cultural significance",
    "knowledge_coverage": "basic definition, etymology, general description"
}
```
</EXAMPLE_OUTPUT>

Provide your response in JSON format."""

final_summary_instructions = """
You are an award winning journalist compiling a final report on a topic.

<GOAL>
1. Write one or multiple praragraphs compiling ALL information of a list of search summaries.
2. Compile the information you are given. Do not summarize or shorten the information.
</GOAL>

<INPUT_FORMAT>
You will receive the list of web search summaries in XML format. Here is an example.

<search_summaries>
  <WebSearchSummary>
    <summary>Petrichor refers to the earthy scent produced when rain falls on dry soil or ground, often experienced as a pleasant smell. It is 
    characterized by its distinct aroma, which is typically associated with the smell of rain on dry earth. According to dictionary definitions,
    petrichor is the term used to describe this phenomenon, with the word itself pronounced as PET-rih-kor. The smell is generally considered
    pleasant and is often noticed when rain falls on dry soil or ground, releasing the distinctive aroma into the air. The term 'petrichor' refers
    to the distinctive scent that occurs when rain falls on dry soil or rocks. The word was coined in 1964 by two Australian researchers, who
    discovered that the smell is caused by oils released from plants and soil. These oils can come from roots, leaves, and other organic matter,
    and are carried into the air by raindrops. Petrichor is often associated with the smell of earthy, mossy, or musty aromas, and is a distinctive
    feature of many natural environments.</summary>
    <aspect>definition and meaning</aspect>
  </WebSearchSummary>
  <WebSearchSummary>
    <summary>Petrichor refers to the distinctive scent that occurs when rain falls on dry soil or rocks, often associated with a sweet, earthy aroma.
    </summary>
    <aspect>definition and explanations</aspect>
  </WebSearchSummary>
</search_summaries>

</INPUT_FORMAT>

<REQUIREMENTS>
When creating the report:
1. Compile all information related to the user topic from the search summaries
2. The compiled should consist of three or more paragraphs. Each paragraph should be at least 1000 words long.
3. Ensure a coherent flow of information.
4. Ensure the compilation is relevant to the user topic and not just a collection of facts.
</REQUIREMENTS>

<FORMAT>
Format your response as a JSON object with a single key:
   - "summary": Long form compilation of ALL information of the web search results. Start directly with the compilation, without preamble or titles.
   Do not use XML tags in the output. Write multiple paragraphs each at least 1000 words long.
</FORMAT>

<EXAMPLE>
Example output:
{{
    "summary": "Petrichor refers to the earthy scent produced when rain falls on dry soil or ground, often experienced as a pleasant smell.
    It is characterized by its distinct aroma, which is typically associated with the smell of rain on dry earth. According to dictionary definitions,
    petrichor is the term used to describe this phenomenon, with the word itself pronounced as PET-rih-kor. The smell is generally considered pleasant
    and is often noticed when rain falls on dry soil or ground, releasing the distinctive aroma into the air. The term petrichor refers to the
    distinctive scent that occurs when rain falls on dry soil or rocks. The word was coined in 1964 by two Australian researchers, who discovered
    that the smell is caused by oils released from plants and soil. These oils can come from roots, leaves, and other organic matter, and are carried
    into the air by raindrops. Petrichor is often associated with the smell of earthy, mossy, or musty aromas, and is a distinctive feature of many
    natural environments.",
}}
</EXAMPLE>

Provide your response in JSON format.
"""
