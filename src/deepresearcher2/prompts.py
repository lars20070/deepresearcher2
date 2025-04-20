#!/usr/bin/env python3

query_instructions = """
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

# summary_instructions = """
# <GOAL>
# Generate a high-quality summary of the web search results and keep it concise / related to the user topic.
# </GOAL>

# <REQUIREMENTS>
# When creating a NEW summary:
# 1. Highlight the most relevant information related to the user topic from the search results
# 2. Ensure a coherent flow of information

# When EXTENDING an existing summary:
# 1. Read the existing summary and new search results carefully.
# 2. Compare the new information with the existing summary.
# 3. For each piece of new information:
#     a. If it's related to existing points, integrate it into the relevant paragraph.
#     b. If it's entirely new but relevant, add a new paragraph with a smooth transition.
#     c. If it's not relevant to the user topic, skip it.
# 4. Ensure all additions are relevant to the user's topic.
# 5. Verify that your final output differs from the input summary.
# </REQUIREMENTS>

# <FORMATTING>
# - Start directly with the updated summary, without preamble or titles. Do not use XML tags in the output.
# </FORMATTING>"""

summary_instructions = """
<GOAL>
Generate a high-quality summary of the web search results and keep it concise / related to the user topic.
</GOAL>

<INPUT_FORMAT>
You will receive the list of web search results in XML format. The XML will use the following schema.

<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">

  <xs:element name="search_results">
    <xs:complexType>
      <xs:sequence>
        <xs:element name="WebSearchResult" maxOccurs="unbounded">
          <xs:complexType>
            <xs:sequence>
              <xs:element name="title" type="xs:string"/>
              <xs:element name="url" type="xs:anyURI"/>
              <xs:element name="content" type="xs:string"/>
            </xs:sequence>
          </xs:complexType>
        </xs:element>
      </xs:sequence>
    </xs:complexType>
  </xs:element>

</xs:schema>

Below is an example of the XML format you will receive.

<search_results>
    <WebSearchResult>
        <title>petrichor, n. meanings, etymology and more | Oxford English Dictionary</title>
        <url>https://www.oed.com/dictionary/petrichor_n</url>
        <content>Petrichor (/ˈpɛtrɪkɔːr/ PET-rih-kor) is the earthy scent produced when rain falls on dry soil.</content>
    </WebSearchResult>
    <WebSearchResult>
        <title>PETRICHOR | English meaning - Cambridge Dictionary</title>
        <url>https://dictionary.cambridge.org/dictionary/english/petrichor</url>
        <content>the smell produced when rain falls on dry ground, usually experienced as being pleasant</content>
    </WebSearchResult>
</search_results>

</INPUT_FORMAT>

<REQUIREMENTS>
When creating a summary:
1. Highlight the most relevant information related to the user topic from the search results
2. Ensure a coherent flow of information
3. Ensure the summary is relevant to the user topic and not just a collection of facts
</REQUIREMENTS>

<FORMAT>
Format your response as a JSON object with ALL of these exact keys:
   - "summary": Summary of ALL web search results. Start directly with the summary, without preamble or titles. Do not use XML tags or Markdown
   formatting in the output. The summary should be at least 100 words long. The summary should be less than 400 words long.
   - "aspect": The specific aspect of the topic being researched
</FORMAT>

<EXAMPLE>
Example output:
{{
    "summary": "Petrichor refers to the earthy scent produced when rain falls on dry soil or ground, often experienced as a pleasant smell.
    It is characterized by its distinct aroma, which is typically associated with the smell of rain on dry earth. According to dictionary definitions,
    petrichor is the term used to describe this phenomenon, with the word itself pronounced as PET-rih-kor. The smell is generally considered pleasant
    and is often noticed when rain falls on dry soil or ground, releasing the distinctive aroma into the air. The term 'petrichor' refers to the
    distinctive scent that occurs when rain falls on dry soil or rocks. The word was coined in 1964 by two Australian researchers, who discovered that
    the smell is caused by oils released from plants and soil. These oils can come from roots, leaves, and other organic matter, and are carried into
    the air by raindrops. Petrichor is often associated with the smell of earthy, mossy, or musty aromas, and is a distinctive feature of many natural
    environments.",
    "aspect": "definition and meaning",
}}
</EXAMPLE>

Provide your response in JSON format."""
