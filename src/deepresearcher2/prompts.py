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

<FORMATTING>
- Start directly with the summary, without preamble or titles. Do not use XML tags in the output.
</FORMATTING>"""
