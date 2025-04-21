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
You will receive the knowledge gaps in XML format. The XML will use the following schema.

<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">

  <xs:element name="reflection">
    <xs:complexType>
      <xs:sequence>
        <xs:element name="knowledge_gaps" type="xs:string"/>
        <xs:element name="knowledge_coverage" type="xs:string"/>
      </xs:sequence>
    </xs:complexType>
  </xs:element>

</xs:schema>

Below is an example of the XML format you will receive.

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
1. Compile all information related to the user topic from the search results
2. The compiled paragraph should be at leasdt 1000 words long.
3. Ensure a coherent flow of information
4. Ensure the compilation is relevant to the user topic and not just a collection of facts
5. The "aspect" value in the JSON response MUST NOT include the topic itself. The aspect should be very specific to the information in the summary.
</REQUIREMENTS>

<FORMAT>
Format your response as a JSON object with ALL of these exact keys:
   - "summary": Long form compilation of ALL information of the web search results. Start directly with the compilation, without preamble or titles.
   Do not use XML tags or Markdown formatting in the output. The summary should be at least 100 words long.
   - "aspect": The specific aspect of the topic being researched. Do not include the topic itself in the aspect.
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

reflection_instructions = """
<GOAL>
Reflect on the summaries and identify knowledge gaps and coverage.
</GOAL>

<INPUT_FORMAT>
You will receive the list of web search summaries in XML format. The XML will use the following schema.

<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">

  <!-- Root element -->
  <xs:element name="search_summaries">
    <xs:complexType>
      <xs:sequence>
        <xs:element name="WebSearchSummary" maxOccurs="unbounded">
          <xs:complexType>
            <xs:sequence>
              <xs:element name="summary" type="xs:string"/>
              <xs:element name="aspect" type="xs:string"/>
            </xs:sequence>
          </xs:complexType>
        </xs:element>
      </xs:sequence>
    </xs:complexType>
  </xs:element>

</xs:schema>

Below is an example of the XML format you will receive.

<search_summaries>
  <WebSearchSummary>
    <summary>Petrichor refers to the earthy scent produced when rain falls on dry soil or ground, often experienced as a pleasant smell. It is 
    haracterized by its distinct aroma, which is typically associated with the smell of rain on dry earth. According to dictionary definitions,
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
When reflecting on the web search summaries:
1. Take all summaries into account and evaluate them in their entirety
2. Identify aspects which have been covered extensively and those which require further exploration
3. Be methodical when identifying knowldge coverage
4. Be creative and think out of the box when searching for knowledge gaps. Do not reply with 'None' or 'Nothing' for the knowledge gaps.
5. Reply only with keywords and phrases, not full sentences.
6. Do not include any XML tags or Markdown formatting in the output.
7. Do not include any preamble or titles in the output.
</REQUIREMENTS>

<FORMAT>
Format your response as a JSON object with ALL of these exact keys:
   - "knowledge_gaps": Aspects of the topic which require further exploration. Be creative, specific and detailed.
   - "knowledge_coverage": Aspects of the topic which have already been covered sufficiently.
</FORMAT>

<EXAMPLE>
Example output:
{{
    "knowledge_gaps": "The scientific research behind the causes of petrichor, its effects on human emotions and psychology, different types of
    petrichor scents in various regions or environments, and potential applications of petrichor in fields like perfumery or aromatherapy.",
    "knowledge_coverage": "definition and meaning of petrichor",
}}
</EXAMPLE>

Provide your response as a list of keywords in JSON format."""
