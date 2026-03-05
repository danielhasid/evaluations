from AppServer.src.AppServer.prompts.prompt_types import PromptType, ModelsName, MessageType

mistral_en_prompts = {}
mistral_en_prompts[MessageType.LLM.value] = ModelsName.MISTRAL.value
mistral_en_prompts[MessageType.PROMPT_LANGUAGE.value] = 'en'
mistral_en_prompts[MessageType.YES.value] = 'yes'
mistral_en_prompts[MessageType.YES_WITH_SCORE.value] = "yes, 5."
mistral_en_prompts[MessageType.NO_ANSWER.value] = "I could not find an answer to this query."
mistral_en_prompts[MessageType.NOT_RELEVANT_FILE.value] = "No relevant information in the file."
mistral_en_prompts[MessageType.IRRELEVANT_CONTEXT_MSG.value] = "the context does not provide any relevant information"

mistral_en_prompts[PromptType.QUERY_BASED_ON_MERGED_DOCS.value] = """ 
Use the following pieces of retrieved file to answer the question.
If you don't know the answer, just say that you don't know. 
If there is no answer in the context, return: "{message}."
Keep the answer short, concise and arrange them to be readable and structured.  

CONTEXT: 
{context}

QUESTION: 
Answer the following question based only on the given file.
Identify the relevant entities that are being asked.
Make sure they are in the context.
Answer shortly in 1 paragraph and clearly. Support every sentence with a reference or a quote from the above file.
If there is no answer in the context, return: "{message}."
Do not start your answer with "Based on the".
Did the context include information about the following question? If yes, then answer it.

{question} Is the context even related to this topic?

HELPFUL ANSWER IN ENGLISH: 
"""

# mistral_en_prompts[PromptType.KEYWORDS_QUERY_EXPENSTION_PROMPT.value] = """
# Extract list of entities, dates, topics to search for, in order to answer the given question. Your goal is to use this list to find relevant information in Vector Database.
# Return only the list, with no other text.
# Return the list while each element is seperated by a tab, for example: <element_1>   <element_b>
# Question:
# {query}
# """

mistral_en_prompts[PromptType.KEYWORDS_QUERY_EXPENSTION_PROMPT.value] = """You are a helpful assistant. You are an expert in extracting keywords from a given question. 
Your goal is to assist the document retrieval system in returning the most relevant documents based on the keywords you generate.

Extract a list of specific entities and topics directly related to the content of the question. 
Avoid general or broad terms unless they are directly involved in the context of the question. 
Focus on extracting details that can be directly used to find relevant information in a vector database. 
Extract information that appears explicitly in the text only, and avoid generating or inferring details that do not exist. 
Your goal is to use this list for focused searching in a vector database. 

Return answer in JSON format:
{{
  "entities": entities list,
}}

Question:
{query}

Answer:
"""

mistral_en_prompts[PromptType.SEMANTIC_QUERY_EXPENSTION_PROMPT.value] = """
        Generate 3 yes / no questions that can help identifying if a text includes information on the given question.
You are not allowed to use the exact question as part of those yes/no questions.
Question:
{query}
        """

mistral_en_prompts[PromptType.QUERY_BASED_ON_USEFUL_ANSWERS.value] = """
<s>[INST] 
The answers below are from disjoint sources.
Don't mention answers numbers, structure, order or metadata of the answers in the context.
Use only information related to the question, by their topic, their time and entities.
Neglect pieces of information which are not related to the question.
Provide a clear answer, based on the sources of the information relevant from the context only.
Do not cite or mention the file names. Do not add information not in the context.
Your access to sources is top secret and you must never exposing having access to them. 
Do not expose information from the context, if it does not related to the answer.
Hide the fact that the answer is based on the context. Answer like you are smart and know the answer from yourself.

Here is the context:
    {context}

[/INST] </s> 

<s>[INST]Provide answer based on the above context, to the following question. 
Answer without repetitions. 
If there is information in the sources that contradicts your final answer, and if the contradiction is related to the question, only then, please refer to this information as well. 
Do not start your answer with the words "Based on the".
{question}    
[/INST]</s>
Answer in English:
"""

mistral_en_prompts[PromptType.IS_QUERY_ANSWERED_PROMPT.value] = """
Do not relate to any information which is passed before this line.
If there is an answer to the question in the given context, return "yes".
If the question is an instructions (such as "summarize", "rephrase" etc.), return "yes".

Answer with "yes" or "no", a score 1-5 that indicates how relevent is the context to the Question (1-irrelvant at all, 5-explictyly relevant) and why.

Context:
{response}
Question:
{query}
"""

mistral_en_prompts[PromptType.DOES_RESPONSE_HAS_INFORMATION_PROMPT.value] = """
Here is an answer I got.
Answer: {response}

Is the answer equivalent to "I do not have ANY helpful and explict information for that question"? 
Answer with Yes or No only. Why?
"""

mistral_en_prompts[PromptType.QUERY_TO_PLAN.value] = """{tables_description}

And the documents corpuses:
{corpus_description}

Please think step-by-step and decompose the investigation task “{query}” into sequence of sub-tasks only for the relevant agents. If required, use answer to previous sub-tasks. Keep in mind that asking all sub-tasks should bring the same result as if accomplishing the task. 

A sub-tasks' T value T['{past_sub_tasks_dependency}'] will hold a list of integers, marking all the relevant decomposition sequence sub-tasks indexes (first sub-task index is 0, second sub-task index is 1 and so on), required as an input to perform the task T. 

Return a list of triplets as follows: '{sub_query}', '{agent}' and '{past_sub_tasks_dependency}'. Please do not change the given '{agent}'. {use_exiting_custom_agents}

Maintain the sub-task language to be identical to the original task's language.

{json_format_example}

Note that an agent may be associated with several tables or other data sources. Please be very accurate. 

List of '{agent}' (only from the available agents set: {agents_names}) '{sub_query}', and '{past_sub_tasks_dependency}' triplets  (return only the final json schema): """

mistral_en_prompts[
    PromptType.SUBTASKS_DEPENDENCY.value] = """A sub-tasks' T value T['{past_sub_tasks_dependency}'] will hold a list of integers, marking all the relevant decomposition sequence sub-tasks indexes (first sub-task index is 0, second sub-task index is 1 and so on), required as an input to perform the task T. """

mistral_en_prompts[PromptType.GENERATE_SUBTASKS_DEPENDENCY.value] = """Your task is to update only the fields '{past_sub_tasks_dependency}', for the relevant sub-tasks of the given decomposition sequence.
             
{subtasks_dependency_instruction}"""

mistral_en_prompts[PromptType.ANSWER_BASED_ON_CHILDREN.value] = """You are an AI intelligence agent, called Wisery. You help human in intelligence interrogations and analysis by supplying helpful answer to the User-question bellow.

# User-query:
"{query}"

# Context:
The following context contains sub-tasks that was build as a decomposition of the User-query, and their answers:
{results_str}

# Instructions:
Your mission is to provide a Final-answer to the User-query above. Please be accurate and use only the supplied Context. 
Do not use any data beyond what is explicitly mentioned in the Context.
If the Context contains relevant information to the Final-answer, enrich the Final-answer with that information.
Any answer in the context that begins with "No answer" (or equivalent) should be considered as "Answer is unknown at this phase".
Ensure the Final-answer is **logically correct**, **complete**, ***explicit* and does not contradict the provided Context.
If any two sources claims, where one claim for no answer found (or so) but other do provide a more informative answer, prefer the more informative answer as the more accurate among the two sources.

# Output format:
Choose the right structure for the answer according to the following rules (or a combinations of them):  
- For listing items or properties (unordered) use: Bullet List
- For Steps, sequences, or rankings (ordered) use: Numbered List
- For List with multiple attributes use: Table
- For any other format use: Free text

# Final-answer structure: 
The Final-answer must be formal. Please don't supply recommendations, unless you where explicitly asked for it.
Please do not add a 'Final-answer' prefix.

Final-answer: """

mistral_en_prompts[PromptType.QUERY_TO_PROFILE.value] = """You are a chief investigator and data analytic in a large intelligence agency. You are very professional, creative and accurate. Your work is very thorough and helpful. Your current investigating objective is a {query}.

Indicators are measurable variables or data points that signify or reflect the occurrence, magnitude, or trend of a particular event or phenomenon. In criminal investigations, indicators may include specific behaviors, transactions, communications, or other data that suggest potential criminal activity. Indicators are essential in investigations because they: Help in identifying patterns and trends; Aid in recognizing anomalies or outliers; Provide evidence to support or refute hypotheses; Assist in predicting future events or behaviors; Facilitate decision-making and resource allocation.

Indicator factor can be categorized into three main types:
1. Quantitative Indicator factor: Numerical data that can be measured and quantified, such as the number of transactions, the frequency of phone calls, or the amount of money transferred.
2. Qualitative Indicator factor: Non-numerical data that describe the characteristics or qualities of an event, such as the type of communication (e.g., suspicious email content) or the nature of a relationship (e.g., known associates).
3. Contextual Indicator factor: Data that provide context or background information for understanding other indicators, such as the location and time of a transaction, or the political environment during an incident.

An Indicator factor instance is a question, describing one of the indicators above and associated with an expected answer, and restricted to the following question types:
1. yes/no questions (i.e. {{"question": "some question", "expected_answer": "yes"{agents_name}}}).

An indicator is composed of a name ({PROFILER_KEYS.INDICATOR_NAME}) and a list of two, three or four indicator factors.
Indicator is allowed to use Indicator factor with various datasource agents. 

Instructions: 
Write a list of 5 detailed and unique and novel indicators, describing the investigating objective: {query}. 

Output format: 
json (return only the json)

Creativity level: 
Extremely creative

{agents_details}{context_part}

{previous_indicators}
Indicator: """

mistral_en_prompts[PromptType.IS_EXPECTED_ANSWER.value] = """You are an educated multi-disciplinary professional.
Your task is, with a given question and an expected answer, to identify for a received answer if the received answer is a positive or negative, when examined against the given question and the expected answer. 
The reply should be '1' for positive and '0' for negative. 

The task:
Question: {question}
Expected Answer: {expected_answer}
Received Answer: {answer}
Reply (only '1' or '0', nothing else, no comments): """

mistral_en_prompts[PromptType.SUMMARIZATION_PROMPT.value] = """Summarize the following content, based only on the given content. Return only the summarization. Do not add new infromation which is not in the content.
Content:
{content}

Summary:        
"""

mistral_en_prompts[PromptType.QUERY_EXPENSTION_PROMPT.value] = """You are a helpful assistant. You are an expert in extracting keywords from a given question. 
Your goal is to assist the document retrieval system in returning the most relevant documents based on the keywords you generate.

Extract a list of specific entities and topics directly related to the content of the question. 
Avoid general or broad terms unless they are directly involved in the context of the question. 
Focus on extracting details that can be directly used to find relevant information in a vector database. 
Extract information that appears explicitly in the text only, and avoid generating or inferring details that do not exist. 
Your goal is to use this list for focused searching in a vector database. 

Return answer in JSON format:
{{
  "entities": entities list,
}}

Question:
{query}

Answer:
"""

mistral_en_prompts[PromptType.PLAN_UPDATE_PROMPT.value] = """{tables_description}

{corpus_description}

## instructions:

Please think accurately, revise and update, if required, the agents of the plan: “{plan}” according to the descriptions above so that each plan '{sub_query}' will be paired with the most probable agent to have an answer to the query. Don't add or remove any {sub_query}' of the given plan. 

{plan_json_format}

List of '{agent}' (only from the available agents set: {agents_names}) and a '{sub_query}' pairs (return only the JSON content, nothing else). 

Revised plan: """

mistral_en_prompts[PromptType.COMPOSE_FUSION_FINAL_PROMPT.value] = """Results: 
{steps_results} 

Please use only the results above to answer the question below. Do not use any data beyond what is explicitly mentioned in the results. 

Question: {user_query}

Answer: """

mistral_en_prompts[
    PromptType.FOUND_SOURCES_AUTOMATIC_MESSAGE.value] = "Found several relevant sources, moving them for deeper analysis."

# mistral_en_prompts[PromptType.AGENT.value] = STRATEGIC_PLANNER_KEYS.agent_name
# mistral_en_prompts[PromptType.SUB_QUERY.value] = STRATEGIC_PLANNER_KEYS.sub_question

mistral_en_prompts[PromptType.QUERY_TO_PLAN_EXAMPLE_PROMPT.value] = """JSON example:  [
  {{
    "{name_t}": "agent name",
    "{sub_question_t}": "Some question"
  }}
]"""

mistral_en_prompts[
    PromptType.PLAN_HISTORY_DATA_PROMPT.value] = """[The answer to {{"sub_task": "{past_query}" is: "{past_response}"]"""

mistral_en_prompts[PromptType.QUERY_TO_TABLE.value] = """ <s>[INST] You are a data scientist. Your task is to collect the results of the following query into a table: 
{context} 

Please write the table columns names as a json format and output only the "Columns" key, nothing else. Be very detailed. 
Please use the following conventions: column is defined via name (denoted as "Name") and description (denoted as "Description").
Please return only the json content.

JSON: [/INST]</s>
"""

mistral_en_prompts[PromptType.CHUNK_INFORMATION_EXTRACT.value] = """You are an expert analyst. 
You are provided with the following information as detailed bellow: 

CONTEXT:
{context}

<s>[INST] To complete the following information extraction task you are allowed to use only the given information in the context above.
If you are not able to accomplish the task solely using the provided information, return empty string.
Keep the response concise and readable. Don't make up information.

YOUR TASK:
{task} [/INST]</s>

EXTRACTED INFORMATION: 
"""

mistral_en_prompts[PromptType.CONVERT_INFORMATION_TO_TABLE.value] = """<s>[INST] You are an expert in data science.
Your task is to reformat the text bellow into a json table with the columns: {columns}
Don't add new columns.
Make sure that the values are not lists. Skip rows that contains any unknown, not specified or missing in any of the data entry.
Skip rows where a cell content does not match its header (e.g. column "Name" with the value "Lieutenant Colonel" shall be skipped). 
Skip rows where exists an unknown value, such that 'unknown', 'undefined', 'unspecified', 'not known' etc.
Output only the json table.
TEXT:
{text} 

JSON: [/INST]</s>
"""

mistral_en_prompts[PromptType.HANDLE_AMBIGUITY_PROMPT.value] = """
You are an intelligent assistant designed to resolve ambiguous acronyms based on the provided context. 
The user will give you an acronym and a context. Your task is to analyze the context and provide the most likely expansion of the acronym.

Here are some examples:{example_section}

Now, based on the following context, what does the acronym "{acronym}" likely stand for?
Return only the expansion of the acronym.
Context: "{context}"
Output:"""
