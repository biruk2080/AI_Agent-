import dotenv
import os
import gradio as gr
from langchain_openai import ChatOpenAI
dotenv.load_dotenv()

# Validate OpenAI API key early and give a clear error (helps with the 401 masked-key problem)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
    raise ValueError("OpenAI API key not found or invalid. Please set the OPENAI_API_KEY environment variable.")

from langchain_google_genai import ChatGoogleGenerativeAI


# chat model dependancy 
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# vector db dependancy 

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough

# prompt template dependancy 
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
# agent dependancy 
from langchain.agents import (
    create_openai_functions_agent,
    Tool,
    AgentExecutor,
)
from langchain import hub
from tools import get_current_wait_time
# test chat moddel 
# create message to the chat model 
chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, openai_api_key=OPENAI_API_KEY)


# create chat templete object 
review_system_template_str = """Your job is to use patient
 reviews to answer questions about their experience at a hospital.
 Use the following context to answer questions. Be as detailed
 as possible, but don't make up any information that's not
 from the context. If you don't know an answer, say you don't know.

 {context}
 """
# create chat template for OS 
OS_system_template_str = """You are an expert assistant trained to answer 
questions about Operating Systems using only the information provided in the 
"OPERATING SYSTEM" document. Use the following context to answer each question 
thoroughly and accurately. Be as detailed as possible, but do not make up any 
information that is not present in the context. If the answer is not found in 
the context, respond with: "I don't know."

Context:
{context}

Question:
{question}
 """

Cancer_system_template_str = """You are an expert assistant to answer 
questions about my experience and skills. Use the following context to answer each question 
thoroughly and accurately. Be as detailed as possible, but do not make up any 
information that is not present in the context. If the answer is not found in 
the context, respond with: "I don't know."

Context:
{context}

Question:
{question}
 """
# prompt templates FOR review 
review_system_prompt = SystemMessagePromptTemplate(
     prompt=PromptTemplate(
         input_variables=["context"], template=review_system_template_str
     )
 )

review_human_prompt = HumanMessagePromptTemplate(
     prompt=PromptTemplate(
         input_variables=["question"], template="{question}"
     )
 )

messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(
     input_variables=["context", "question"],
     messages=messages,
 )

# prompt templates for os 
OS_system_prompt = SystemMessagePromptTemplate(
     prompt=PromptTemplate(
         input_variables=["context"], template=OS_system_template_str
     )
 )

OS_human_prompt = HumanMessagePromptTemplate(
     prompt=PromptTemplate(
         input_variables=["question"], template="{question}"
     )
 )

messages2 = [OS_system_prompt, OS_human_prompt]

OS_prompt_template = ChatPromptTemplate(
     input_variables=["context", "question"],
     messages=messages2,
 )

# prompt templates FOR cancer
cancer_system_prompt = SystemMessagePromptTemplate(
     prompt=PromptTemplate(
         input_variables=["context"], template=Cancer_system_template_str
         )
 )

review_human_prompt = HumanMessagePromptTemplate(
     prompt=PromptTemplate(
         input_variables=["question"], template="{question}"
     )
 )

messages3 = [cancer_system_prompt, review_human_prompt]

# rename to avoid overwriting the earlier review_prompt_template
cancer_prompt_template = ChatPromptTemplate(
     input_variables=["context", "question"],
     messages=messages3,
 )
# vector db instance 
REVIEWS_CHROMA_PATH = "chroma_data/"
reviews_vector_db = Chroma(
    persist_directory=REVIEWS_CHROMA_PATH,
    embedding_function=OpenAIEmbeddings()
)

reviews_retriever = reviews_vector_db.as_retriever(k=10)
# chain for reviews (was being overwritten)
reviews_chain = (
    {"context": reviews_retriever, "question": RunnablePassthrough()}
    | review_prompt_template
    | chat_model
    | StrOutputParser()
)

# chain FOR OS (use a distinct variable)
os_chain = (
    {"context": reviews_retriever, "question": RunnablePassthrough()}
    | OS_prompt_template
    | chat_model
    | StrOutputParser()
)

# chain for Cancer (uncommented and created as separate chain)
cancer_chain = (
    {"context": reviews_retriever, "question": RunnablePassthrough()}
    | cancer_prompt_template
    | chat_model
    | StrOutputParser()
)

# agent 

# tools 
tools = [
    Tool(
        name="Reviews",
        func=reviews_chain.invoke,
        description="""Useful when you need to answer questions
        about paticent reviews or experiences at the hospital.
        Not useful for answering questions about specific visit
        details such as payer, billing, treatment, diagnosis,
        chief complaint, hospital, or physician information.
        Pass the entire question as input to the tool. For instance,
        if the question is "What do patients think about the triage system?",
        the input should be "What do patients think about the triage system?"
        """,
    ),
    Tool(
        name="Operating_System",
        func=os_chain.invoke,  # point to the OS-specific chain
        description="""You are a highly knowledgeable and helpful AI assistant 
        trained to answer questions strictly based on a document 
        titled **"OPERATING SYSTEM"**. Your job is to help users understand concepts,
        processes, and definitions related to operating systems by extracting accurate
        information from the document. You do not have access to external knowledge or 
        the internet—your answers must come only from the provided context.
        Please follow these instructions carefully:
        - Use the information provided in the context section to answer the user’s question.
        - Give clear, detailed, and informative answers that are easy to understand.
        - Only include information that is directly supported by the context. 
        Do not speculate or make assumptions.
        - If the context does not contain enough information to answer the question, 
        respond with: **"I don’t know."**
        - Aoid repeating the context verbatim—summarize and explain it naturally.
        - Maintain a professional and educational tone..
        """,
    ),
    Tool(
        name="Waits",
        func=get_current_wait_time,
        description="""Use when asked about current wait times
        at a specific hospital. This tool can only get the current
        wait time at a hospital and does not have any information about
        aggregate or historical wait times. This tool returns wait times in
        minutes. Do not pass the word "hospital" as input,
        only the hospital name itself. For instance, if the question is
        "What is the wait time at hospital A?", the input should be "A".
        """,
    ),
]
# pre- build langcahin agent tool
hospital_agent_prompt = hub.pull("hwchase17/openai-functions-agent")

# use to pass input to the langchain agent function 
hospital_agent = create_openai_functions_agent(
    llm=chat_model,
    prompt=hospital_agent_prompt,
    tools=tools,
)
# excute the task in the langchain agent 
hospital_agent_executor = AgentExecutor(
    agent=hospital_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)


def ChatPot(message):
     try:
         # pass the message as a string (not a set) and unwrap the agent's result
         result = hospital_agent_executor.invoke({"input": message})
         final_output = result.get("output", "")
         return final_output
     except Exception as e:
         # return a concise, actionable error message
         return f"Error invoking agent: {e}"

iface = gr.Interface(
    fn=ChatPot,
    inputs=["text"],
    outputs="text",
    live=False,
)
iface.launch(share=True)
