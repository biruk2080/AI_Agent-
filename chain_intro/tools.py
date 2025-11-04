import random
import time
from langchain_community.vectorstores import Chroma

import dotenv
import os
os.environ["OPENAI_API_KEY"] = ''
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
# from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough


def get_current_wait_time(hospital: str) -> int | str:
    """Dummy function to generate fake wait times"""

    if hospital not in ["A", "B", "C", "D"]:
        return f"Hospital {hospital} does not exist"

    # Simulate API call delay
    time.sleep(1)

    return random.randint(0, 10000)