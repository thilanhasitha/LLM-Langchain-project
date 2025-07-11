import os
import traceback
import pandas as ps
import json
from dotenv import load_dotenv
from src.mcqgenerater.utils import read_file,get_table_data
from src.mcqgenerater.logger import logging


from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import sequentialChain

load_dotenv()

key = os.getenv("key2")

llm = ChatOpenAI(key2 = key,model_name = "gpt-3.5-turbo",temperature = 0.7)

template = """
Text:{text}
You are an expert mcq maker. Given the above text,it is your job to \
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Ensure to make {number} as MCQs.
### RESPONSE_JSON
{response_json}

"""
quiz_generation_prompt = PromptTemplate(
    input_variables = ["text","number","subject","tone","response_json"],
    template=template
)

quiz_chain = LLMChain(llm=llm,prompts = quiz_generation_prompt,output_key = "quiz",verbose=True)

template2="""
You are an expert english grammerian and writter. Given a multiple choice quiz for {subjects} students. \
You need to evaluate the complexity of the quesstion and  give a complete analysis of the quiz. Only use at max 50 words for complexity
if the quiz is not at per with the cognitive and analytical abilities of the students, \
update the quiz questions which needs to be changed and change the tone such that it ferfectly fit the students abilities.
Quiz_MCQs:
{quiz}

Check from at expert English writter of the above quiz:
"""
quiz_evaluation_prompt = PromptTemplate(input_variables=["subject","quiz"],template=template2)

review_chain= LLMChain(llm=llm,prompt = quiz_evaluation_prompt,output_key = "review",verbose=True)

generate_evaluate_chain =sequentialChain(chains=[quiz_chain,review_chain], input_variables=["text", "number", "subject", "tone", "response_json"],

                                          output_variables=["quiz","review"],verbose=True)

