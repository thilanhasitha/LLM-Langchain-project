from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI  # ✅ new import
from dotenv import load_dotenv
import os


def generate_evaluate_chain(inputs):
    prompt = PromptTemplate(
        input_variables=["text", "number", "subject", "tone"],
        template="""
        Create {number} multiple choice questions for a {subject} topic.
        Make the tone of questions {tone}.
        Text to use: {text}
        Return the questions and answers in JSON format.
        """
    )
    KEY = os.getenv("key2")
    llm = OpenAI(temperature=0.7,openai_api_key = KEY)
    chain = LLMChain(llm=llm, prompt=prompt)

    return chain.run({
        "text": inputs["text"],
        "number": inputs["number"],
        "subject": inputs["subject"],
        "tone": inputs["tone"]
    })


