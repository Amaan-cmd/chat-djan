from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import os

# For quick testing, set your API key here if not set globally (remove this in prod)
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# Initialize chat model (this is the new preferred way)
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# 1. Prompt to extract preferences
extract_prompt = PromptTemplate(
    input_variables=["conversation"],
    template="""
You are a career counselor. From the following conversation, extract the user's interests, hobbies, strengths, and values in a list format.

Conversation:
{conversation}

Provide the extracted information as bullet points.
"""
)

extract_chain = LLMChain(llm=llm, prompt=extract_prompt)

# 2. Prompt to recommend career path based on interests
recommend_prompt = PromptTemplate(
    input_variables=["interests"],
    template="""
Based on the following interests and preferences, recommend one of these career paths: STEM, Arts, Sports.
Write a short, friendly explanation for the recommendation.

Interests:
{interests}
"""
)

recommend_chain = LLMChain(llm=llm, prompt=recommend_prompt)

def get_user_preferences(conversation_text):
    result = extract_chain.run(conversation=conversation_text)
    return result.strip()

def recommend_career_path(interests_text):
    recommendation = recommend_chain.run(interests=interests_text)
    return recommendation.strip()


if __name__ == "__main__":
    conversation = """
    I really enjoy coding and solving math problems. I love building robots and playing chess. 
    Sometimes I like to paint and listen to music, but I spend most of my time working on tech projects.
    """

    print("Extracting preferences...")
    extracted = get_user_preferences(conversation)
    print(extracted)

    print("\nRecommending career path...")
    recommendation = recommend_career_path(extracted)
    print(recommendation)

