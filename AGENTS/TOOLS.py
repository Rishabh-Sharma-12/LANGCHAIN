from langchain_community.tools import DuckDuckGoSearchRun,ShellTool
import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent,Tool
from langchain_core.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

llm=ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="mixtral-8x7b-32768",
    temperature=0.3
)

#------------------------------------------------------

# search=DuckDuckGoSearchRun()
# result=search.invoke("")
# print(result)

#------------------------------------------------------

# shell_tool=ShellTool()
# result=shell_tool.invoke("cd .. && ls")
# print(result)

#------------------------------------------------------

# Initialize DuckDuckGo Tool
search = DuckDuckGoSearchRun()

# Dummy user input (replace with dynamic input in a real app)
user_profile = {
    "name": "Rishabh",
    "mood": "relaxed",
    "genres": "sci-fi, drama",
    "actors": "Leonardo DiCaprio, Emma Stone",
    "languages": "English, Hindi",
    "favorites": "Interstellar, La La Land"
}

# Create search prompt from user profile
search_prompt = (
    f"Movie recommendations for someone who is feeling {user_profile['mood']}, "
    f"likes {user_profile['genres']} genres, enjoys movies with {user_profile['actors']}, "
    f"prefers {user_profile['languages']} language, and loved {user_profile['favorites']}."
)

# Use DuckDuckGo to search
result = search.invoke(search_prompt)

# Print the search result
print("🎬 Recommended based on your profile:\n")
print(result)