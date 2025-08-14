# Import necessary modules for prompt templates, HuggingFace integration, output parsing, and parallel execution
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# -------------------- Prompt Definitions --------------------

# Prompt 1: Used to generate a detailed summary of the input text.
# The {text} variable will be replaced with the user's input.
prompt_1 = PromptTemplate(
    template="""
    Genrate a summary of the text and create a detailed summary of the text provided ->{text}
    """,
    input_variables=["text"]
)

# Prompt 2: Used to generate a multiple-choice question (MCQ) test based on the input text.
# The {text} variable is again replaced with the user's input.
prompt_2 = PromptTemplate(
    template="""
    Genrate a mcq test with 4 options with the {text} in the last provide list with all the correct answers
    """,
    input_variables=["text"]
)

# Prompt 3: Used to merge the generated summary and MCQ test into a single set of notes.
# The {summary} and {mcq} variables will be filled with the outputs from prompt_1 and prompt_2, respectively.
prompt_3 = PromptTemplate(
    template="""
    genrate a decent notes which will help in practice merge the {summary} and the test over it into a single notes {mcq}
    """,
    input_variables=["summary", "mcq"]
)

# -------------------- LLM and Model Setup --------------------

# Initialize the HuggingFace LLM endpoint with the specified model and parameters.
# This model will be used for all text generation tasks in the chain.
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",  # Model repository ID
    task="text-generation",                         # Task type
    max_new_tokens=300                              # Maximum number of tokens to generate
)

# Wrap the LLM endpoint for chat-style interaction.
model = ChatHuggingFace(llm=llm)

# Output parser to convert the LLM's output into a string.
parser = StrOutputParser()

# -------------------- Parallel Chain Construction --------------------

# Create a parallel chain that runs two sub-chains in parallel:
#   - One generates a summary (using prompt_1)
#   - The other generates an MCQ test (using prompt_2)
# The outputs are collected in a dictionary with keys 'summary' and 'mcq'.
parllel_chains = RunnableParallel({
    'summary': prompt_1 | model | parser,  # Chain for summary generation
    'mcq': prompt_2 | model | parser       # Chain for MCQ generation
})

# -------------------- Merge Chain Construction --------------------

# This chain takes the outputs from the parallel chains ('summary' and 'mcq'),
# merges them using prompt_3, and generates the final notes.
merge_chain = (
    prompt_3 | model | parser
)

# -------------------- Full Chain Composition --------------------

# The full chain first runs the parallel chains to get the summary and MCQ,
# then merges them into a single set of notes.
chain = parllel_chains | merge_chain

# -------------------- Example Invocation --------------------

# Example input text about the stock market.
input_text = {
    'text': """
                     The stock market is a vital component of the global financial system, acting as a platform where individuals and institutions can buy and sell ownership shares in publicly traded companies. It plays a crucial role in the economic development of a country by enabling businesses to raise capital for expansion, innovation, and operations. Companies initially offer shares to the public through an Initial Public Offering (IPO), after which these shares are traded on stock exchanges. In India, the two primary stock exchanges are the Bombay Stock Exchange (BSE) and the National Stock Exchange (NSE). Globally, other major exchanges include the New York Stock Exchange (NYSE), NASDAQ, and the London Stock Exchange (LSE).

Stock prices fluctuate constantly due to a wide range of factors, including economic indicators, company performance, interest rates, geopolitical events, and investor sentiment. Positive news such as strong earnings reports, new product launches, or favorable government policies can drive stock prices up, while negative events such as economic downturns, political instability, or scandals can cause prices to fall. Market participants—ranging from retail investors and institutional investors to hedge funds and day traders—use various strategies and tools, including technical analysis, fundamental analysis, and algorithmic trading, to make investment decisions.

The stock market is often seen as a reflection of a country's economic health. When stock indices like the NIFTY 50 or the Sensex in India rise consistently, it is generally interpreted as a sign of economic growth and investor confidence. Conversely, a falling market can signal economic trouble or uncertainty. Investors can make profits through capital gains when stock prices rise, or by earning dividends—periodic payments made by some companies to their shareholders out of profits.

However, investing in the stock market involves risk. Prices can be volatile and influenced by unpredictable external factors. While some investors make significant gains, others may experience losses if they are not careful or well-informed. That’s why financial education, diversification of investments, and long-term planning are essential for minimizing risk and maximizing returns. Many investors also rely on mutual funds or exchange-traded funds (ETFs), which pool money from multiple investors to invest in a diversified portfolio of stocks, reducing individual risk.

In recent years, technology has made the stock market more accessible to the public. Online trading platforms and mobile apps allow individuals to monitor stock prices in real time and place trades with just a few clicks. Additionally, advancements in data analytics and artificial intelligence have transformed the way investors analyze markets and make decisions.

In conclusion, the stock market is more than just a place for buying and selling shares—it is a cornerstone of modern economies that facilitates capital formation, wealth creation, and financial growth. While it offers great opportunities, it also demands careful study, patience, and discipline. A well-informed and thoughtful approach to investing can help individuals and institutions benefit from the dynamic world of the stock market.
    """
}

# Invoke the full chain with the example input.
# This will generate the summary, MCQ, and merged notes.
result = chain.invoke(input_text)

# Print the final generated notes.
print(result)

# Print the ASCII representation of the computation graph for debugging/visualization.
chain.get_graph().print_ascii()