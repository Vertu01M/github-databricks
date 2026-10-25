
from databricks_langchain.vectorstores import DatabricksVectorSearch
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import mlflow

## Enable MLflow Tracing
mlflow.langchain.autolog()

model_config = mlflow.models.ModelConfig(development_config="rag_chain_config.yaml")

# Turn the Vector Search index into a LangChain retriever
vector_search_as_retriever = DatabricksVectorSearch(
    endpoint=model_config.get("vector_search_endpoint_name"),
    index_name=model_config.get("vector_search_index"),
    columns=["doc_id", "asin", "overall", "summary_emoji", "document_text"],
).as_retriever(search_kwargs={"k": 5})  # number of reviews the retriever returns

# Enable the Review App and MLflow to properly track/display retrieved chunks
mlflow.models.set_retriever_schema(primary_key="doc_id", text_column="document_text")

# Method to format the docs returned by the retriever into the prompt,
# keeping the emoji and rating attached to each passage (see Cell 3's note)
def format_context(docs):
    chunk_contents = [
        f"Passage [{d.metadata.get('summary_emoji', '❓')} {d.metadata.get('overall', '?')}/5, "
        f"product {d.metadata.get('asin', 'unknown')}]: {d.page_content}\n"
        for d in docs
    ]
    return "".join(chunk_contents)

from langchain_core.prompts import ChatPromptTemplate
from databricks_langchain.chat_models import ChatDatabricks
from operator import itemgetter

prompt = ChatPromptTemplate.from_messages([
    (   # System prompt contains the sentiment-aware instructions
        "system",
        """You are a shopping assistant for baby products, helping customers understand what
other buyers experienced before they purchase. Every passage of context below comes from a real
customer review and includes: the product ID, the star rating, a sentiment emoji (😍😊😐😞😡)
computed from that review's own text — NOT your interpretation — and an excerpt of the review.

Rules:
- Base your answer ONLY on the passages in the context. Do not invent product details or reviews.
- Always report sentiment using the emoji already attached to each passage — never assign your own.
- If the passages disagree (mixed sentiment), say so explicitly rather than picking one side.
- If the context is insufficient, say "I don't have enough reviews to answer that."

Context: {context}""",
    ),
    ("user", "{question}"),  # user's question
])

# Our foundation model answering the final prompt
model = ChatDatabricks(
    endpoint=model_config.get("llm_model_serving_endpoint_name"),
    extra_params={"temperature": 0.01, "max_tokens": 500},
)

# Return the most recent user message as the question
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

# RAG Chain
chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "context": itemgetter("messages")
        | RunnableLambda(extract_user_query_string)
        | vector_search_as_retriever
        | RunnableLambda(format_context),
    }
    | prompt
    | model
    | StrOutputParser()
)

# Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=chain)

