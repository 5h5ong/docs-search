import os
import openai
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import pinecone
from typing import TypedDict, Callable, Awaitable


class LangchainReturnType(TypedDict):
    init_pinecone: Callable[[], None]
    make_query: Callable[[str], Awaitable[str]]


def langchain(
    PINECONE_API_KEY: str, PINECONE_API_ENV: str, OPEN_API_KEY: str
) -> LangchainReturnType:
    def make_embeddings():
        MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
        MODEL_KWARGS = {"device": "cpu"}

        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME, model_kwargs=MODEL_KWARGS
        )
        return embeddings

    def init_pinecone():
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

    def make_langchain_pinecone(index_name: str, embeddings: HuggingFaceEmbeddings):
        docsearch = Pinecone.from_existing_index(index_name, embedding=embeddings)
        return docsearch

    def make_pinecone_search(pinecone: Pinecone, text: str):
        docs = pinecone.similarity_search(text)
        return docs

    def make_langchain_qa():
        llm = OpenAI(
            temperature=0, openai_api_key=OPEN_API_KEY, client=openai.ChatCompletion
        )
        chain = load_qa_chain(llm, chain_type="stuff")
        return chain

    async def make_query(query: str) -> str:
        embeddings = make_embeddings()
        pinecone = make_langchain_pinecone("example-index", embeddings)
        docs = make_pinecone_search(pinecone, query)
        chain = make_langchain_qa()
        result = chain.run(input_documents=docs, question=query)
        return result

    return {"init_pinecone": init_pinecone, "make_query": make_query}
