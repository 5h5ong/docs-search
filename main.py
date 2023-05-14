from fastapi import FastAPI
from functions.langchain import langchain
from models.route_types import Query
from dotenv import load_dotenv
import os

app = FastAPI()

load_dotenv()


PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")
OPEN_API_KEY = os.environ.get("OPEN_API_KEY")

if PINECONE_API_KEY is None:
    raise Exception("Pinecone의 API Key가 환경 변수에 존재하지 않습니다.")

if PINECONE_API_ENV is None:
    raise Exception("Pinecone의 ENV가 환경 변수에 존재하지 않습니다.")

if OPEN_API_KEY is None:
    raise Exception("OpenAI의 API Key가 환경 변수에 존재하지 않습니다.")

lc = langchain(PINECONE_API_KEY, PINECONE_API_ENV, OPEN_API_KEY)

lc["init_pinecone"]()


@app.post("/query/")
async def query_post(query: Query):
    result = await lc["make_query"](query.query)
    return {"query": result}
