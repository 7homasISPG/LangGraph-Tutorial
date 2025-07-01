from fastapi import FastAPI, UploadFile, File
import PyPDF2
import pandas as pd
import textwrap
from pydantic import BaseModel
from cdb import upsert_documents
from agentic_rag import agentic_rag
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

class QueryModel(BaseModel):
    query: str

@app.post("/upload/")
async def upload_files(files: list[UploadFile] = File(...)):
    all_chunks = []
    for file in files:
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension == "pdf":
            pdf_text = ''
            reader = PyPDF2.PdfReader(file.file)
            for page in reader.pages:
                pdf_text += page.extract_text() or ''
            chunks = textwrap.wrap(pdf_text, 1000)
        elif file_extension == "csv":
            df = pd.read_csv(file.file)
            csv_text = df.to_string(index=False)
            chunks = textwrap.wrap(csv_text, 1000)
        else:
            return {"error": f"Unsupported file type: {file.filename}"}
        all_chunks.extend(chunks)

    upsert_documents(documents=all_chunks)
    return {"message": "Files uploaded and processed", "total_chunks": len(all_chunks)}

@app.post("/query/")
async def query_data(query_data: QueryModel):
    result = agentic_rag.invoke({"query": query_data.query})
    return {"response": result["response"]}
