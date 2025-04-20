"""
SHL Assessment Recommendation System - Core Logic

This module provides the core functionality for recommending SHL assessments
based on job descriptions or natural language queries.
"""

import os
import re
import json
import time
import requests
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Ensure we're using the right Pydantic version
os.environ["PYDANTIC_USE_DEPRECATION_WARNINGS"] = "False"

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_core.output_parsers import JsonOutputParser
# Always use pydantic.v1 for LangChain compatibility
from pydantic.v1 import BaseModel, Field
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Define the output schema
class Assessment(BaseModel):
    url: str = Field(description="URL to the assessment product page")
    adaptive_support: str = Field(description="Whether the assessment supports adaptive testing ('Yes' or 'No')")
    description: str = Field(description="Description of the assessment")
    duration: int = Field(description="Duration of the assessment in minutes")
    remote_support: str = Field(description="Whether the assessment supports remote testing ('Yes' or 'No')")
    test_type: List[str] = Field(description="Categories or types of the assessment")

class AssessmentRecommendations(BaseModel):
    recommended_assessments: List[Assessment] = Field(description="List of recommended SHL assessments")

def extract_urls(text):
    """Extract URLs from text using regex."""
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[-\w%!$&\'()*+,;=:~./?]*)?(?:\?[-\w%!$&\'()*+,;=:~./?]*)?(?:#[-\w%!$&\'()*+,;=:~./?]*)?'
    return re.findall(url_pattern, text)

def scrape_job_description(url):
    """Scrape job description content from a URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text and clean it up
        text = soup.get_text(separator='\n')
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text
    except Exception as e:
        return f"Error scraping URL {url}: {str(e)}"

def setup_pinecone():
    """Initialize and connect to Pinecone index."""
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index if it doesn't exist
    index_name = "shl-assessments"
    dimension = 768  # Set this to the dimension of your embeddings

    # List existing indexes and create if not exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # Connect to the index
    index = pc.Index(index_name)
    return index, index_name

def setup_vector_store(index_name):
    """Initialize vector store from existing Pinecone index."""
    # Initialize the OpenAI embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )

    # Initialize vector store
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        text_key="chunk",
        pinecone_api_key=PINECONE_API_KEY
    )

    return vectorstore

def get_json_assessment_recommendations(query, vectorstore, top_k=15, temperature=0, model="gpt-4.1-mini"):
    """
    Process a query and return SHL assessment recommendations in JSON format.

    Args:
        query (str): The query to process
        vectorstore: Vector store for document retrieval
        top_k (int): Number of relevant documents to retrieve
        temperature (float): Controls randomness in output
        model (str): The OpenAI model to use

    Returns:
        dict: JSON object with assessment recommendations
    """
    # Use a JSON-specific prompt template
    json_template = """
    You are an HR assessment expert who recommends SHL assessments based on job requirements.

    Context information about available assessments:
    {context}

    User query: {input}

    Based ONLY on the assessment information in the context, recommend between 1-10 most relevant assessments.

    Return your recommendations in JSON format only with these fields for each assessment:
    - url: The URL to the assessment
    - adaptive_support: "Yes" or "No"
    - description: Brief description of the assessment
    - duration: The duration in minutes (as a number)
    - remote_support: "Yes" or "No"
    - test_type: Array of test categories

    Only recommend assessments that match the requirements in the query.
    Consider skills, time constraints, and job level requirements.
    """

    json_prompt = PromptTemplate(
        template=json_template,
        input_variables=["context", "input"]
    )

    # Initialize the parser and language model
    parser = JsonOutputParser(pydantic_object=AssessmentRecommendations)

    # Create a model specifically for JSON output
    json_llm = ChatOpenAI(
        model_name=model,
        temperature=temperature,
        openai_api_key=OPENAI_API_KEY,
        response_format={"type": "json_object"}  # Force JSON response
    )

    # Create the document chain for JSON output
    json_document_chain = create_stuff_documents_chain(
        llm=json_llm,
        prompt=json_prompt
    )

    # Create retrieval chain
    json_qa_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(search_kwargs={"k": top_k}),
        combine_docs_chain=json_document_chain
    )

    # Process the query
    result = json_qa_chain.invoke({"input": query})

    # The response should already be structured as JSON
    return result["answer"]

def get_enhanced_json_recommendations(query, vectorstore, top_k=15, temperature=0, model="gpt-4.1-mini"):
    """
    Process a query with potential job description URLs and return SHL assessment recommendations in JSON format.

    Args:
        query (str): A query that may contain job description URLs
        vectorstore: Vector store for document retrieval
        top_k (int): Number of relevant documents to retrieve
        temperature (float): Controls randomness in output
        model (str): The OpenAI model to use

    Returns:
        dict: JSON object with assessment recommendations
    """
    try:
        urls = extract_urls(query)

        if not urls:
            # No URLs found, process query normally
            return get_json_assessment_recommendations(query, vectorstore, top_k, temperature, model)

        # URLs found, scrape content
        job_descriptions = []
        for url in urls:
            job_desc = scrape_job_description(url)
            if len(job_desc) > 200:  # Only use if we got meaningful content
                job_descriptions.append(f"Job Description from {url}:\n{job_desc}")

        # Remove URLs from the original query
        clean_query = query
        for url in urls:
            clean_query = clean_query.replace(url, "")
        clean_query = clean_query.strip()

        # Combine the original query (minus URLs) with the job descriptions
        if job_descriptions:
            combined_query = f"{clean_query}\n\nAnalyze these job descriptions to recommend assessments:\n\n" + "\n\n".join(job_descriptions)
        else:
            combined_query = clean_query

        # Use existing function with enhanced query
        return get_json_assessment_recommendations(combined_query, vectorstore, top_k, temperature, model)

    except Exception as e:
        error_message = f"Error processing query with URLs: {str(e)}"
        return {"error": error_message}

def recommend_assessments(query, top_k=15, temperature=0, model="gpt-4.1-mini"):
    """
    Main function to process a query and output assessment recommendations.

    Args:
        query (str): The query to process (may include URLs)
        top_k (int): Number of relevant documents to retrieve
        temperature (float): Controls randomness in output
        model (str): The OpenAI model to use

    Returns:
        dict: JSON object with assessment recommendations
    """
    # Set up Pinecone and vector store
    try:
        index, index_name = setup_pinecone()
        vectorstore = setup_vector_store(index_name)

        # Process the query
        recommendations = get_enhanced_json_recommendations(query, vectorstore, top_k, temperature, model)
        
        # Parse the JSON if it's a string
        if isinstance(recommendations, str):
            recommendations = json.loads(recommendations)
            
        return recommendations
    except Exception as e:
        return {"error": str(e)}