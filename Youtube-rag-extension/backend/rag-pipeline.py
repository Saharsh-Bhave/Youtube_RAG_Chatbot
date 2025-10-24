from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import re

def get_answer(video_id: str, question: str) -> str:
    YTapi = YouTubeTranscriptApi()
    transcript_list = YTapi.fetch(video_id, languages=["en"])
    raw_transcript = " ".join(chunk.text for chunk in transcript_list)
    clean_transcript = re.sub(r"\[.*?\]", "", raw_transcript)
    clean_transcript = re.sub(r"\s+", " ", clean_transcript).strip()

    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    chunks = splitter.create_documents([clean_transcript])

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs={"k":4})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    prompt = PromptTemplate(
    template = """
        You are a highly knowledgeable assistant specialized in analyzing YouTube video transcripts.

        Your task is to provide a **comprehensive and well-structured answer** to the user's question based **only on the given transcript context**.

        Guidelines:
        - Use only the facts, details, and phrasing from the transcript.
        - If relevant information is missing, clearly state that the context does not provide enough detail.
        - Write in complete sentences and elaborate thoughtfully.
        - Organize your response logically — include key points, examples, or reasoning **only from the transcript**.
        - Avoid repeating the question or hallucinating information not in the transcript.

        Context:
        {context}

        Question:
        {question}

        Answer in 4–6 sentences minimum.
        """,
            input_variables=["context", "question"]
    )

    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    final_prompt = prompt.invoke({"context": context_text, "question": question})
    answer = llm.invoke(final_prompt)
    return answer.content