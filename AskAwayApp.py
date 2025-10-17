import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import re

load_dotenv()
# Function to extract the video id from provided URL
# There are two tpes of URLs, long and short, but the video_id will be the same 11 characters of
# uppercase, lowercase, numericals, hyphens, or underscores.
# In the long format, the video_id is preceeded by either "v=" or "/" and in short format by ".be/"
#
# Streamlit UI
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="centered",
    initial_sidebar_state="collapsed",
)
#
# Dark Page theme Styling
#
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: white;
    font-family: "Segoe UI", sans-serif;
}

/* Hide 'Press Enter to submit' text */
div[data-testid="stTextInput"] > div > div > div > div:nth-child(2) {
    display: none !important;
}

/* Make input boxes dark */
[data-baseweb="input"] > div {
    background-color: #1E1E1E !important;
    color: white !important;
    border: 1px solid #444 !important;
    border-radius: 8px !important;
}

/* Input placeholder color */
[data-baseweb="input"] input::placeholder {
    color: #888 !important;
}

/* Stylish answer box */
.answer-box {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 0px 8px rgba(255,255,255,0.05);
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)
#
# Header
st.markdown("<h1 style='text-align:center;'>🎬 YouTube RAG Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Ask any question about a YouTube video transcript!</p>", unsafe_allow_html=True)
#
# Functions
#
def extract_video_id(url):
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",  # pattern for long URL
        r"youtu\.be\/([0-9A-Za-z_-]{11})"   # pattern for short URL
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)   # group(1) contains the video_id
    return None

# Function to fetch and clean transcript from the youtube-transcript-api
#
def get_transcript(video_id):
    YTapi = YouTubeTranscriptApi()  # Creating an instance of the API
    try:
        transcript_list = YTapi.fetch(video_id, languages = ["en"])
        #
        # The API provides results in a FetchecTranscriptSnippet object, in the following format:
        #snippets=[
            # FetchedTranscriptSnippet(
            #     text="Hey there",
            #     start=0.0,
            #     duration=1.54,
            # ),
            # FetchedTranscriptSnippet(
            #     text="how are you",
            #     start=1.54,
            #     duration=4.16,)
        #
        # The resulting transcript will be provided on the basis of timestamps, (when the text starts and till what duration it is maintained.)
        # The API gets the transcript sentence by sentence in a broken format, so we need to join it into a single plain text.
        
        # Join text including the verbal cues like [Music], [Applause]
        #
        raw_transcript = " ".join(chunk.text for chunk in transcript_list)
        
        # Clean the text by removing these non-verbal cues using regex functions
        #
        clean_transcript = re.sub(r"\[.*?\]", "", raw_transcript)
        clean_transcript = re.sub(r"\s+", " ", clean_transcript).strip()
        return clean_transcript

    except TranscriptsDisabled:
        return None
    
# Input of URL and user's question
with st.form("youtube_form"):
    youtube_url = st.text_input("🎬 Enter YouTube URL", placeholder="Paste link here")
    user_question = st.text_input("💬 Your Question", placeholder="Ask anything")
    submit_button = st.form_submit_button("Ask AI 🎯")

if submit_button:
        
    if youtube_url and user_question:
        video_id = extract_video_id(youtube_url)

        if not video_id:
            st.error("Invalid YouTube URL!")
        else:
            with st.spinner("Getting the Transcripts..."):
                transcript = get_transcript(video_id)
            
            if not transcript:
                st.error("No Transcript available for this video.")
            else:
                splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
                chunks = splitter.create_documents([transcript])

                #using the following embedding model to create embeddings of the question and the transcript
                embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
                #we provide our chunks along with our embedding model to the vector store to create an embedding for each chunk
                vector_store = FAISS.from_documents(chunks, embeddings)
                # retrieve using our vector store, that will use a simple similarity search and get 4 most similar results in the output.
                retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k": 4})
                # Creating a prompt template with user query and retrieved context(documents) to be provided to the LLM
                #
                prompt = PromptTemplate(
                    template = """
                        You are a helpful assistant.
                        Answer the Question ONLY from the provided transcript context.
                        If the context is insufficient, just say you do not know.
                        
                        Context:
                        {context}

                        Question: {question}
                    """,
                    input_variables= ['context', 'question']
                )
                # Format retrieved docs to create the complete context
                def format_docs(retrieved_docs):
                    return " ".join(doc.page_content for doc in retrieved_docs)
                
                parallel_chain = RunnableParallel({
                    'context': retriever | RunnableLambda(format_docs),
                    'question': RunnablePassthrough()
                })
                # Forming an LLM
                #
                llm = ChatOpenAI(model = "gpt-4o-mini", temperature=0.2, streaming= True)
                parser = StrOutputParser()
                main_chain = parallel_chain | prompt | llm | parser
                
                st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
                st.write("**Answer:**")

                    # Stream the answer gradually
                answer_container = st.empty()
                answer_text = ""
                for chunk in main_chain.stream(user_question):
                     answer_text += chunk
                     answer_container.markdown(answer_text)
                st.markdown("</div>", unsafe_allow_html=True)