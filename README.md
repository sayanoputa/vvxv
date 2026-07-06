## install
pip install streamlit, langchain_community, langchain_text_splitters, langchain_chroma, langchain_ollama, langchain_core, langchain_core

# config
DATA_PATH = "data/{filename}" \
DB_PATH = VECTORSTORE \
EMBEDDING_MODEL = "nomic-embed-text" \
LLM_MODEL = "llama3.1" (can use 3.0 if installed) \
CHUNK_SIZE = 800 \
CHUNK_OVERLAP = 120 \

# ingest
def load_doc(): \
    loader = TextLoader(DATA_PATH, encoding="utf-8") \
    docs = loader.load \
    return docs 

def split_doc(docs): \
    splitter = RecursiveCharacterTextSplitter(chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP) \
    chunks = splitter.split_documents(docs) \
    return chunks 

def create_vector_database(chunks): \
    if(os.path_exists(DB_PATH)): \
        shutil.rmtree(DB_PATH) \
    embeddings = OllamaEmbeddings(model = EMBEDDING_MODEL) \
    vectorstore = Chroma.from_documents(documents = chunks, embedding = embeddings, persist_directory = DB_PATH) \
    return vectorstore 

def main(): \
    if not os.path.exists(DATA_PATH): \
        return \
    create_vector_database(split_doc(load_doc)) 

# rag_chain
def format_docs(docs): \
    res = [] \
    for index, doc in enumerate(docs, start=1): \
        res.append(f"{doc.page_content}\n - source {index}") \
    return res 

def get_vectorstore(): \
    embeddings = OllamaEmbeddings(model = EMBEDDING_MODEL) \
    return Chroma(persist_directory = DB_PATH, embedding_function = embeddings) 

def get_prompt_template(): \
    template = """context: {context}, question: {question} """ \
    return ChatPromptTemplate.from_template(template) 

def answer_question(question): \
    retriever = get_vectorstore.as_retriever(search_kwargs = {"k" = 4 }) \
    context = format_documents(retriever.invoke(question)) \ 
    llm = ChatOllama(model = LLM_MODEL, temperature = 0.2) \
    chain = prompt | llm | StrOutputParser() \
    return chain.invoke({"context": context, "question": question})
    
# app
st.set_page_config(page_title = "Chatbot", page_icon = "📄", layout = "centered") 

st.title("simple chatbot") 

st.write("enter question")

if not os.path.exists(DB_PATH): \
    st.warn("no DB. run ingest")

question = st.text_area("question", height = 120, placeholder = " ")

ask_button = st.button("ask")

if ask_button: \
    answer = answer_question(question) \
    st.write(answer)

