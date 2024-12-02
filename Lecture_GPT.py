import os
import tiktoken
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks.manager import get_openai_callback
import gradio as gr
from loguru import logger
from PyPDF2 import PdfReader
from langchain.schema import messages_from_dict, messages_to_dict
from dotenv import load_dotenv


# .env 파일 로드
load_dotenv()


# API Key
api_key = os.environ.get("OPENAI_API")

# 사전 학습 데이터 저장 경로
PRETRAINED_VECTORSTORE_PATH = "pretrained_faiss_index"

# 사전 학습 PDF 경로
PRETRAINED_PDF_PATH = r"C:\vscode_prj\waifu_v2\과달카날 전역 - 나무위키.pdf"


# 토큰 길이 계산
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


# PDF 파일 유효성 검사
def is_valid_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            if len(reader.pages) > 0:
                return True
    except Exception as e:
        logger.error(f"Invalid PDF: {e}")
    return False


# PDF 문서에서 텍스트 추출 및 문서 리스트 생성
def get_text(file_paths):
    doc_list = []

    for file_path in file_paths:
        logger.info(f"Processing document: {file_path}")

        if file_path.endswith(".pdf") and is_valid_pdf(file_path):
            loader = PyPDFLoader(file_path)
            documents = loader.load_and_split()
            doc_list.extend(documents)
        else:
            logger.error(f"Skipping invalid or unsupported PDF: {file_path}")

    return doc_list


# 텍스트 청크 생성
def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


# 벡터스토어 생성 및 저장
def create_and_save_vectorstore(documents, path):
    text_chunks = get_text_chunks(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    vectordb.save_local(path)
    logger.info(f"Vectorstore saved at {path}")
    return vectordb


# 벡터스토어 로드
def load_vectorstore(path):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    # allow_dangerous_deserialization=True 추가
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


# 대화 체인 생성
def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4", temperature=0)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="mmr", verbose=True),
        memory=memory,
        output_key="answer",
        return_source_documents=True,
        verbose=True,
    )
    return conversation_chain


# Gradio UI 동작 함수
def response_with_rag_and_history(docs, user_input, system_prompt, chat_history):
    openai_api_key = api_key
    global pretrained_vectorstore  # 사전 학습된 벡터스토어 사용
    if docs:
        documents = get_text([doc.name for doc in docs])
        if documents:
            # 사용자 PDF 데이터를 기존 벡터스토어에 통합
            user_vectorstore = create_and_save_vectorstore(
                documents, "temp_faiss_index"
            )
            pretrained_vectorstore.merge_from(user_vectorstore)

    # 대화 체인 생성
    conversation_chain = get_conversation_chain(pretrained_vectorstore, openai_api_key)

    # 시스템 프롬프트와 사용자 입력 추가
    inputs = {"question": user_input, "chat_history": chat_history}
    if system_prompt:
        inputs["system_prompt"] = system_prompt

    # OpenAI API 호출 및 응답
    try:
        with get_openai_callback() as callback:
            response = conversation_chain(inputs)
            assistant_response = response["answer"]

            # Gradio Chatbot 요구사항에 맞게 chat_history 업데이트
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": assistant_response})

            return assistant_response, chat_history
    except Exception as e:
        logger.error(f"Error during OpenAI API call: {e}")
        return str(e), chat_history


# Gradio UI 설정
with gr.Blocks() as demo:
    gr.Markdown("## RAG 챗봇 (사전 학습 + 사용자 추가 PDF 분석)")

    # PDF 업로드
    pdf_input = gr.File(
        label="PDF 파일 업로드 (선택)", file_types=[".pdf"], file_count="multiple"
    )
    system_prompt = gr.Textbox(
        label="System Prompt", placeholder="시스템 프롬프트는 선택사항입니다."
    )
    user_input = gr.Textbox(label="질문 입력", placeholder="질문을 입력하세요.")
    chatbot = gr.Chatbot(label="대화 기록", elem_id="chat_history", type="messages")
    state = gr.State([])  # 대화 기록을 저장할 상태
    output = gr.Textbox(label="응답", lines=5)

    # 버튼
    generate_btn = gr.Button("응답 생성")
    generate_btn.click(
        response_with_rag_and_history,
        inputs=[pdf_input, user_input, system_prompt, state],
        outputs=[output, chatbot],
    )

# 사전 학습된 벡터스토어 로드
if os.path.exists(PRETRAINED_VECTORSTORE_PATH):
    pretrained_vectorstore = load_vectorstore(PRETRAINED_VECTORSTORE_PATH)
else:
    # 사전 학습 데이터를 생성하고 저장 (초기 실행 시 필요)
    logger.info(f"Creating vectorstore from {PRETRAINED_PDF_PATH}")
    pretrained_documents = get_text([PRETRAINED_PDF_PATH])  # 경로 문자열로 전달
    pretrained_vectorstore = create_and_save_vectorstore(
        pretrained_documents, PRETRAINED_VECTORSTORE_PATH
    )

demo.launch()
