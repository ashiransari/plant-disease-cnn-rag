import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import chainlit as cl
from langchain_groq import ChatGroq


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

import torchvision.models as models
from torch.serialization import add_safe_globals
add_safe_globals({models.ResNet: models.ResNet})

# === Paths ===
MODEL_PATH = r"C:\Users\91797\Downloads\Chatbot1245\Chatbot1245\resnet18.pt"
PDF_PATH = r"C:\Users\91797\Downloads\Chatbot1245\Chatbot1245\Plant Diseases.pdf"

# === Class Labels ===
DISEASE_CLASSES = {
    0: "Grape Leaf Blight",
    1: "Grape Black Measles",
    2: "Grape Black Rot",
    3: "Grape Healthy",
    4: "Rice Healthy",
    5: "Tomato Early Blight",
    6: "Tomato Late Blight",
    7: "Tomato Leaf Mold",
    8: "Tomato Healthy",
    9: "Rice Bacterial Leaf Blight",
    10: "Rice Leaf Blast",
    11: "Rice Sheath Blight"
}


# === Load Model ===
def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = torch.load(path, map_location=device, weights_only=False)
        model.eval()
        return model.to(device), True
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, False

model, model_loaded = load_model(MODEL_PATH)

# === Image Preprocessing ===
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_disease(image_path):
    if not model_loaded:
        return "Unknown", 0.0
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
            print(pred)
        return DISEASE_CLASSES.get(pred, "Unknown Disease"), confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Unknown", 0.0

# === Load PDF as Vector Store ===
def load_pdf_data():
    try:
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        print(f"PDF loading error: {e}")
        return None

@cl.on_chat_start
async def start_chat():
    await cl.Message("🌱 Welcome! Upload a leaf image or ask about plant diseases.").send()
    vectorstore = load_pdf_data()

    if not vectorstore:
        await cl.Message("⚠️ Failed to load plant data. Chat might be limited.").send()
        cl.user_session.set("chain", None)
        return

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt = PromptTemplate(
        input_variables=["context", "question", "disease"],
        template=""" 
        The predicted disease is: {disease}
        You are a helpful assistant for plant disease diagnosis.

        Context:
        {context}

        Question:
        {question}
        """
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        api_key='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'
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=prompt,
        return_source_documents=True
    )

    cl.user_session.set("chain", chain)
    cl.user_session.set("chat_history", [])

@cl.on_message
async def handle_query(message: cl.Message):
    chain = cl.user_session.get("chain")
    history = cl.user_session.get("chat_history")

    question = message.content
    print(f"Question: {message.content}")

    print(f"Image Path: {message.elements[0].path}")

    name, conf = predict_disease(message.elements[0].path)
    print(f"Predicted Disease is {name} with Confidence {conf}")

    result = chain.invoke({
            "disease": name,
            "question": question,
            "chat_history": history
        })

    print(result)

    answer = result["answer"]
    history.append((message.content, answer))
    cl.user_session.set("chat_history", history)

    await cl.Message(answer).send()