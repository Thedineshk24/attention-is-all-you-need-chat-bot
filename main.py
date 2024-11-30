from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain_mistralai import ChatMistralAI
import os

app = FastAPI()

# Global variable to hold the retrieval chain
retrieval_chain = None

# Load FAISS index and initialize components
@app.on_event("startup")
async def load_resources():
    global retrieval_chain
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        "faiss_index_attention",  # Path to your FAISS index
        embeddings,
        allow_dangerous_deserialization=True,
    )
    
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    
    combine_docs_chain = create_stuff_documents_chain(
        ChatMistralAI(
            api_key=os.getenv("MISTRALAI_API_KEY"),  # Ensure the MISTRALAI_API_KEY is set in your environment variables
            model_name="mistral-medium",
        ),
        retrieval_qa_chat_prompt,
    )
    
    retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)
    print("Resources loaded successfully.")

# WebSocket Endpoint to handle real-time queries
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected.")
    try:
        while True:
            query = await websocket.receive_text()  # Receive query from the client
            print(f"Received query: {query}")
            response = process_query(query)  # Process the query
            await websocket.send_text(response)  # Send the response back to the client
    except WebSocketDisconnect:
        print("Client disconnected.")

# Query processing logic: Uses LangChain's retrieval chain to process the query
def process_query(query: str) -> str:
    try:
        result = retrieval_chain.invoke({"input": query})  # Process the query with retrieval chain
        return result["answer"]  # Return the answer
    except Exception as e:
        return f"An error occurred: {str(e)}"  # Return error message if something goes wrong

# HTML Client for testing via WebSocket
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>WebSocket Client</title>
    </head>
    <body>
        <h1>LangChain WebSocket API</h1>
        <input id="queryInput" type="text" placeholder="Enter your query" />
        <button onclick="sendQuery()">Send</button>
        <ul id="messages"></ul>

        <script>
            // Open WebSocket connection to FastAPI server
            const ws = new WebSocket("ws://localhost:8000/ws");
            ws.onmessage = function(event) {
                const messages = document.getElementById('messages');
                const message = document.createElement('li');
                message.textContent = 'Response: ' + event.data;  // Display the response from server
                messages.appendChild(message);
            };

            // Function to send the input query to the WebSocket server
            function sendQuery() {
                const input = document.getElementById("queryInput");
                ws.send(input.value);  // Send the value of input box
                input.value = '';  // Clear the input box after sending
            }
        </script>
    </body>
</html>
"""

# Serve the HTML client when visiting the root endpoint
@app.get("/")
async def get():
    return HTMLResponse(html)

# To run the FastAPI app, use the following command:
# uvicorn main:app --reload
