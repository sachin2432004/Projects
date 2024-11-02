import streamlit as st
from langchain.llms import CTransformers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
import time
from playsound import playsound
from gtts import gTTS
import os


# Initialize session state to store conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

st.title("IITM BBA ASSISTANT Chatbot")

# Prepare the template for prompting the AI
# Update the template to ensure only a clear, helpful answer is returned
template = """Use the following pieces of information to answer the user's question.
thoughly check the file and find any intent which matches then respond If you are unsure of the answer, simply say "I don't know", but do not return conflicting answers.
Context: {context}
Question: {question}
Only return the most relevant answer below and nothing else. and if you give answer don't print i don't know 
Helpful answer:
"""
# Load the language model
# @st.cache_resource


def load_llm():
    return CTransformers(model="D:\\BBA_chatbot project\\Llama-Guard-3-11B-Vision",
                         model_type='llama',
                         config={'max_new_tokens': 256, 'temperature': 0.01})

llm = load_llm()

# Load the interpreted information from the local database
# @st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda'})

embeddings = load_embeddings()

@st.cache_resource
def load_db():
    return FAISS.load_local("faiss", embeddings, allow_dangerous_deserialization=True)

db = load_db()

# Prepare a version of the llm pre-loaded with the local content
retriever = db.as_retriever(search_kwargs={'k': 2})

prompt = PromptTemplate(
    template=template,
    input_variables=['context', 'question'])

qa_llm = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type='stuff',
                                     retriever=retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs={'prompt': prompt})

# Display conversation history
# Display conversation history
for i, (q, a) in enumerate(st.session_state.conversation):
    st.text_area("Question:", value=q, height=100, max_chars=None, key=f"q_{i}", disabled=True)
    st.text_area("Answer:", value=a, height=150, max_chars=None, key=f"a_{i}", disabled=True)
    st.markdown("---")

# Get the user's question
user_question = st.text_input("Enter your question:")



if st.button('Ask'):

    if user_question:
        with st.spinner('Thinking...'):
            output = qa_llm({'query': user_question})
            answer = output["result"]

            # Add the new Q&A pair to the conversation history
            st.session_state.conversation.append((user_question, answer))

             #Convert the answer to speech using gTTS
            # tts=gTTS(answer,lang='en')
            # audio_file="response.mp3"
            # tts.save(audio_file)


            # Display the new answer with a typing effect
            full_response = ""
            response_area = st.empty()
            for i in range(len(answer) + 1):
                full_response = answer[:i]
                response_area.markdown(f"**Answer:** {full_response}▌")
                
                time.sleep(0.01)
            response_area.markdown(f"**Answer:** {full_response}")
              # Play audio in a separate thread
            # playsound(audio_file)
            

           

           

        # Clear the input box for the next question
        st.rerun()

        #remove the audio file after playing
        os.remove(audio_file)
