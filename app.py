import streamlit as st
from rag_chain import load_rag_chain

st.set_page_config(
    page_title="Legal Document Assistant",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ Legal Document Assistant")
st.caption("Ask anything about the Bharatiya Nyaya Sanhita (BNS)")
st.divider()

# Load RAG chain only once
if "rag_chain" not in st.session_state:
    with st.spinner("Loading AI model... please wait"):
        st.session_state.rag_chain = load_rag_chain()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if question := st.chat_input("Ask a legal question..."):

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.rag_chain.invoke({"input": question})
            answer = response["answer"]
            st.markdown(answer)

            with st.expander("📄 View Source Sections Used"):
                for i, doc in enumerate(response["context"]):
                    st.write(f"**Source Chunk {i+1}:**")
                    st.write(doc.page_content[:400])
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})