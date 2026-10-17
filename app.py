import streamlit as st
import plotly.express as px
import pandas as pd
from utils.pdf_processor import prepare_documents_for_rag, pages_to_chunks
from utils.rag_engine import build_vector_store, ask_nexus, generate_document_summary

st.set_page_config(
    page_title="Nexus — Multi-Document AI",
    page_icon="🔮",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0a0a0f; }
    [data-testid="stSidebar"] {
        background-color: #12121e;
        border-right: 1px solid #2a2a4a;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #12121e 100%);
        border: 1px solid #4a4a8a;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricValue"] {
        color: #6c63ff !important;
        font-weight: 700;
        font-size: 1.4rem;
    }
    [data-testid="stMetricLabel"] { color: #8888cc !important; }
    h1 {
        color: #6c63ff !important;
        font-weight: 800 !important;
        text-shadow: 0 0 30px rgba(108,99,255,0.4);
    }
    h2, h3 { color: #e2e2f0 !important; }
    h4 { color: #c0c0e0 !important; }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #12121e;
        border-bottom: 1px solid #2a2a4a;
    }
    .stTabs [data-baseweb="tab"] { color: #6666aa; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #6c63ff !important;
        border-bottom: 2px solid #6c63ff !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1a1a2e 0%, #2a2a4a 100%);
        color: #6c63ff;
        border: 1px solid #4a4a8a;
        border-radius: 8px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #6c63ff 0%, #5a52dd 100%);
        color: white;
        transform: translateY(-1px);
    }
    [data-testid="stChatInput"] {
        background: #1a1a2e;
        border: 1px solid #4a4a8a;
        border-radius: 12px;
    }
    [data-testid="stChatMessage"] {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        margin-bottom: 8px;
    }
    hr { border-color: #2a2a4a; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0a0f; }
    ::-webkit-scrollbar-thumb { background: #4a4a8a; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #6c63ff; }
</style>
""", unsafe_allow_html=True)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_summaries" not in st.session_state:
    st.session_state.doc_summaries = {}
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0

with st.sidebar:
    st.markdown("## 🔮 Nexus")
    st.caption("Multi-Document AI Intelligence")
    st.divider()

    st.subheader("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload multiple PDFs to analyze together"
    )

    if uploaded_files:
        if st.button("🚀 Process Documents", use_container_width=True):
            with st.spinner("Processing documents..."):
                pages, doc_summaries = prepare_documents_for_rag(uploaded_files)
                chunks, metadata = pages_to_chunks(pages)
                vector_store = build_vector_store(chunks, metadata)

                st.session_state.vector_store = vector_store
                st.session_state.doc_summaries = doc_summaries
                st.session_state.doc_names = [f.name for f in uploaded_files]
                st.session_state.total_chunks = len(chunks)
                st.session_state.total_pages = len(pages)

                st.success(f"✅ Processed {len(uploaded_files)} documents!")

    st.divider()
    st.subheader("💡 Try asking")
    example_questions = [
        "What is the revenue for this year?",
        "Summarize the key highlights",
        "What are the major risks mentioned?",
        "Compare both companies performance",
        "What is the employee headcount?",
        "What are the strategic priorities?",
        "How did profits change year over year?",
        "What technology investments were made?",
    ]

    for q in example_questions:
        if st.button(q, use_container_width=True, key=q):
            st.session_state.pending_question = q

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown("""
<div style='padding: 1rem 0 0.5rem 0;'>
    <h1 style='color: #6c63ff; font-size: 2.2rem; font-weight: 800; margin: 0;
    text-shadow: 0 0 30px rgba(108,99,255,0.4);'>
        🔮 Nexus
    </h1>
    <p style='color: #6666aa; font-size: 0.95rem; margin: 6px 0 0 0; font-style: italic;'>
        Multi-Document AI Intelligence — Ask across any document, get grounded answers
    </p>
</div>
""", unsafe_allow_html=True)

if st.session_state.vector_store is None:
    st.info("👈 Upload your PDFs and click Process Documents to get started")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 🔮 What Nexus can do
        - Answer questions across multiple PDFs
        - Compare documents side by side
        - Extract key metrics and figures
        - Cite exact sources and page numbers
        """)
    with col2:
        st.markdown("""
        ### 📁 Supported files
        - Annual reports
        - Research papers
        - Legal contracts
        - Technical manuals
        - Financial statements
        """)
    with col3:
        st.markdown("""
        ### 🤖 Powered by
        - RAG (Retrieval Augmented Generation)
        - FAISS vector similarity search
        - Sentence Transformers embeddings
        - Groq LLM for grounded answers
        """)
else:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents", len(st.session_state.doc_names))
    col2.metric("Pages Processed", st.session_state.total_pages)
    col3.metric("Chunks Indexed", st.session_state.total_chunks)
    col4.metric("Embedding Model", "MiniLM-L6")

    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "🤖 AI Chat", "📋 Document Summaries", "📊 Document Stats"
    ])

    with tab1:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    st.caption(f"📍 Sources: {', '.join(msg['sources'])}")

        question = st.chat_input("Ask anything across your documents...")

        if "pending_question" in st.session_state:
            question = st.session_state.pending_question
            del st.session_state.pending_question

        if question and st.session_state.vector_store:
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching across documents..."):
                    answer, sources = ask_nexus(
                        st.session_state.vector_store,
                        question,
                        st.session_state.doc_names,
                        st.session_state.messages
                    )
                    st.markdown(answer)
                    if sources:
                        st.caption(f"📍 Sources: {', '.join(sources)}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

    with tab2:
        if st.session_state.doc_summaries:
            for doc_name, info in st.session_state.doc_summaries.items():
                with st.expander(f"📄 {doc_name} — {info['pages']} pages, {info['size_mb']} MB"):
                    if st.button(f"Generate AI Summary for {doc_name}", key=f"sum_{doc_name}"):
                        with st.spinner("Generating summary..."):
                            summary = generate_document_summary(
                                st.session_state.vector_store,
                                doc_name
                            )
                            st.markdown(summary)
                    st.markdown("**Preview:**")
                    st.text(info['preview'][:500] + "...")

    with tab3:
        if st.session_state.doc_summaries:
            st.markdown("#### Document Overview")

            doc_data = pd.DataFrame([
                {
                    'Document': name,
                    'Pages': info['pages'],
                    'Size (MB)': info['size_mb'],
                }
                for name, info in st.session_state.doc_summaries.items()
            ])
            st.dataframe(doc_data, use_container_width=True, hide_index=True)

            col_a, col_b = st.columns(2)
            with col_a:
                fig1 = px.bar(
                    doc_data,
                    x='Document',
                    y='Pages',
                    title="Pages per Document",
                    color='Pages',
                    color_continuous_scale=['#4a4a8a', '#6c63ff'],
                )
                fig1.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    coloraxis_showscale=False,
                    height=350
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col_b:
                fig2 = px.pie(
                    doc_data,
                    names='Document',
                    values='Pages',
                    title="Page Distribution",
                    hole=0.4,
                    color_discrete_sequence=['#6c63ff', '#4a4a8a']
                )
                fig2.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=350
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("#### Processing Stats")
            col_c, col_d, col_e = st.columns(3)
            col_c.metric("Total Pages", st.session_state.total_pages)
            col_d.metric("Total Chunks", st.session_state.total_chunks)
            col_e.metric(
                "Avg Chunk Size",
                f"~800 tokens"
            )