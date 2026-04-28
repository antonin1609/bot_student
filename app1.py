import os
import base64
import streamlit as st

from rag_utils import (
    get_embedder,
    build_documents,
    create_embeddings,
    retrieve,
    format_context,
    format_sources,
)
from llm_utils import get_client, stream_answer

st.set_page_config(
    page_title="Assistant Étudiant IA",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.title-box {
    background: linear-gradient(90deg, #1f77b4, #6a5acd);
    padding: 18px 22px;
    border-radius: 16px;
    color: white;
    margin-bottom: 18px;
}
.small-note {opacity: 0.9; font-size: 0.95rem;}
.section-box {
    border: 1px solid rgba(120,120,120,0.2);
    border-radius: 14px;
    padding: 14px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

for key, default in {
    "messages": [],
    "history": [],
    "docs": None,
    "embeddings": None,
    "embedder": None,
    "indexed_signature": None,
    "last_answer": "",
    "selected_source": "Tous les documents"
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def file_signature(files):
    if not files:
        return None
    return tuple((f.name, getattr(f, "size", None)) for f in files)

def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode("utf-8")

def analyze_image_with_groq(api_key, uploaded_file, prompt):
    from groq import Groq
    client = Groq(api_key=api_key)
    encoded_image = encode_image(uploaded_file)
    image_type = uploaded_file.type or "image/png"
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_type};base64,{encoded_image}"
                        }
                    }
                ]
            }
        ],
        temperature=0.2,
        max_completion_tokens=1200
    )
    return response.choices[0].message.content.strip()

with st.sidebar:
    st.markdown("## 🎓 Assistant Étudiant IA")
    st.markdown("PDF + vision + réponses structurées.")
    st.divider()

    uploaded_files = st.file_uploader(
        "📄 Charge tes PDF",
        type=["pdf"],
        accept_multiple_files=True
    )

    uploaded_image = st.file_uploader(
        "🖼️ Charge une image",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False
    )

    mode = st.radio(
        "Mode d'assistance",
        ["Question libre", "Résumé", "Questions d'examen", "Explication", "Fiche de révision", "Vision image"],
        index=0
    )

    top_k = st.slider("Nombre d'extraits récupérés", 3, 10, 5)
    temperature = st.slider("Créativité", 0.0, 1.0, 0.2, 0.05)
    st.divider()

    if uploaded_files and mode != "Vision image":
        file_options = ["Tous les documents"] + [f.name for f in uploaded_files]
        current = st.session_state.selected_source if st.session_state.selected_source in file_options else "Tous les documents"
        st.session_state.selected_source = st.selectbox(
            "Limiter à un document",
            file_options,
            index=file_options.index(current)
        )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        index_button = st.button("⚙️ Indexer")
    with col2:
        reset_button = st.button("🗑️ Reset")

    if reset_button:
        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.docs = None
        st.session_state.embeddings = None
        st.session_state.embedder = None
        st.session_state.indexed_signature = None
        st.session_state.last_answer = ""
        st.session_state.selected_source = "Tous les documents"
        st.rerun()

st.markdown("""
<div class="title-box">
    <h2 style="margin:0;">🎓 Assistant Étudiant IA</h2>
    <p class="small-note" style="margin:0;">Réponses structurées, sources, streaming, PDF multiples et vision.</p>
</div>
""", unsafe_allow_html=True)

api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
if not api_key:
    st.warning("Ajoute la clé GROQ_API_KEY dans les secrets.")
    
if mode == "Vision image":
    st.markdown("### Analyse d'image")
    if uploaded_image:
        st.image(uploaded_image, caption=uploaded_image.name, use_container_width=True)
        vision_prompt = st.text_input(
            "Prompt pour l'image",
            value="Décris cette image clairement pour un étudiant, en donnant les éléments importants."
        )
        if st.button("Analyser l'image"):
            if not api_key:
                st.error("Clé GROQ_API_KEY manquante.")
                st.stop()
            with st.spinner("Analyse de l'image en cours..."):
                result = analyze_image_with_groq(api_key, uploaded_image, vision_prompt)
            st.markdown(result)
            st.download_button(
                "⬇️ Télécharger l'analyse",
                data=result,
                file_name="analyse_image.txt",
                mime="text/plain"
            )
    else:
        st.info("Charge une image pour lancer le mode vision.")
    st.stop()

if not uploaded_files:
    st.info("Charge tes PDF dans la barre latérale pour commencer.")
    st.stop()

sig = file_signature(uploaded_files)

if index_button or st.session_state.embeddings is None or st.session_state.indexed_signature != sig:
    prog = st.progress(0.0, text="Préparation des documents...")

    def prog_cb(v):
        prog.progress(min(1.0, max(0.0, float(v))), text="Indexation en cours...")

    with st.spinner("Lecture, découpage et indexation des PDF..."):
        st.session_state.embedder = get_embedder()
        st.session_state.docs = build_documents(uploaded_files, progress_callback=prog_cb)
        if not st.session_state.docs:
            st.error("Aucun texte exploitable trouvé dans les PDF.")
            st.stop()
        st.session_state.embeddings = create_embeddings(
            st.session_state.docs,
            st.session_state.embedder,
            progress_callback=prog_cb
        )
        st.session_state.indexed_signature = sig

    prog.empty()
    st.success(f"{len(st.session_state.docs)} extraits indexés.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

placeholder = {
    "Question libre": "Pose ta question ici...",
    "Résumé": "Résume le chapitre ou le document...",
    "Questions d'examen": "Génère des questions sur le chapitre...",
    "Explication": "Explique-moi ce concept simplement...",
    "Fiche de révision": "Crée une fiche de révision sur..."
}

user_question = st.chat_input(placeholder[mode])

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    st.session_state.history.append({"role": "user", "content": user_question})

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les documents..."):
            results = retrieve(
                user_question,
                st.session_state.docs,
                st.session_state.embeddings,
                st.session_state.embedder,
                top_k=top_k,
                source_filter=st.session_state.selected_source
            )
            context = format_context(results)
            sources = format_sources(results)

        if not api_key:
            st.error("Clé GROQ_API_KEY manquante.")
            st.stop()

        client = get_client(api_key)
        stream = stream_answer(
            client=client,
            question=user_question,
            context=context,
            history=st.session_state.history,
            mode=mode,
            temperature=temperature,
            max_completion_tokens=2000
        )

        answer = st.write_stream(stream)
        st.session_state.last_answer = answer

        if sources:
            st.caption("Sources utilisées: " + " | ".join(sources))

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.history.append({"role": "assistant", "content": answer})

if st.session_state.last_answer is not None:
    answer_text = str(st.session_state.last_answer)

    st.download_button(
        label="⬇️ Télécharger la dernière réponse",
        data=answer_text,
        file_name="reponse_assistant_etudiant.txt",
        mime="text/plain"
    )
