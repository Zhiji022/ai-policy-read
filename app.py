from langchain_core.prompts import ChatPromptTemplate
import chainlit as cl

from utils.prompts import RAG_PROMPT
from utils.vector_store import get_default_documents, get_vector_store, process_uploaded_file, process_webpage
# from utils.advanced_chunking import get_enhanced_documents
from utils.models import FINE_TUNED_EMBEDDING, RAG_LLM
from utils.rag import RAGRunnables, create_rag_chain

from urllib.request import urlopen
import tempfile


welcome_message = """Hi, I am your AI-policy assistant. I can help you understand how the AI industry is evolving, especially as it relates to politics.
My answers will be based on the following two documents:
1. 2024: National Institute of Standards and Technology (NIST) Artificial Intelligent Risk Management Framework (PDF)
2. 2022: Blueprint for an AI Bill of Rights: Making Automated Systems Work for the American People (PDF)\n
If you need help with more updated information, upload a pdf file or provide a URL now.
"""

@cl.on_chat_start
async def start():
    
    # ask new document
    res = await cl.AskActionMessage(content=welcome_message,
                                    actions=[cl.Action(name="upload", value="upload", label="📄Upload"),
                                            cl.Action(name="url", value="url", label="🛜URL"),
                                            cl.Action(name="continue", value="continue", label="🤷🏻‍♀️Continue")]
                                    ).send()
    new_doc = None
    web_doc = None
    
    if res and res.get("value") == "continue": 
        pass
    
    elif res and res.get("value")=="url":
        
        url = await cl.AskUserMessage(content="Please provide a URL", timeout=30).send()
        print(url)
        
        try:
            
            with urlopen(url['content']) as webpage:
                web_content = webpage.read()
                
            with tempfile.NamedTemporaryFile('w', suffix = '.html') as temp:
                temp.write(web_content.decode())
                temp.seek(0)
                web_doc = process_webpage(temp.name)
           
            await cl.Message(content="New information accepted✅").send()
        
        except:
            
            await cl.Message(content="Invalid URL. Skipping new info...🚩").send()
    
    elif res and res.get("value") == "upload":
        files = await cl.AskFileMessage(
            content="Please upload a pdf file to begin!",
            accept=["application/pdf"],
            max_size_mb=4,
            timeout=90,
        ).send()
        file = files[0]

        msg = cl.Message(content=f"Processing `{file.name}`...", disable_human_feedback=True)
        await msg.send()
        
        # process new document
        new_doc = process_uploaded_file(file)
    
    # process documents
    documents = get_default_documents()
    
    if new_doc:
        documents.extend(new_doc)
    elif web_doc:
        documents.extend(web_doc)
    else:
        pass
    
    # create rag chain
    rag_runnables = RAGRunnables(
                        rag_prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT),
                        vector_store = get_vector_store(documents, FINE_TUNED_EMBEDDING, emb_dim=384),
                        llm = RAG_LLM
                    )
    rag_chain = create_rag_chain(rag_runnables.rag_prompt_template, 
                                 rag_runnables.vector_store, 
                                 rag_runnables.llm)
    
    cl.user_session.set('chain', rag_chain)

@cl.on_message    
async def main(message):
    chain = cl.user_session.get("chain")

    # msg = cl.Message(content="")
    result = await chain.ainvoke({'question': message.content})

    answer = result['response']
    
    source_documents = result['context']  # type: List[Document]
    text_elements = []
    
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            
            # Create the text element referenced in the message   
            source_name = f"source - {source_idx}"           
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"
    

    await cl.Message(content=answer, elements=text_elements).send()