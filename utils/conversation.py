from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

    
def converse(docsearch,temp_val,OPENAI_API_KEY,memory,question):
    retriever = docsearch.as_retriever(search_kwargs={'k': 10})

    prompt_template = """
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    You can also greet the customer, or talk in general.
    Helpful answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}

    llm = ChatOpenAI(model="gpt-4o", temperature=temp_val, max_tokens=4096, api_key=OPENAI_API_KEY)

    qa = RetrievalQA.from_chain_type(llm=llm,memory=memory, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)

    response = qa({"query": question})
    return response