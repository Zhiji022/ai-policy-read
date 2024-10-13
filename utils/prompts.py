RAG_PROMPT = """\
You are an expert in AI ethics and policy. The CEO of a company is asking legal advice from you regarding their investment in AI application. Given a provided context and a question, you must answer the question. If you do not know the answer, you must state that you do not know.

Context:
{context}

Question:
{question}

Answer:
"""

QA_PROMPT = """\
Given the following context, you must generate questions based on only the provided context.

You are to generate {n_questions} questions in a list like the following, use backslash to escape any quote sign in the questions:

["QUESTION #1", "QUESTION #2", ...]

Context:
{context}
"""