---
title: AI Policy Reader
emoji: 😎
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "4.32.0"
app_file: app.py
pinned: false
---

## <h1 align="center" id="heading">🔍AI Policy 2024</h1>

This is the midterm assignment of the AI Engineering Bootcamp from [AI Makerspace](https://aimakerspace.io/). The aim of the project is to help the leadership of the company understanding how the AI industry is evolving, especially as it relates to politics, as an AI solution engineer. The main context of the problem is in these two documents:
- [2022: Blueprint for an AI Bill of Rights: Making Automated Systems Work for the American People (PDF)](https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf)
- [2024: National Institute of Standards and Technology (NIST) Artificial Intelligent Risk Management Framework (PDF)](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)

The following mind map summarizes the tasks and questions that will be addressed through the development of this app. Details will be discussed below.
<p align = "center" draggable=”false” ><img src="https://github.com/Zhiji022/ai-policy-read/blob/main/images/mindmap.png"
     width="auto"
     height="auto"/>
</p>

### Overview
A simple retrieval augmented generation application with the above two documents as contexts were built and deployed, with the option of additional information as a url or uploading a pdf file. Two chunking strategies were tested. A finetuned small embedding model was compared with a larger base embedding model. The performance of the app was evaluated using metrices including faithfulness, relevancy, context precision and context recall under the ragas framework.

### Chunking strategy

#### Base chunking strategy
While dealing with pdf file, the default chunking method was selected as pymupdf loader with recursive character splitting. Chunk size is user defined and the boundary of the text is handled by the default stop signs. 
- Pros: fast and easy
- Cons: logic of the document structure is not retained; sentences and paragraphs can be cut abruptly; artifacts like tables and images are not handled properly.

#### Enhanced chunking
To overcome the cons of the default chunking, an enhanced method is proposed and implemented. By investigating the documents, a combination of three methods will be used to extract different parts of the documents.
- Text: markdown loader
- Table: pdfplumber llm 
- html: beautiful soup loader
By loading and chunking text as markdown, the nature structure of the document is preserved and easily detectable. Thus, the logic and of the text boundary is much closer to what it originally intended. Since the tables in the document have clear and unified pattern, the parser successfully extracted the information without being distorted by the format. The bs4 loader is able to load the html file which is provided by the user by implementing the Beautiful Soup library.
Pros: texts are more logically chunked and information are better groupped and preserved.
Cons: some manual cleaning is required

#### Finetuning embedding model
Two embedding models were tested. The snowflake-arctic-embed-m-v1.5 as base and the fine tuned snowflake-arctic-embed-xs. The selection is based on the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) on retrieval task. The model was fine tuned using a synthetically generated dataset.

#### Evaluation
Combination of the chunking strategies and models were evaluated using ragas framework. Here are the metrics to be presented:
- faithfulness: diviation of answers from to the context
- answer relevancy: answers relevant to the question
- context precision: most agreed context to the ground truth are ranked high
- context recall: alignment of the context with the ground truth
For detail, please refer to the [ragas documentation](https://docs.ragas.io/en/stable/concepts/metrics/index.html)

And here is the evaluation:
| **Chunking** | **Model** | **Faithfulness** | **Answer relevence** | **Context recall** | **Context precision** | 
|:------------:|:---------:|:----------------:|:--------------------:|:------------------:|:---------------------:|
| default      | base      | 0.8049           | 0.8946               | 0.6981             | 0.6903                |
| advanced     | base      | 0.7227           | 0.9565               | 0.7870             | 0.8539                |
| default      | finetuned | 0.9316           | 0.9501               | 0.8972             | 0.9273                |
| advanced     | fintuned  | 0.8106           | 0.9589               | 0.8565             | 0.9106                |

When base model is used, advanced chunking boosted all metrics except the faithfulness. When the default chunking method is used, finetuning the model significantly helped with all metrics. However, it is surprising that the combination of default chunking and finetuned model combination beats the advanced chunking and finetuned model combination. While there is not enough information to make a conclusion, it is obvious that a tiny finetuned model easily out performs a base foundation model. 

#### Managing expection
- What is the story that you will give to the CEO to tell the whole company at the launch next month?
To start, I will present some questions and answers from the app such as:
Q: What are some examples of known issues that should be reported in GAI systems?
A: Harmful Bias and Homogenization; Dangerous, Violent, or Hateful Content; Obscene, Degrading, and/or Abusive Content; Confabulation; Information Security Incidents; Inappropriate or Harmful Content Generation; Errors and Near-Misses
Then, I will invite the CEO and other leaders to ask their questions and concerns and use the bot to get answers and contexts

- There appears to be important information not included in our build, for instance, the 270-day update on the 2023 executive order on Safe, Secure, and Trustworthy AI.  How might you incorporate relevant white-house briefing information into future versions? 
Users will be prompted to provide additional information in the form of URL or uploading a pdf file. The app will process the those information in the back and add them to the context.