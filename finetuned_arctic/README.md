---
base_model: Snowflake/snowflake-arctic-embed-xs
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
- dot_accuracy@1
- dot_accuracy@3
- dot_accuracy@5
- dot_accuracy@10
- dot_precision@1
- dot_precision@3
- dot_precision@5
- dot_precision@10
- dot_recall@1
- dot_recall@3
- dot_recall@5
- dot_recall@10
- dot_ndcg@10
- dot_mrr@10
- dot_map@100
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:2730
- loss:MatryoshkaLoss
- loss:MultipleNegativesRankingLoss
widget:
- source_sentence: What steps can be taken to mitigate the risks associated with GAI
    systems?
  sentences:
  - 'Action ID: GV-4.3-003

    Suggested Action: Verify information sharing and feedback mechanisms among individuals
    and

    organizations regarding any negative impact from GAI systems.

    GAI Risks: Information Integrity; Data

    Privacy'
  - '48. The definitions of ''equity'' and ''underserved communities'' can be found
    in the Definitions section of this framework as well as in Section 2 of The Executive
    Order On Advancing Racial Equity and Support [for Underserved Communities Through
    the Federal Government. https://www.whitehouse.gov/](https://www.whitehouse.gov)
    briefing-room/presidential-actions/2021/01/20/executive-order-advancing-racial-equity-and-support­
    for-underserved-communities-through-the-federal-government/


    49. Id.'
  - 'Action ID: GV-6.1-001

    Suggested Action: Categorize different types of GAI content with associated third-party
    rights (e.g.,

    copyright, intellectual property, data privacy).

    GAI Risks: Data Privacy; Intellectual

    Property; Value Chain and

    Component Integration'
- source_sentence: What tasks are associated with AI Actor governance and oversight?
  sentences:
  - 'GOVERN 1.1: Legal and regulatory requirements involving AI are understood, managed,
    and documented.: MANAGE 4.2: Measurable activities for continual improvements
    are integrated into AI system updates and include regular

    engagement with interested parties, including relevant AI Actors.

    AI Actor Tasks: Governance and Oversight: AI Actor Tasks: AI Deployment, AI Design,
    AI Development, Affected Individuals and Communities, End-Users, Operation and

    Monitoring, TEVV'
  - "Beyond harms from information exposure (such as extortion or dignitary harm),\
    \ wrong or inappropriate inferences of PII can contribute to downstream or secondary\
    \ harmful impacts. For example, predictive inferences made by GAI models based\
    \ on PII or protected attributes can contribute to adverse decisions, leading\
    \ to representational or allocative harms to individuals or groups (see Harmful\
    \ Bias and Homogenization below).  #### Trustworthy AI Characteristics: Accountable\
    \ and Transparent, Privacy Enhanced, Safe, Secure and Resilient\n\n 2.5. Environmental\
    \ Impacts\n\n Training, maintaining, and operating (running inference on) GAI\
    \ systems are resource-intensive activities, with potentially large energy and\
    \ environmental footprints. Energy and carbon emissions vary based on what is\
    \ being done with the GAI model (i.e., pre-training, fine-tuning, inference),\
    \ the modality of the content, hardware used, and type of task or application.\n\
    \n Current estimates suggest that training a single transformer LLM can emit as\
    \ much carbon as 300 round- trip flights between San Francisco and New York. In\
    \ a study comparing energy consumption and carbon emissions for LLM inference,\
    \ generative tasks (e.g., text summarization) were found to be more energy- and\
    \ carbon-intensive than discriminative or non-generative tasks (e.g., text classification).\
    \ \n\n Methods for creating smaller versions of trained models, such as model\
    \ distillation or compression, could reduce environmental impacts at inference\
    \ time, but training and tuning such models may still contribute to their environmental\
    \ impacts. Currently there is no agreed upon method to estimate environmental\
    \ impacts from GAI. \n\n Trustworthy AI Characteristics: Accountable and Transparent,\
    \ Safe\n\n 2.6. Harmful Bias and Homogenization"
  - "#### • Accessibility and reasonable accommodations\n\n • AI actor credentials\
    \ and qualifications  • Alignment to organizational values\n\n#### • Auditing\
    \ and assessment  • Change-management controls  • Commercial use  • Data provenance\
    \ #### • Data protection  • Data retention  • Consistency in use of defining key\
    \ terms  • Decommissioning  • Discouraging anonymous use  • Education  • Impact\
    \ assessments  • Incident response  • Monitoring  • Opt-outs\n\n#### • Risk-based\
    \ controls  • Risk mapping and measurement  • Science-backed TEVV practices  •\
    \ Secure software development practices  • Stakeholder engagement  • Synthetic\
    \ content detection and labeling tools and techniques\n\n • Whistleblower protections\
    \  • Workforce diversity and interdisciplinary teams\n\n#### Establishing acceptable\
    \ use policies and guidance for the use of GAI in formal human-AI teaming settings\
    \ as well as different levels of human-AI configurations can help to decrease\
    \ risks arising from misuse, abuse, inappropriate repurpose, and misalignment\
    \ between systems and users. These practices are just one example of adapting\
    \ existing governance protocols for GAI contexts. \n\n A.1.3. Third-Party Considerations\n\
    \n Organizations may seek to acquire, embed, incorporate, or use open-source or\
    \ proprietary third-party GAI models, systems, or generated data for various applications\
    \ across an enterprise. Use of these GAI tools and inputs has implications for\
    \ all functions of the organization – including but not limited to acquisition,\
    \ human resources, legal, compliance, and IT services – regardless of whether\
    \ they are carried out by employees or third parties. Many of the actions cited\
    \ above are relevant and options for addressing third-party considerations."
- source_sentence: What specific topic is covered in Chapter 3 of the AI Risk Management
    Framework by NIST?
  sentences:
  - "Liang, W. et al. (2023) GPT detectors are biased against non-native English writers.\
    \ arXiv. https://arxiv.org/abs/2304.02819\n\n Luccioni, A. et al. (2023) Power\
    \ Hungry Processing: Watts Driving the Cost of AI Deployment? arXiv. https://arxiv.org/pdf/2311.16863\n\
    \n Mouton, C. et al. (2024) The Operational Risks of AI in Large-Scale Biological\
    \ Attacks. RAND. https://www.rand.org/pubs/research_reports/RRA2977-2.html.\n\n\
    \ Nicoletti, L. et al. (2023) Humans Are Biased. Generative Ai Is Even Worse.\
    \ Bloomberg. https://www.bloomberg.com/graphics/2023-generative-ai-bias/.\n\n\
    \ National Institute of Standards and Technology (2024) Adversarial Machine Learning:\
    \ A Taxonomy and Terminology of Attacks and Mitigations https://csrc.nist.gov/pubs/ai/100/2/e2023/final\n\
    \n National Institute of Standards and Technology (2023) AI Risk Management Framework.\
    \ https://www.nist.gov/itl/ai-risk-management-framework\n\n National Institute\
    \ of Standards and Technology (2023) AI Risk Management Framework, Chapter 3:\
    \ AI Risks and Trustworthiness. https://airc.nist.gov/AI_RMF_Knowledge_Base/AI_RMF/Foundational_Information/3-sec-characteristics\n\
    \n National Institute of Standards and Technology (2023) AI Risk Management Framework,\
    \ Chapter 6: AI RMF Profiles. https://airc.nist.gov/AI_RMF_Knowledge_Base/AI_RMF/Core_And_Profiles/6-sec-profile"
  - "###### WHAT SHOULD BE EXPECTED OF AUTOMATED SYSTEMS\n\n The expectations for\
    \ automated systems are meant to serve as a blueprint for the development of additional\
    \ technical standards and practices that are tailored for particular sectors and\
    \ contexts.\n\n**Equitable.** Consideration should be given to ensuring outcomes\
    \ of the fallback and escalation system are equitable when compared to those of\
    \ the automated system and such that the fallback and escalation system provides\
    \ equitable access to underserved communities.[105]\n\n**Timely. Human consideration\
    \ and fallback are only useful if they are conducted and concluded in a** timely\
    \ manner. The determination of what is timely should be made relative to the specific\
    \ automated system, and the review system should be staffed and regularly assessed\
    \ to ensure it is providing timely consideration and fallback. In time-critical\
    \ systems, this mechanism should be immediately available or, where possible,\
    \ available before the harm occurs. Time-critical systems include, but are not\
    \ limited to, voting-related systems, automated building access and other access\
    \ systems, systems that form a critical component of healthcare, and systems that\
    \ have the ability to withhold wages or otherwise cause immediate financial penalties.\n\
    \n**Effective.** The organizational structure surrounding processes for consideration\
    \ and fallback should be designed so that if the human decision-maker charged\
    \ with reassessing a decision determines that it should be overruled, the new\
    \ decision will be effectively enacted. This includes ensuring that the new decision\
    \ is entered into the automated system throughout its components, any previous\
    \ repercussions from the old decision are also overturned, and safeguards are\
    \ put in place to help ensure that future decisions do not result in the same\
    \ errors.\n\n**Maintained. The human consideration and fallback process and any\
    \ associated automated processes** should be maintained and supported as long\
    \ as the relevant automated system continues to be in use.\n\n**Institute training,\
    \ assessment, and oversight to combat automation bias and ensure any** **human-based\
    \ components of a system are effective.**"
  - "**Institute training, assessment, and oversight to combat automation bias and\
    \ ensure any** **human-based components of a system are effective.**\n\n**Training\
    \ and assessment. Anyone administering, interacting with, or interpreting the\
    \ outputs of an auto­** mated system should receive training in that system, including\
    \ how to properly interpret outputs of a system in light of its intended purpose\
    \ and in how to mitigate the effects of automation bias. The training should reoc­\
    \ cur regularly to ensure it is up to date with the system and to ensure the system\
    \ is used appropriately. Assess­ ment should be ongoing to ensure that the use\
    \ of the system with human involvement provides for appropri­ ate results, i.e.,\
    \ that the involvement of people does not invalidate the system's assessment as\
    \ safe and effective or lead to algorithmic discrimination.\n\n**Oversight. Human-based\
    \ systems have the potential for bias, including automation bias, as well as other**\
    \ concerns that may limit their effectiveness. The results of assessments of the\
    \ efficacy and potential bias of such human-based systems should be overseen by\
    \ governance structures that have the potential to update the operation of the\
    \ human-based system in order to mitigate these effects. **HUMAN ALTERNATIVES,**\
    \ **CONSIDERATION, AND** **FALLBACK**\n\n###### WHAT SHOULD BE EXPECTED OF AUTOMATED\
    \ SYSTEMS\n\n The expectations for automated systems are meant to serve as a blueprint\
    \ for the development of additional technical standards and practices that are\
    \ tailored for particular sectors and contexts.\n\n**Implement additional human\
    \ oversight and safeguards for automated systems related to** **sensitive domains**\n\
    \nAutomated systems used within sensitive domains, including criminal justice,\
    \ employment, education, and health, should meet the expectations laid out throughout\
    \ this framework, especially avoiding capricious, inappropriate, and discriminatory\
    \ impacts of these technologies. Additionally, automated systems used within sensitive\
    \ domains should meet these expectations:"
- source_sentence: What is the primary goal of protecting the public from algorithmic
    discrimination?
  sentences:
  - 'Action ID: MS-1.3-001

    Suggested Action: Define relevant groups of interest (e.g., demographic groups,
    subject matter

    experts, experience with GAI technology) within the context of use as part of

    plans for gathering structured public feedback.

    GAI Risks: Human-AI Configuration; Harmful

    Bias and Homogenization; CBRN

    Information or Capabilities'
  - 'Action ID: GV-6.1-001

    Suggested Action: Categorize different types of GAI content with associated third-party
    rights (e.g.,

    copyright, intellectual property, data privacy).

    GAI Risks: Data Privacy; Intellectual

    Property; Value Chain and

    Component Integration'
  - '**Protect the public from algorithmic discrimination in a proactive and ongoing
    manner**


    **Proactive assessment of equity in design. Those responsible for the development,
    use, or oversight of** automated systems should conduct proactive equity assessments
    in the design phase of the technology research and development or during its acquisition
    to review potential input data, associated historical context, accessibility for
    people with disabilities, and societal goals to identify potential discrimination
    and effects on equity resulting from the introduction of the technology. The assessed
    groups should be as inclusive as possible of the underserved communities mentioned
    in the equity definition: Black, Latino, and Indigenous and Native American persons,
    Asian Americans and Pacific Islanders and other persons of color; members of religious
    minorities; women, girls, and non-binary people; lesbian, gay, bisexual, transgender,
    queer, and intersex (LGBTQI+) persons; older adults; persons with disabilities;
    persons who live in rural areas; and persons otherwise adversely affected by persistent
    poverty or inequality. Assessment could include both qualitative and quantitative
    evaluations of the system. This equity assessment should also be considered a
    core part of the goals of the consultation conducted as part of the safety and
    efficacy review.


    **Representative and robust data. Any data used as part of system development
    or assessment should be** representative of local communities based on the planned
    deployment setting and should be reviewed for bias based on the historical and
    societal context of the data. Such data should be sufficiently robust to identify
    and help to mitigate biases and potential harms.'
- source_sentence: How can human subjects revoke their consent according to the suggested
    action?
  sentences:
  - "Disinformation and misinformation – both of which may be facilitated by GAI –\
    \ may erode public trust in true or valid evidence and information, with downstream\
    \ effects. For example, a synthetic image of a Pentagon blast went viral and briefly\
    \ caused a drop in the stock market. Generative AI models can also assist malicious\
    \ actors in creating compelling imagery and propaganda to support disinformation\
    \ campaigns, which may not be photorealistic, but could enable these campaigns\
    \ to gain more reach and engagement on social media platforms. Additionally, generative\
    \ AI models can assist malicious actors in creating fraudulent content intended\
    \ to impersonate others.\n\n Trustworthy AI Characteristics: Accountable and Transparent,\
    \ Safe, Valid and Reliable, Interpretable and Explainable\n\n 2.9. Information\
    \ Security\n\n Information security for computer systems and data is a mature\
    \ field with widely accepted and standardized practices for offensive and defensive\
    \ cyber capabilities. GAI-based systems present two primary information security\
    \ risks: GAI could potentially discover or enable new cybersecurity risks by lowering\
    \ the barriers for or easing automated exercise of offensive capabilities; simultaneously,\
    \ it expands the available attack surface, as GAI itself is vulnerable to attacks\
    \ like prompt injection or data poisoning. \n\n Offensive cyber capabilities advanced\
    \ by GAI systems may augment cybersecurity attacks such as hacking, malware, and\
    \ phishing. Reports have indicated that LLMs are already able to discover some\
    \ vulnerabilities in systems (hardware, software, data) and write code to exploit\
    \ them. Sophisticated threat actors might further these risks by developing GAI-powered\
    \ security co-pilots for use in several parts of the attack chain, including informing\
    \ attackers on how to proactively evade threat detection and escalate privileges\
    \ after gaining system access.\n\n Information security for GAI models and systems\
    \ also includes maintaining availability of the GAI system and the integrity and\
    \ (when applicable) the confidentiality of the GAI code, training data, and model\
    \ weights. To identify and secure potential attack points in AI systems or specific\
    \ components of the AI"
  - 'Action ID: GV-4.2-003

    Suggested Action: Verify that downstream GAI system impacts (such as the use of
    third-party

    plugins) are included in the impact documentation process.

    GAI Risks: Value Chain and Component

    Integration'
  - 'Action ID: MS-2.2-003

    Suggested Action: Provide human subjects with options to withdraw participation
    or revoke their

    consent for present or future use of their data in GAI applications.

    GAI Risks: Data Privacy; Human-AI

    Configuration; Information

    Integrity'
model-index:
- name: SentenceTransformer based on Snowflake/snowflake-arctic-embed-xs
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: Unknown
      type: unknown
    metrics:
    - type: cosine_accuracy@1
      value: 0.6483516483516484
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.7362637362637363
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.7802197802197802
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.8543956043956044
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.6483516483516484
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.2454212454212454
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.15604395604395602
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.08543956043956043
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.6483516483516484
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.7362637362637363
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.7802197802197802
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.8543956043956044
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.7430085801122739
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.7085633612419325
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.7162684913786879
      name: Cosine Map@100
    - type: dot_accuracy@1
      value: 0.6483516483516484
      name: Dot Accuracy@1
    - type: dot_accuracy@3
      value: 0.7362637362637363
      name: Dot Accuracy@3
    - type: dot_accuracy@5
      value: 0.7802197802197802
      name: Dot Accuracy@5
    - type: dot_accuracy@10
      value: 0.8543956043956044
      name: Dot Accuracy@10
    - type: dot_precision@1
      value: 0.6483516483516484
      name: Dot Precision@1
    - type: dot_precision@3
      value: 0.2454212454212454
      name: Dot Precision@3
    - type: dot_precision@5
      value: 0.15604395604395602
      name: Dot Precision@5
    - type: dot_precision@10
      value: 0.08543956043956043
      name: Dot Precision@10
    - type: dot_recall@1
      value: 0.6483516483516484
      name: Dot Recall@1
    - type: dot_recall@3
      value: 0.7362637362637363
      name: Dot Recall@3
    - type: dot_recall@5
      value: 0.7802197802197802
      name: Dot Recall@5
    - type: dot_recall@10
      value: 0.8543956043956044
      name: Dot Recall@10
    - type: dot_ndcg@10
      value: 0.7430085801122739
      name: Dot Ndcg@10
    - type: dot_mrr@10
      value: 0.7085633612419325
      name: Dot Mrr@10
    - type: dot_map@100
      value: 0.7162684913786879
      name: Dot Map@100
---

# SentenceTransformer based on Snowflake/snowflake-arctic-embed-xs

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [Snowflake/snowflake-arctic-embed-xs](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [Snowflake/snowflake-arctic-embed-xs](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs) <!-- at revision 742da4f66e1823b5b4dbe6c320a1375a1fd85f9e -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 384 tokens
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'How can human subjects revoke their consent according to the suggested action?',
    'Action ID: MS-2.2-003\nSuggested Action: Provide human subjects with options to withdraw participation or revoke their\nconsent for present or future use of their data in GAI applications.\nGAI Risks: Data Privacy; Human-AI\nConfiguration; Information\nIntegrity',
    'Disinformation and misinformation – both of which may be facilitated by GAI – may erode public trust in true or valid evidence and information, with downstream effects. For example, a synthetic image of a Pentagon blast went viral and briefly caused a drop in the stock market. Generative AI models can also assist malicious actors in creating compelling imagery and propaganda to support disinformation campaigns, which may not be photorealistic, but could enable these campaigns to gain more reach and engagement on social media platforms. Additionally, generative AI models can assist malicious actors in creating fraudulent content intended to impersonate others.\n\n Trustworthy AI Characteristics: Accountable and Transparent, Safe, Valid and Reliable, Interpretable and Explainable\n\n 2.9. Information Security\n\n Information security for computer systems and data is a mature field with widely accepted and standardized practices for offensive and defensive cyber capabilities. GAI-based systems present two primary information security risks: GAI could potentially discover or enable new cybersecurity risks by lowering the barriers for or easing automated exercise of offensive capabilities; simultaneously, it expands the available attack surface, as GAI itself is vulnerable to attacks like prompt injection or data poisoning. \n\n Offensive cyber capabilities advanced by GAI systems may augment cybersecurity attacks such as hacking, malware, and phishing. Reports have indicated that LLMs are already able to discover some vulnerabilities in systems (hardware, software, data) and write code to exploit them. Sophisticated threat actors might further these risks by developing GAI-powered security co-pilots for use in several parts of the attack chain, including informing attackers on how to proactively evade threat detection and escalate privileges after gaining system access.\n\n Information security for GAI models and systems also includes maintaining availability of the GAI system and the integrity and (when applicable) the confidentiality of the GAI code, training data, and model weights. To identify and secure potential attack points in AI systems or specific components of the AI',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval

* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.6484     |
| cosine_accuracy@3   | 0.7363     |
| cosine_accuracy@5   | 0.7802     |
| cosine_accuracy@10  | 0.8544     |
| cosine_precision@1  | 0.6484     |
| cosine_precision@3  | 0.2454     |
| cosine_precision@5  | 0.156      |
| cosine_precision@10 | 0.0854     |
| cosine_recall@1     | 0.6484     |
| cosine_recall@3     | 0.7363     |
| cosine_recall@5     | 0.7802     |
| cosine_recall@10    | 0.8544     |
| cosine_ndcg@10      | 0.743      |
| cosine_mrr@10       | 0.7086     |
| **cosine_map@100**  | **0.7163** |
| dot_accuracy@1      | 0.6484     |
| dot_accuracy@3      | 0.7363     |
| dot_accuracy@5      | 0.7802     |
| dot_accuracy@10     | 0.8544     |
| dot_precision@1     | 0.6484     |
| dot_precision@3     | 0.2454     |
| dot_precision@5     | 0.156      |
| dot_precision@10    | 0.0854     |
| dot_recall@1        | 0.6484     |
| dot_recall@3        | 0.7363     |
| dot_recall@5        | 0.7802     |
| dot_recall@10       | 0.8544     |
| dot_ndcg@10         | 0.743      |
| dot_mrr@10          | 0.7086     |
| dot_map@100         | 0.7163     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 2,730 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                           |
  |:--------|:----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                               |
  | details | <ul><li>min: 8 tokens</li><li>mean: 15.71 tokens</li><li>max: 33 tokens</li></ul> | <ul><li>min: 19 tokens</li><li>mean: 183.25 tokens</li><li>max: 467 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                      | sentence_1                                                                                                                                                                                                               |
  |:------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>What is the Action ID associated with the suggested action?</code>                        | <code>Action ID: MS-2.12-004<br>Suggested Action: Verify effectiveness of carbon capture or offset programs for GAI training and<br>applications, and address green-washing concerns.<br>GAI Risks: Environmental</code> |
  | <code>What is the suggested action regarding carbon capture or offset programs?</code>          | <code>Action ID: MS-2.12-004<br>Suggested Action: Verify effectiveness of carbon capture or offset programs for GAI training and<br>applications, and address green-washing concerns.<br>GAI Risks: Environmental</code> |
  | <code>What specific concerns should be addressed in relation to carbon capture programs?</code> | <code>Action ID: MS-2.12-004<br>Suggested Action: Verify effectiveness of carbon capture or offset programs for GAI training and<br>applications, and address green-washing concerns.<br>GAI Risks: Environmental</code> |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          284,
          256,
          128,
          64,
          32
      ],
      "matryoshka_weights": [
          1,
          1,
          1,
          1,
          1
      ],
      "n_dims_per_step": -1
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 10
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 10
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `eval_use_gather_object`: False
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss | cosine_map@100 |
|:------:|:----:|:-------------:|:--------------:|
| 0.2924 | 50   | -             | 0.5949         |
| 0.5848 | 100  | -             | 0.6455         |
| 0.8772 | 150  | -             | 0.6680         |
| 1.0    | 171  | -             | 0.6721         |
| 1.1696 | 200  | -             | 0.6811         |
| 1.4620 | 250  | -             | 0.6850         |
| 1.7544 | 300  | -             | 0.6959         |
| 2.0    | 342  | -             | 0.7021         |
| 2.0468 | 350  | -             | 0.7008         |
| 2.3392 | 400  | -             | 0.7043         |
| 2.6316 | 450  | -             | 0.7017         |
| 2.9240 | 500  | 5.9671        | 0.7018         |
| 3.0    | 513  | -             | 0.7039         |
| 3.2164 | 550  | -             | 0.7014         |
| 3.5088 | 600  | -             | 0.7039         |
| 3.8012 | 650  | -             | 0.7022         |
| 4.0    | 684  | -             | 0.7058         |
| 4.0936 | 700  | -             | 0.7039         |
| 4.3860 | 750  | -             | 0.7061         |
| 4.6784 | 800  | -             | 0.7030         |
| 4.9708 | 850  | -             | 0.7073         |
| 5.0    | 855  | -             | 0.7073         |
| 5.2632 | 900  | -             | 0.7071         |
| 5.5556 | 950  | -             | 0.7095         |
| 5.8480 | 1000 | 3.5897        | 0.7103         |
| 6.0    | 1026 | -             | 0.7080         |
| 6.1404 | 1050 | -             | 0.7075         |
| 6.4327 | 1100 | -             | 0.7089         |
| 6.7251 | 1150 | -             | 0.7087         |
| 7.0    | 1197 | -             | 0.7102         |
| 7.0175 | 1200 | -             | 0.7101         |
| 7.3099 | 1250 | -             | 0.7134         |
| 7.6023 | 1300 | -             | 0.7130         |
| 7.8947 | 1350 | -             | 0.7133         |
| 8.0    | 1368 | -             | 0.7142         |
| 8.1871 | 1400 | -             | 0.7125         |
| 8.4795 | 1450 | -             | 0.7163         |


### Framework Versions
- Python: 3.11.9
- Sentence Transformers: 3.2.0
- Transformers: 4.44.1
- PyTorch: 2.4.0
- Accelerate: 0.34.2
- Datasets: 3.0.0
- Tokenizers: 0.19.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MatryoshkaLoss
```bibtex
@misc{kusupati2024matryoshka,
    title={Matryoshka Representation Learning},
    author={Aditya Kusupati and Gantavya Bhatt and Aniket Rege and Matthew Wallingford and Aditya Sinha and Vivek Ramanujan and William Howard-Snyder and Kaifeng Chen and Sham Kakade and Prateek Jain and Ali Farhadi},
    year={2024},
    eprint={2205.13147},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->