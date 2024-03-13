# Dissertation repository
This repo contains the code and dataset used and created for my MSc Data Science dissertation. The following is the abstract:

  Research papers’ publications keep increasing over time. Nevertheless, reading papers is a complicated task: it is hard to keep up with the number of publications, and information is scattered across sections in an unstructured PDF file. If a database with structured data from scientific papers existed, we could create several impactful applications across different STEM areas. Thus, this work proposes using Large Language Models (LLMs) to extract numeric entities from scientific text across 26 research fields. For this, we 1) discuss why LLMs among different deep-learning techniques are ideal for this task, 2) build our own dataset with a self-proposed labelling approach, and 3) fine-tune nine models using Parameter Efficient Tuning. Results show LLMs have great retrieval capabilities with recall and F1 scores of 90.17 and 87.56, respectively, outperforming, for example, ChatGPT’s 10-shot learning. A web application is built with the results of the model, and 18 academics evaluated it in an online survey, demonstrating the usefulness of the product as well as the existence of a niche of users whose search experience could be highly improved.

We find in this repo:
- The dataset used for the model fine-tuning. This was self-created and annotated.
- Some evaluation functions and scripts that are used to perform the metrics of the structured information extraction.
- The samples used for the in-context learning evaluation.
- The dissertation document in PDF.
- The dissertation responses to the survey in Excel.
- The predictions of each of the tested model's outputs.
- The code for the Streamlit web application that has the main search engine. 
