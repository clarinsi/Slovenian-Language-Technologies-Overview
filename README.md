# Slovenian Language Technologies Overview
An ever-expanding collaborative overview of the knowledge on large language models (LLMs), speech technologies, and other NLP technologies for Slovenian language.

Content:
- [Instruction-Tuned LLMs for Slovenian](#instruction-tuned-llms-for-slovenian)
- [Embedding Models & RAG for Slovenian](#embedding-models--rag-for-slovenian)
- [Automatic Speech Recognition (ASR) for Slovenian](#automatic-speech-recognition-asr-for-slovenian)
- [Machine Translation for Slovenian](#machine-translation-for-slovenian)
- [Fine-Tuned Models for Slovenian](#fine-tuned-models-for-slovenian)
- [BERT-like pretrained models for Slovenian](#bert-like-pretrained-models-for-slovenian)

## Instruction-Tuned LLMs for Slovenian

**Open-Source Models**:
- specialised for Slovenian: recently-available GaMS models by [CJVT](https://huggingface.co/cjvt): [OPT_GaMS-1B-Chat](https://huggingface.co/cjvt/OPT_GaMS-1B-Chat) in [GaMS-1B-Chat](https://huggingface.co/cjvt/GaMS-1B-Chat), 1B models, developed as part of the [POVEJMO](https://povejmo.si/) project - bigger models will follow as the final products of this project
- multilingual models that performed well on Slovenian and South Slavic languages (and dialects) based on the COPA task (see [paper by Ljubešić et al., 2024](https://aclanthology.org/2024.vardial-1.18.pdf)): [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1), [mt0-xxl](https://huggingface.co/bigscience/mt0-xxl) and [Aya](https://huggingface.co/CohereForAI/aya-101)

Based on experience (e.g., [paper by Ljubešić et al., 2024](https://aclanthology.org/2024.vardial-1.18.pdf), using its predecesor GPT-4), [closed-source GPT-4o by OpenAI](https://openai.com/index/hello-gpt-4o/) still performs the best for Slovenian.

**Benchmarks**:

**Papers**:
- [JSI and WüNLP at the DIALECT-COPA Shared Task: In-Context Learning From Just a Few Dialectal Examples Gets You Quite Far](https://aclanthology.org/2024.vardial-1.18.pdf) (Ljubešić et al., 2024)


## Embedding Models & RAG for Slovenian

**Open-Source Embedding Models**:
- based on paper evaluating retrieval capabilities ([Kuzman et al., 2024](https://is.ijs.si/wp-content/uploads/2024/10/IS2024_Volume-A-DRAFT-1.pdf)), the best smaller-sized open-source embedding models for Slovenian are [BGE-M3](https://huggingface.co/BAAI/bge-m3) and [Multilingual-E5-large](https://huggingface.co/intfloat/multilingual-e5-large)

**Benchmarks**:
- RAG benchmark for retrieval capabilities of the RAG pipeline: [PandaChat-RAG Benchmark](https://github.com/TajaKuzman/pandachat-rag-benchmark)

**Papers**:
- [PandaChat-RAG: Towards the Benchmark for Slovenian RAG Applications](https://is.ijs.si/wp-content/uploads/2024/10/IS2024_Volume-A-DRAFT-1.pdf) (page 15) (Kuzman et al., 2024)

## Automatic Speech Recognition (ASR) for Slovenian

**Open-Source Models**:
- [Slovene Conformer CTC BPE E2E Automated Speech Recognition model RSDO-DS2-ASR-E2E 2.0](
http://hdl.handle.net/11356/1737): ASR model, developed inside the [RSDO project](https://rsdo.slovenscina.eu/en/speech-technologies), that is available on the CLARIN.SI repository and [GitHub](https://github.com/clarinsi/Slovene_ASR_e2e) ([demo](https://www.slovenscina.eu/en/razpoznavalnik)). Note: The maximal accepted audio duration is 300s.
- [Whisper](https://huggingface.co/openai/whisper-large-v3) model: open-source OpenAI model that is massively multilingual.

**Benchmarks**:
- [SloBench Speech Recognition benchmark](https://slobench.cjvt.si/leaderboard/view/10)
 
**Papers**:

## Machine Translation for Slovenian

**Open-Source Models**:
- [No Language Left Behind (NLLB)](https://github.com/facebookresearch/fairseq/tree/nllb) massively multilingual models are frequently used for large-scale machine translation.
- Inside the [ParlaMint](https://www.clarin.eu/parlamint-project-information) project dealing with parliamentary texts, the OPUS-MT models used through [EasyNMT](https://github.com/UKPLab/EasyNMT) library were shown to be the most useful for our purposes. For Slovenian to English, we used the [opus-mt-sla-en](https://huggingface.co/Helsinki-NLP/opus-mt-sla-en) model.
- [Neural Machine Translation model for Slovene-English language pair RSDO-DS4-NMT 1.2.6](https://www.clarin.si/repository/xmlui/handle/11356/1736): MT model, developed inside the [RSDO project](https://rsdo.slovenscina.eu/en/machine-translation). There is a [demo](https://www.slovenscina.eu/en/prevajalnik) available. Code for the API service is [available here](https://github.com/clarinsi/Slovene_NMT).

**Benchmarks**:
- SloBench Machine Translation benchmarks: [Slovenian-to-English](https://slobench.cjvt.si/leaderboard/view/7) and [English-to-Slovenian](https://slobench.cjvt.si/leaderboard/view/8)

**Papers**:

## Fine-Tuned Models for Slovenian

**Models & Papers**:
- Sentiment in parliamentary texts: [Multilingual parliament sentiment regression model XLM-R-ParlaSent](https://huggingface.co/classla/xlm-r-parlasent) ([Mochtak et al., 2024](https://aclanthology.org/2024.lrec-main.1393/))
- Text genre prediction: [X-GENRE classifier - multilingual text genre classifier](https://huggingface.co/classla/xlm-roberta-base-multilingual-text-genre-classifier) ([Kuzman et al., 2023](https://www.mdpi.com/2504-4990/5/3/59))
- News topic prediction: [Text classification model SloBERTa-Trendi-Topics 1.0](https://huggingface.co/cjvt/sloberta-trendi-topics) ([Kosem et al., 2023](https://journals.uni-lj.si/slovenscina2/article/download/12073/13790))

**Benchmarks**:
- [Natural language inference](https://slobench.cjvt.si/leaderboard/view/9) benchmark at SloBench
- [Slovene SuperGLUE](https://slobench.cjvt.si/leaderboard/view/3) benchmark at SloBench
- [Named Entity Recognition](https://slobench.cjvt.si/leaderboard/view/12) benchmark at SloBench
- [Universal Dependency Parsing](https://slobench.cjvt.si/leaderboard/view/11) benchmark at SloBench

**Papers**:

##  BERT-like pretrained models for Slovenian

**Monolingual / Smaller multilingual Models**:
- [SloBERTa](https://huggingface.co/EMBEDDIA/sloberta): monolingual Slovenian BERT-like model
- [CroSloEngual BERT](https://huggingface.co/EMBEDDIA/crosloengual-bert): a trilingual model trained on Croatian, Slovenian, and English corpora

**Massively Multilingual Models**:
- [Massively multilingual XLM-RoBERTa](https://huggingface.co/FacebookAI/xlm-roberta-large) model: frequently used for fine-tuning on Slovenian and multilingual data for various NLP tasks
- [Multilingual parliamentary model XLM-R-parla](https://huggingface.co/classla/xlm-r-parla): XLM-RoBERTa model, additionally pretrained on parliamentary data, to be used for NLP tasks applied on parliamentary texts

**Papers**:
