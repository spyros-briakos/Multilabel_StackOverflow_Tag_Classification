# Multilabel Tag Prediction on StackOverflow Questions

![Alt text](images/stackoverflow.png)

## Introduction
The current case-study revolves around tag prediction of Stack Overflow questions. We utilise [StackSample Kaggle dataset](https://www.kaggle.com/datasets/stackoverflow/stacksample), which represents approximately 10% of Stack Overflow Q&A corpus. More specifically we only utilise the following files:
1. `Questions.csv`
    * A unique identifier of the user that created each question
    * A unique identifier of the question itself
    * Creation and closing datetimes corresponding to each question
    * The cumulative reaction score of each question (zero, positive or negative)
    * The title and main body of each question.
3. `Tags.csv`
    * A unique identifier for each question.
    * One or more associated tags.

I am utilising a diverse range of NLP tools and models, spanning from basic ML traditional models to advanced neural networks, like BERT that has been fine-tuned for the specific task. Finally we merge all the results together so as to compare from metrics and efficiency perspective.


All three following notebooks can be configured with desired number M, which represents number of Top Tag Combinations, and retrieve the respective dataset's subset for experimental purposes. 

## EDA 
Due to time and resources limitations, from the beginning of this project, we knew that we must retain a proper subset of the original dataset. After merging two csv files, we aimed to identify insights for Tags via plots and statistics, thus leading us to the result to experiment in keeping only top N tags. After some trial and errors, we decided it (for future ease) to experiment in keeping with the top M tag combinations. Note that we decided to opt only the Questions with positive cummulative score, as we believe this kind of questions, most of the times provide valuable insights and solutions, therefore more quality data.

The biggest time allocation of this project was for sure EDA, and one of its subgoals, which is preprocessing. We gave a strong focus with many manual examples exploration, in order to reassure pipeline's stability. We treated specially words like the name of a Tag, because they give a strong weight to prediction. In addition, most of the tags are programming languages, packages, versions, systems and other and due to this fact we were very cautius with punctuation (C#,C++,.NET).

> ***    
Preprocessing Pipeline:
- Lowercase
- Cleaning
   - Removal of noise from tag words
   - Removal of HTML tags
   - Removal of IPV4 adresses 
   - Removal of URLs
   - Removal of redundant symbols (spaces, newline etc.)
   - Fix Contractions
- Tokenization (NLTK) on words!=tags
- Punctuation Removal (except of '.-#+')
- Removal of Stopwords
- Removal of only-puncts words
- Removal of only-digit words
- Joining 
> ***    



After a single run of `EDA.ipynb` notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spyros-briakos/Multilabel_StackOverflow_Tag_Prediction/blob/main/notebooks/EDA.ipynb), having opted M value, it is produced a preprocessed.csv, which contains data with top M tag combinations, ready to manipulate afterwards on model notebooks. In that way, code is more generic and we have the ability to store different subsets of original dataset, as we prefer.











## Baseline Model

With similar thought process, `Baseline_Model.ipnyb` notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spyros-briakos/Multilabel_StackOverflow_Tag_Prediction/blob/main/notebooks/Baseline_Model.ipynb) defines the M number in order to retrieve, whichever preprocessed version of dataset the user prefers. 

Our problem is multilabel, thus we already know that Tag may be more than one and this is a tricky point that we need to focus about splitting the original dataset properly. We implemented a function which will be utilised in both model notebooks, whose goal is to split the data properly into train,test or train,val,test with configurable sizes. Through this manual function we are reassured that our final sets are balanced in of all the possible Tag Combinations. 

> [!TIP]
> To split data properly and result to well distributed train,val,test sets try convert multilabel to single label by calculating all label combinations. With that logic we are pretty sure that sets will be balanced and contain a based-on-ratio-equally number of examples, leading to sufficient model's exposure to all cases. 


## BERT Model

`LLM_Model.ipynb` notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spyros-briakos/Multilabel_StackOverflow_Tag_Prediction/blob/main/notebooks/LLM_Model.ipynb)

A series of experiments took place in Google Colab, where all notebooks run, utilising for BERT GPU, due to its heavy architecture. As Devlin proposed for finetuning tasks, we experiment with batch size: {16,32} and a small number of epochs: 3 (time restrictions).






















## Experiment Table

<table>

  <tr>
    <td colspan="3" align="center">EDA</td>
    <td colspan="3" align="center">Linear SVC</td>
    <td colspan="4" align="center">BERT</td>
  </tr>
  
  <tr>
    <td align="center">Questions</td>
    <td align="center">Tag Combinations</td>
    <td align="center">Unique Tags</td>
    <td align="center">Hamming Loss</td>
    <td align="center">Micro-F1</td>
    <td align="center">Macro-F1</td>
    <td align="center">Hamming Loss</td>
    <td align="center">Micro-F1</td>
    <td align="center">Macro-F1</td>
    <td align="center">Epoch GPU (Batch Size)</td>
  </tr>
  <tr>
    <td align="center">33.374</td>
    <td align="center">20</td>
    <td align="center">16</td>
    <td align="center">0.03</td>
    <td align="center">0.83</td>
    <td align="center">0.81</td>
    <td align="center">0.02</td>
    <td align="center">0.86</td>
    <td align="center">0.85</td>
    <td align="center">12 (32)</td>
  </tr>
  <tr>
    <td align="center">42.369</td>
    <td align="center">35</td>
    <td align="center">28</td>
    <td align="center">0.02</td>
    <td align="center">0.8</td>
    <td align="center">0.79</td>
    <td colspan="4" align="center">N/A</td>
  </tr>
  <tr>
    <td align="center">48.505</td>
  <td align="center">50</td>
  <td align="center">38</td>
  <td align="center">0.01</td>
  <td align="center">0.79</td>
  <td align="center">0.77</td>
  <td align="center">0.01</td>
  <td align="center">0.84</td>
  <td align="center">0.81</td>
  <td align="center">18 (16)</td>
  </tr>
<tr>
  <td align="center">62.118</td>
  <td align="center">100</td>
  <td align="center">74</td>
  <td align="center">0.01</td>
  <td align="center">0.78</td>
  <td align="center">0.7</td>
  <td align="center">0.01</td>
  <td align="center">0.82</td>
  <td align="center">0.7</td>
  <td align="center">21 (16)</td>
</tr>

<tr>
  <td align="center">70.474</td>
  <td align="center">150</td>
  <td align="center">104</td>
  <td align="center">0.01</td>
  <td align="center">0.76</td>
  <td align="center">0.68</td>
  <td colspan="4" align="center">N/A</td>
</tr>
  <tr>
    <td align="center">76.766</td>
    <td align="center">200</td>
    <td align="center">133</td>
    <td colspan="8" align="center">MEMORY RAM CRASH</td>
  </tr>
</table>


Unfortunately on the experiment procedure, we face RAM issues with 200 Top Tag Combinations, therefore we must limit in smaller subsets of original dataset.

Baseline model is **Linear Support Vector Machines** and except of its pretty descent performance, we have to mention that was very time efficient, managing to train in just a few seconds.

For a more sophisticated model we opted from HuggingFace library BERTforSequenceClassification architecture with **BERT 'bert-base-uncased'** model. Our choice is justified by the fact that we have a multilabel classification problem and also BERT is very popular for text classification. 

