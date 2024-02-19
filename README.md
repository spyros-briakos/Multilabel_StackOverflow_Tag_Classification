# Multilabel Tag Prediction on StackOverflow

![Alt text](https://example.com/image.png)

https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/Stack_Overflow_logo.svg/1280px-Stack_Overflow_logo.svg.png

## Introduction
The current case-study revolves around tag prediction of Stack Overflow questions. We utilise [StackSample Kaggle dataset](https://www.kaggle.com/datasets/stackoverflow/stacksample), which represents approximately 10% of Stack Overflow Q&A corpus. More specifically we only utilise the following files:
1. Questions.csv
    * A unique identifier of the user that created each question
    * A unique identifier of the question itself
    * Creation and closing datetimes corresponding to each question
    * The cumulative reaction score of each question (zero, positive or negative)
    * The title and main body of each question.
3. Tags.csv
    * A unique identifier for each question.
    * One or more associated tags.



## EDA 
Due to time and resources limitations, from the beginning of this project, we knew that we must retain a proper subset of the original dataset. After merging two csv files, we aimed to identify insights for Tags via plots and statistics, thus leading us to the result to experiment in keeping only top N tags. After some trial and errors, we decided it (for future ease) to experiment in keeping with the top M tag combinations. Note that we decided to opt only the Questions with positive cummulative score, as we believe this kind of questions, most of the times provide valuable insights and solutions, therefore more quality data.


The biggest time allocation of this project was for sure EDA, and one of its subgoals, which is preprocessing. We gave a strong focus with many manual examples exploration, in order to reassure pipeline's stability. We treated specially words like the name of a Tag, because they give a strong weight to prediction. In addition, most of the tags are programming languages, packages, versions, systems and other and due to this fact we were very cautius with punctuation (C#,C++,.NET).

After a single run of EDA notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spyros-briakos/Multilabel_StackOverflow_Tag_Prediction/blob/main/notebooks/EDA.ipynb), having opted M value, it is produced a preprocessed.csv, which contains data with top M tag combinations, ready to manipulate afterwards on model notebooks. In that way, code is more generic and we have the ability to store different subsets of original dataset, as we prefer.











## Baseline Model

With similar thought process Baseline_Model notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spyros-briakos/Multilabel_StackOverflow_Tag_Prediction/blob/main/notebooks/Baseline_Model.ipynb) is 

> [!TIP]
> Helpful advice for doing things better or more easily.


## LLM Model
dsadsadsada





















## Experiment Table

<table>

  <tr>
    <td colspan="3">EDA</td>
    <td colspan="4">Baseline Model</td>
    <td colspan="4">LLM Model</td>
  </tr>
  
  <tr>
    <td>Questions</td>
    <td>Tag Combinations</td>
    <td>Unique Tags</td>
    <td>Hamming Loss</td>
    <td>Micro-F1</td>
    <td>Macro-F1</td>
    <td>Time</td>
    <td>Hamming Loss</td>
    <td>Micro-F1</td>
    <td>Macro-F1</td>
    <td>Epoch Time with GPU</td>
  </tr>
  <tr>
    <td>33.374</td>
    <td>20</td>
    <td>16</td>
    <td>0.03</td>
    <td>0.83</td>
    <td>0.81</td>
    <td>3 mins</td>
    <td>0.02</td>
    <td>0.86</td>
    <td>0.84</td>
    <td>11 mins</td>
  </tr>
  <tr>
    <td>42.369</td>
    <td>35</td>
    <td>28</td>
    <td>0.02</td>
    <td>0.8</td>
    <td>0.79</td>
    <td>6 mins</td>
    <td colspan="4" align="center">N/A</td>
  </tr>
  <tr>
    <td>48.505</td>
  <td>50</td>
  <td>38</td>
  <td>0.01</td>
  <td>0.79</td>
  <td>0.77</td>
  <td>8 mins</td>
  <td>0.02</td>
  <td>0.68</td>
  <td>0.28</td>
  <td>16 mins</td>
  </tr>
<tr>
  <td>62.118</td>
  <td>100</td>
  <td>74</td>
  <td>0.01</td>
  <td>0.78</td>
  <td>0.7</td>
  <td>17 mins</td>
  <td>0.02</td>
  <td>0.68</td>
  <td>0.28</td>
  <td>21 mins</td>
</tr>

<tr>
  <td>70.474</td>
  <td>150</td>
  <td>104</td>
  <td>0.01</td>
  <td>0.76</td>
  <td>0.68</td>
  <td>24 mins</td>
  <td colspan="4" align="center">N/A</td>
</tr>
  <tr>
    <td>76.766</td>
    <td>200</td>
    <td>133</td>
    <td colspan="8" align="center">MEMORY RAM CRASH</td>
  </tr>
</table>


Unfortunately on the experiment procedure, we face RAM issues with 200 Top Tag Combinations, therefore we must limit in smaller subsets of original dataset.

Baseline model is **Linear Support Vector Machines** and except of its pretty descent performance, we have to mention that was very time efficient, managing to train in just a few seconds.

For a more sophisticated model we opted from HuggingFace library BERTforSequenceClassification architecture with **BERT 'bert-base-uncased'** model. Our choice is justified by the fact that we have a multilabel classification problem and also BERT is very popular for text classification. 

