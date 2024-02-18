# Multilabel_StackOverflow_Tag_Classification

## Intro
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

In this step, we perform Exploratory Data Analysis, in order to deeply understand our data. 

You will most likely need to use the Questions.csv and Tags.csv files in order to complete the tasks described in the following sec- tions. 


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weiji14/deepbedmap/]

## Baseline Model

## BERT Model

> [!NOTE]
> Useful information that users should know, even when skimming content.

> [!TIP]
> Helpful advice for doing things better or more easily.

> [!IMPORTANT]
> Key information users need to know to achieve their goal.

> [!WARNING]
> Urgent info that needs immediate user attention to avoid problems.

> [!CAUTION]
> Advises about risks or negative outcomes of certain actions.

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
    <td colspan="4">N/A</td>
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
  <td colspan="4">N/A</td>
</tr>
  <tr>
    <td>76.766</td>
    <td>200</td>
    <td>133</td>
    <td colspan="8">MEMORY RAM CRASH</td>
  </tr>
</table>


Unfortunately on the experiment procedure, we face RAM issues with 200 Top Tag Combinations, therefore we must limit in smaller subsets of original dataset.
