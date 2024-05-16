# Instruction Fine Tunning
## In-context learning (ICL) - one/few shot inference
Limitation
    In context learning may not work for smaller model.
    Example takeup space in the context window. So, reduce the room for other useful information.

## Fine Tunning at a High Level
Train the base model.
It is supervised learning model.

Pretrained Model > Task Specific Example - Prompt and Completion > Fine Tunned LLM.

## Uning Prompts to fine-tune LLMs with Instructions
Feed examples to pretrained model.
    Classify this review: I love this DVD! Sentiment: Positive.
    ...

If you want to fine tune about summarizing task then Example should have texts and summary in example.
Similar for translation.

It is full fine tunning. Update all parameters. It need enough compute and budget like full training.

First Step is - Prepare Data.
    Developer have assembled prompt template libraries that to take existing data set and convert to instruction dataset.
Divide into training, validation and test

You pass training prompt to model, it generate a completion. Now you compare result with actual training set. You can use standard cross entropy function to calculate loss.
Back propagation to update weight.

# Fine Tunning on a Single Task
Limited dataset serve our purpose in fine tuning.

## Catastrophic Forgetting
Fine tuning can significantly increase the performance of a model on a specific task, it can degrade performance of other tasks.

## How to avoid Catastrophic Forgetting?
Check if it really impact your use case.
If impact, then fine-tune on multiple tasks at the same time.
Or Consider Parameter Efficient Fine Tunning (PEFT)
    It retain original weight. Only small layer related to tasks.

# Multitask Instruction Fine Tunning
Training Set contain data for all tasks.
Drawback - Lots of data.

## Instruction Fine Tunning with FLAN
FLAN models refer to a specific set of instructions used to perform instruction fine tunning.
FLAN-T5; FLAN-PALM

FLAN-T5: Fine Tunned version of pre-trained T5 Models
    It is a great, general purpose, instruct model

# Scaling Instruct Model
https://arxiv.org/abs/2210.11416

# Model Evaluation
## Model Evaluation Metrics
Accuracy = Correct Prediction/Total Prediction
But this is not easy to calculate in LLM.
    "Mike really love drinking team" = "Mike adores sipping tea".
    But how you calculate similarity.

    "Mike does not drink coffee" != 'Mike does drink coffee". 

    Human can see difference and similarity.

Two metrics - ROUGE and BLEU SCORE
Rouge
    Used for text summarization
    Compare a summary to one or more reference summaries.
BLEU
    Used for text translation.

## Terminology
The dog lay on the rug as I sipped a cup of tea.

Unigram - A single word - cup
Bigram - dig lay
n-gram

ROUGE 1 Recall - Unigram matches/Unigram in reference.
ROUGE 1 Precision - Unigram matches/Unigram in Output.

ROUGE 1 F1 - 2 * (precision * recall)/(precision + recall)

ROUGE 1 - fail for "Not" word.

Rouge 2 similar to 1.. but it consider all bi-gram

## Rouge L
Longest Common Subsequence
recall = LCS(Gen, Ref)/unigram in reference
precision = LCS/unigram in output.

Bad sentence have good score.
Reference = "It is cold outside". Output = 'cold cold cold cold' or "outside cold it is"

## BLEU
Average(precision across range of n-gram sizes)

# Benchmark
Actual check, we have some existing dataset and values.
Choose write dataset

## Evaluation Benchmark
GLUE, SuperGLUE, HELM, BIG-bench; MMLU