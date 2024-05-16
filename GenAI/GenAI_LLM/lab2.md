# Fine Tune - Full and Paramete one.
FLAN-T5 Model
Python3
 ml.m5.2xlarge

PyTorch
rouge_score=0.1.2

import

Load Dataset and LLM
    Same as lab1
    model
    tokenizer

Print Model Parameters

Test the model with Zero Shot Reference

Perform Full Fine Tuning
    Tokenize some examples
    Train, Test and Validate
    Trainer to train

Download the fine tune from s3.
Test the same.
ROUGE Metrics - Compare with Original llm and trained.

# PEFT
LoraConfig - Here we tell rank

Apply PEFT Config on MOdel.

Trainer to train the model. (adaptor)
Download from S3.(adaptor)

merge the adaptor with LLM.

FINE > PEFT > LLM
