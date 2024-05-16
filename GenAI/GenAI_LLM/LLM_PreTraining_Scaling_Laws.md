# Pre-training LLM
Lifecycle has a step - Select: Choose an existing model or pretrain your own.
## Consideration for choosing model
Founation Model - Pretrained LLMs
    Mostly Used
Train Your Own LLM - Custom LLMs

## Model Hubs
Hugging Faces; PyTorch
This hub contains model cards - Detail; Use Case; Bias/Risk/Limitation; Training Details; Evalution

we choose based on the tasks.

## Model Architectures and Pre-training Objective
Large unstructured text data > LLM Model to Train 
    Model identify pattern and structure in languages.
    Minimize loss function.

Encoder generate embedding for each token represnetation.

If data is from public sites, we need to preprocess the data to feed only quality data to model.

Transformer 3 variance
    Encoder only; Encoder, Decoder models; Decoder Only models.

### Encoder Only - Autoencoding Models
Trained using Masked Language Model. It traverse the token in both direction.

Use for Token Classification.

Predict middle word.

Good Use Case-
    Sentiment Analysis
    Named Entity Recognization
    Word Classification

BERT and ROBERTA

### Autoregressor Model - Decoder Only.
Predict next token based on previous tokens.

Only see token before one.
Context is unidirectional.

Use for generating tokens.

Good Use Cases
    Text Generation
    Other emergent behavior
        Depend on model size.

GPT and BLOOMER

### Sequence to Sequence Model - Use Both.
Good Use Cases
    Translation
    Text Summarization
    Question/Answer

T5, BART

## Significance of Scale: Task Ability
Larger the model, it can carry your task without or with little training.

# Computational Challenge of Training LLMs
CUDA Out of Memory Error - CUDA is used by ML Framework to efficiently do operation like multiplication.
## Approx GPU RAM needed to store 1 B Parameters
1 parameters = 4 Bytes (32 bit float)
1B parameters = 4 * 10^9 = 4 GB
    4GB @ 32-bit full precision

Other than parameters -
    Adam Optimizers (2 states) = +8 bytes per parameter
    Gradients = +4 bytes per parameter
    Activation and Temp Memory (Variable Size) = + 8 bytes per parameter (high end estimation)

Actual = 24GB @32 bit full precision

## Quantization - 1 way to reduce memory
Instead of 32 bit precision, we can project the same on 16 or 8 bit precision.

FP32 - 1 bit for sign + 8 bit for exponent + 23 bits for Fraction.
FP16 - 1 bit for sign + 5 bit              + 10 bits

You loose some precision.
BF16 or BFLOAT16 - It is used now alots. 1+8+7

Quantization-aware training (QAT) learn the quantization scaling factors during training.

As model size get larger, you will need to split your model across multiple GPUs for Training.

# Efficient Multi-GPU Compute Strategies
## When to use Distribute Compute
When model too big to fit for single GPU.
Model fit on GPU, train data in parallel.

## Distribute Data Parallel (DDP)
Requirement - Our model can fit on one GPU.

Dataloader distribute data on different GPUs. 
Each dataset in processed in parallel. Forward/BackwardProcess
Synchronize Gradient - Combine the result of each GPU
Update all models.

Data Loader > GPUs > (Forward/Backward)'s (one for one GPU) > Synchonize > Update Model's 

## Fully Sharded Data Paraller (FSDP)
Motivated by ZeRO paper - zero data overlap between GPUs.
It is useful when model doesn't fit on GPU.

If we see the memory usage the Optimizer is using twice as space of parameter.

3 Stage Optimization.

Data Loader > GPUs >  > Get Weights > Forwards > Get Weights > Backward's (one for one GPU) > Synchonize > Update Model's 
Here we distribute the parameter + gradient + optimizer as well.

Help reduce overall GPU memory utilization.
Support offloading to cpu if needed.
Configure sharding factor
    1 means DDP.

2.28 B parameters, same performance for DDP and FSDP

# Scaling Laws and Compute Optimal Models
## Scaling choices for pre-taining
Goal - Maximize Model Performance
    Increase dataset size
    Number of parameters.
    Constraints - Compute Budget
## Compute Budget for Training LLMs
1 petaflop/s-day = floating point operations performed at rate of 1 petaFLOP per sec for one day.
1 petaFLOP/s = 10^15 floating point operations per second.
NVIDIA V100 = 8 GPU for 24 hrs.
NVIDIA A100 = 2 GPU for 24 hrs.

A huge number of petaflop/s-day is needed to train model from scratch.

Above 3 factors, if 2 factors are fix then we can increase model efficiency by increasing 3rd one.
Smaller model trained on large dataset could perform well as large model.

Compute optimal training dataset is 20x number of parameters.
    Number of tokens.

LLaMa-65B is trained on approx. correct size.
GPT-3, OPT-175B, BLOOM are trained on less data.

# Pre-training for Domain Adaption
Legal Language - The word in legal language hardly used in outside word.
    Sometime the word used in specific context.
Medical Langauge - 'Shortcut clear to only medicine shop'.

BloombergGPT - A model for finance.
    51% financial data + 49% public data.

    Use chinchilla approach - 20x.

# Domain Specific Training: BloombergGPT
Decoder only language model.
Financial dataset comprising news articles, reports, and market data, to increase its understanding of finance and enabling it to generate finance-related natural language text.

https://arxiv.org/abs/2303.17564

# Links
https://www.amazon.com/Generative-AI-AWS-Multimodal-Applications/dp/1098159225/

https://arxiv.org/pdf/1706.03762
https://arxiv.org/pdf/2303.17564
https://arxiv.org/pdf/2203.15556
https://arxiv.org/pdf/2005.14165
https://arxiv.org/pdf/2302.13971
https://arxiv.org/pdf/2204.05832

https://huggingface.co/models
https://huggingface.co/tasks

https://arxiv.org/abs/2001.08361


https://www.coursera.org/learn/classification-vector-spaces-in-nlp

https://arxiv.org/abs/2211.05100
