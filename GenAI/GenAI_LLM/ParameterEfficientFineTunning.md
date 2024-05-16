# Parameter Efficient Fine Tuning
## Full fine tuning of Large LLM is challenging
Need lots of memory, GPU etc.
You need memory for Temp Memory; Forward Activations; Gradient; Optimizer States; Trainable Weights.

It create full copy of original LLM per task.

## PEFT
Only update small subset of parameters.
LLM with most layer frozen.
Small number of trainable layers.

LLM with additional layers of PEFT

So, number of trainable weights are smaller. 10-20% of original.

Less prone to catastrophic forgetting.

## PEFT save space and its flexible
Much smaller footprints

## PEFT Tradesoff
Parameter Efficiency
Memory Efficiency
Model Performance
Inference Cost
Traing Speed

## PEFT Methods
Selective - Select subset of initial LLM Parameters to fine-tune.
Reparameterization - Reparameterize model weights using a low rank representation - LoRA
Additive - Add trainable layers or parameters to model - Adaptor and Soft Prompts.

# PEFT technique 1 - LoRA
Low Ranked Adaption of Large Language Models (LoRA)
## Tranformer - Recap
                                Output --------
                                   ^           |
                                   |           |
                                Softmax     
                                   |           |
         Encoder -------------  Decoder
            ^                      ^
            |                      |           | 
        Embedding              Embeddings
            ^                      ^           |
            |                      | ----------
                     Inputs


Input Prompt turns into tokens.
Tokens are converted to embedding vectors.
Encoder and Decoder both have 2 types of NNs:
    Self-attention
    Feed Forward Network
The weights are learned during pre-training.
Self Attention
    embeddings are feed to self-attention layers.
    It calculate attention scores.

LoRA - Self attention
       Reduce number of parameters fine tuned in self-attention by freezing all of the models parameters.
       Inject 2 rank decomposition matrices. Its size is same as weight that we want to modify.
       Train the weights of smaller matrices.

       Steps to update model for inference 
       Matrics multiply the low rank matrices = B X A
       Add to original weigths = W + B X A

       No impact on inference latency.

Train different rank decomposition matrices for different tasks.
Update weights before inferences.

Main question is how to choose 'Rank'? It decide the number of trainable parameters.
r=1 to n, increase performance. But after n, it keep consistent.

# PEFT Technique 2 - Soft Prompts
Prompt Tunning with soft prompt. 
Objective to improve performance without changing weights.

Prompt Tuning is not Prompt Engineering.

## Prompt Tunning adds trainable "soft prompts" to the input
You add additional tokens to your prompt.
Set of trainable tokens are called soft prompts.
Its length as token embedding vectors.
    Typically 20-100 virtual tokens.
Here model get trained which virtual token to append with supervised learning.

We can use different set of soft prompts for different set of tasks.
Just change the soft prompt at the time of inference.

Prompt tunning doesn't perform well for smaller model as compare to fine tunning.
Prompt Tunning = Full Tunning for Large Model 


# Links
https://www.amazon.com/Generative-AI-AWS-Multimodal-Applications/dp/1098159225/
https://arxiv.org/pdf/2210.11416
https://research.google/blog/introducing-flan-more-generalizable-language-models-with-instruction-fine-tuning/
https://crfm.stanford.edu/helm/lite/latest/
https://openreview.net/pdf?id=rJ4km2R5t7
https://super.gluebenchmark.com/
https://aclanthology.org/W04-1013.pdf
https://arxiv.org/pdf/2009.03300
https://arxiv.org/pdf/2206.04615
https://arxiv.org/pdf/2303.15647
https://arxiv.org/pdf/2211.15583
https://arxiv.org/pdf/2106.09685
https://arxiv.org/pdf/2305.14314
https://arxiv.org/pdf/2104.08691
