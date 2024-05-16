# GenAI Use Case - Summarize Dialogue
8vCPU + 32 GB RAM
Python 3

%pip install -U datasets==2.17.0
%pip install --upgrade pip
%pip install --disable-pip-version-check \
    torch==1.13.1 \
    torchdata==0.5.1 --quiet
%pip install transformers==4.27.2 --quiet

dataset - library maintained by Hugging Face. It have data sets for train, fine tune etc.

Imports

Using Public Data named as dialoguesum > Explore some data - Conversation and Summary.
model = flan-t5
tokenizer // based on model flan-t5. Text to Embeddings.

Now use some example to generate summary. - It is not good.

Instruction Prompt
    Different Prompt Template
    Zero Shot
    One Shot
    Few Shot

    One Shot = 4 Shot. 

Configuration Parameters
    sampling, temperature.



