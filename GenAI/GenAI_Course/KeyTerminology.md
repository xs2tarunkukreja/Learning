# LLM (Large Language Model)
When we type in mobile, it suggest next word.
ChatGPT is also LLM.

LLM a powerful AI models designed for understanding and generating human like text.
    Text - Unserstand text, process text, generate text... text
    They know word, grammer, sentense, context all with great accuracy.

LLM is subset of Gen AI.

Special type of NN i.e. transformers. Transformer is brain for LLM.
NN output is one word at a time.
It predict only one word at a time. It is in cycle to generate complete response.

Massive NN - Billions of parameters.

Use Cases -
    Content Generation
    Chatbot and virtual assistance
    Language Translation
    Text Summarization
    QnA

# Prompt Engineering
Core component of GenAI.

Questions you ask to Alexa, Siri, Google.

A promot is a specific question, command or input that you provide to AI System to request a particular response, information or action.
    Text Generation - Summarize the keypoint of this reseach article on Gen AI and its impact on retail industry.
    Creative Writing - Write a short story about a detective solving a robbery case.
    Image Generation - Generate an image of yellow car on country side road.
    Code Generation - Write a python code to input 2 numbers and calculate their sum.

Ask clear, accurate, with context question.

PE is the process of crafting well defined and structured input queries to interact with AI System, in order to get accurate and relevant response.

Best Practices
    Clearly convey the desired response.
    Provide context or background information.
    Balance simplicity and complexity
    Iterative testing and refinement

# Embeddings
Machine don't understand text. They understand only numbers.

Embedding is numerical representation of text. They are essential for AI Model to understand and work with human language effectively.

Example
    "I eat ice cream."
    I to 20 doesn't work as it doesn't have context.
    Icecream taster "great" or My icecream melted, "great" - Different meaning due to different context.

    First LLM breack sentence into chunks or tokens. "I" "eat" "ice" "cream" 
    Chunks or Token pass to NN or Transformer and it generate embedding so, you have meaning, context, relationship .. everything.

    One work > lots of numbers (embeddings). It only understand by transformer model generated them.

    All embeddings > model > one word generated.

# Fine Tuning
More "Targeted" training for specific task

LLMs are trained for general purpose. If you are looking for specific task or on specific dataset from your organization.

Train pre-trained LLM on our own dataset.

Fine Tunning a LLM is the process of adapting a pre-trained model to perform specific tasks or to cater to a particular domain more effectively.

Three Types -
    1. Self Supervised - Domain Specific Training Data > LLM : Trained
    2. Supervised - Labeled Training Data > LLM : Trained.
    3. Reinforcement - Feedback based learning method. Its output is good, we assign good score and output bad, we assign bad score.

we are not creating anything from scratch. There is no universal solution. It is not magical one time process.

# Summary