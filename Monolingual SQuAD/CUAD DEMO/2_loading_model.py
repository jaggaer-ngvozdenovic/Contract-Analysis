# CONTRACT REVIEW

## SUMMARY

### Functionalities:
# Loads the associated fine tuned model for CUAD dataset.
# Tests model with sample query.
# Uses a simple approach to extract the model prediction.

### Problems:
# What if the model predicts an end token that is before the start token in the contract?
# Start and end tokens are predicted independently by the model, so some logic needs to be implemented for that
# Most models accepts only 512 tokens (question and contract combined).
# Most contracts are much longer than 512 words and the answer to a specific contract review query might, for example, be on page 27 of the contract.


## CODE

# Generic model class that will be instantiated as one of the model classes of the library (with a question answering head)
from transformers import AutoModelForQuestionAnswering
# Generic tokenizer class that will be instantiated as one of the tokenizer classes of the library
from transformers import AutoTokenizer
import torch
# works with json files
import json


### LOAD MODEL ###
# Instantiates one of the model classes of the library (with a question answering head) from a configuration.
model = AutoModelForQuestionAnswering.from_pretrained('./cuad-models/roberta-base/')
# Prepares the inputs for a model
# use_fast=False because QuestionAnswering model is not (yet) compatible with fast tokenizers which have smarter overflow handling
tokenizer = AutoTokenizer.from_pretrained('./cuad-models/roberta-base/', use_fast=False)

### TEST MODEL ###
# Load contracts and associated queries
with open('./cuad-data/CUADv1.json') as json_file:
    data = json.load(json_file)

# Load the third query of the first contract ("What is the date of the contract?")
question = data['data'][0]['paragraphs'][0]['qas'][2]['question']
# Display the first 100 words of the contract
paragraph = ' '.join(data['data'][0]['paragraphs'][0]['context'].split()[:100])

### MAKING A FIRST TEST PREDICTION ###
# The way Q&A models work is that question and contract are concatenated (separated by a special token),
# tokenized (i.e preparing the input so the model can understand it), and then fed into the model.
# The model will then provide two outputs: the start logits and the end logits.
# The start logits describe the probability for each word in the string to be the beginning of the answer for the question.
# Similarly, the end logits describe the probabilities for each word to be the end of the answer.
# In order to get the best prediction from the model all we now have to do is pick the start and end tokens with the highest probabilities.

# Concatenates & encodes question and paragraph
encoding = tokenizer.encode_plus(text=question, text_pair=paragraph)
# Extracts the embeddings for model prediction
inputs = encoding['input_ids']
# Get the tokens
tokens = tokenizer.convert_ids_to_tokens(inputs)

# make model prediction
outputs = model(input_ids=torch.tensor([inputs]))

# get the start and end logits
start_scores = outputs.start_logits
end_scores = outputs.end_logits

# libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
#sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (16,8)

# Pull the scores out of PyTorch Tensors and convert them to 1D numpy arrays.
s_scores = start_scores.detach().numpy().flatten()
e_scores = end_scores.detach().numpy().flatten()

# We'll use the tokens as the x-axis labels. In order to do that, they all need
# to be unique, so we'll add the token index to the end of each one.
token_labels = []
for (i, token) in enumerate(tokens):
    token_labels.append('{:} - {:>2}'.format(tokenizer.convert_tokens_to_string(token), i))

# Create a barplot showing the start word score for all of the tokens.
ax = sns.barplot(x=token_labels[80:120], y=s_scores[80:120], ci=None)

# Turn the xlabels vertical.
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

# Turn on the vertical grid to help align words to scores.
ax.grid(True)

plt.title('Start Word Scores')

plt.show()

# Create a barplot showing the end word score for all of the tokens.
ax = sns.barplot(x=token_labels[80:120], y=e_scores[80:120], ci=None)

# Turn the xlabels vertical.
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

# Turn on the vertical grid to help align words to scores.
ax.grid(True)

plt.title('End Word Scores')

plt.show()

# Retrieve start and end tokens with the highest probability
start_index = torch.argmax(start_scores)
end_index = torch.argmax(end_scores)

# Retrieve the answer predicted by the model
answer = tokenizer.convert_tokens_to_string(tokens[start_index:end_index+1])
answer.strip()





