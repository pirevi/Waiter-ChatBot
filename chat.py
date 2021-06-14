import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

file = "data.pth"
data = torch.load(file)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state) # loads all the saved weights
model.eval()

bot_name = "Simbi"
print("Let's chat! type 'quit' to exit")
while True:
    sentence = input('You: ')
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape([1, X.shape[0]]) # input shape
    X = torch.from_numpy(X) # convert to torch tensor

    output = model(X) # gives likelihood
    _, predicted = torch.max(output, dim=1) # get index of max value
    tag = tags[predicted.item()] # gets the tag name

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand..")
