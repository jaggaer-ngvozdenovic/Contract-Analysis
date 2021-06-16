from scripts.predict import run_prediction
import json

file = './cuad-data/test.json'
with open(file) as json_file:
    data = json.load(json_file)


questions = []
for i, q in enumerate(data['data'][0]['paragraphs'][0]['qas']):
    question = data['data'][0]['paragraphs'][0]['qas'][i]['question']
    questions.append(question)
contract = data['data'][0]['paragraphs'][0]['context']


with open('temp/contract.txt', 'w') as f:
    f.write(' '.join(contract.split()))


predictions = run_prediction(questions, contract, 'cuad-models/roberta-base/')


with open('temp/predictions.txt', 'w') as f:
    for i, p in enumerate(predictions):
        f.write(f"Question {i+1}: {questions[int(p)]}\nAnswer: {predictions[p]}\n\n")

