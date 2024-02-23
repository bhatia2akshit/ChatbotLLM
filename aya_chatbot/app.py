from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


app = Flask(__name__)

device = "cuda"  # the device to load the model onto

checkpoint = "CohereForAI/aya-101"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype=torch.float16).to(device)

@app.route('/')
def index():
    return render_template('frontend.html')

@app.route('/query', methods=['POST'])
def query():
    user_input = request.json.get('message')  # Get the user's message from the request

    tur_inputs = tokenizer.encode(user_input, return_tensors="pt").to(device)

    tur_outputs = model.generate(tur_inputs, max_new_tokens=1000)
    encodeds = tokenizer.decode(tur_outputs[0])

    response = {'response': encodeds}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
