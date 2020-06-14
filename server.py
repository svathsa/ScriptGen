import os
import helper
import torch
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, request, render_template, redirect, url_for
from rnn import RNN

app = Flask(__name__, static_url_path='/static')

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
vocab_to_int['<PAD>'] = 0
int_to_vocab[0] = '<PAD>'
pad_word = helper.SPECIAL_WORDS['PADDING']
trained_rnn = RNN(len(vocab_to_int)-1, len(vocab_to_int)-1, 200, 250, 2)
trained_rnn.load_state_dict(torch.load('./save/rnn1.pt', map_location=torch.device('cpu')))
trained_rnn.eval()

def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def get_prediction(rnn, prime_id, int_to_vocab, token_dict, pad_value, predict_len=100):
    # First we initialize the input sequence with the prime word.
    # We pad the entire row with 0. The shape of the sequence is
    # 1 x sequence_length
    sequence_length = 10
    current_seq = np.full((1, sequence_length), pad_value)
    current_seq[-1][-1] = prime_id
    # all our predictions will be stored in a predicted 
    # sequence which is initially set to the prime word
    # passed into the function
    predicted = [int_to_vocab[prime_id]]
    for _ in range(predict_len):
        if torch.cuda.is_available():
            current_seq = torch.LongTensor(current_seq).cuda()
        else:
            current_seq = torch.LongTensor(current_seq)
        
        # initialize the hidden state
        hidden = rnn.init_hidden(current_seq.size(0))
        
        # get the output of the RNN
        output, _ = rnn(current_seq, hidden)
        p = F.softmax(output, dim=1).data
        if torch.cuda.is_available():
            p = p.cpu()
        
        # Use top k sampling to get the index of the next word
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()
        
        # select the likely next word index with some element of randomness
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i, p=p/p.sum())
        
        # retrieve that word from the dictionary
        word = int_to_vocab[word_i]
        predicted.append(word)
        
        if torch.cuda.is_available():
            current_seq = current_seq.cpu()
        current_seq = np.roll(current_seq, -1, 1) #np.roll will left-shift with -1 by 1.
        current_seq[-1][-1] = word_i
    
    for i, word in enumerate(predicted):
        if word[len(word)-1] == ':':
            predicted[i] = '<b>' + word.upper() + '</b>'
        elif i > 0 and (predicted[i-1][len(predicted[i-1]) - 1] == '.' or predicted[i-1][len(predicted[i-1]) - 1] == '>'):
            predicted[i] = word[0].upper() + word[1:]
        if word == 'i':
            predicted[i] = 'I'
        if i == len(predicted) - 1 and predicted[i][len(predicted[i])-1] != '.':
            predicted[i] += '.'
    gen_sentences = ' '.join(predicted)
    
    # Replace punctuation token
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
    gen_sentences = gen_sentences.replace('\n', '<br>')
    gen_sentences = gen_sentences.replace('(', '(')
    gen_sentences = '<div class="script_body"> ' + gen_sentences + " </div>"
    # return all the sentences
    return gen_sentences


@app.route('/', methods=['GET'])
def render_page():
    return render_template('index.html', glon='')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        prime_word = request.form['prime_word']
        gen_len = request.form['gen_len']
        if not is_int(gen_len) or int(gen_len) > 2500:
            glon = '<div class="script_body"> ' + "Please enter a valid number &#128522;." + '</div>'
            return render_template('index.html', glon=glon)
        prime_word = prime_word.lower()
        if prime_word not in vocab_to_int:
            glon = '<div class="script_body"> ' + "Please enter a valid Seinfeld character &#128522;." + '</div>'
            return render_template('index.html', glon=glon)
        gen_len = int(gen_len)
        generated_script = get_prediction(trained_rnn, vocab_to_int[prime_word + ':'], int_to_vocab, token_dict, vocab_to_int[pad_word], gen_len)
        return render_template('index.html', glon=generated_script)

if __name__ == '__main__':
    app.run(debug=False,port=os.getenv('PORT',5000))
