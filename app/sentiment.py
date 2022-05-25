import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request
from string import punctuation
from collections import Counter

app = Flask(__name__)

# read data from text files
with open('data/reviews.txt', 'r') as f:
    reviews = f.read()
with open('data/labels.txt', 'r') as f:
    labels = f.read()
    
# get rid of punctuation
reviews = reviews.lower() # lowercase, standardize
all_text = ''.join([c for c in reviews if c not in punctuation])

# split by new lines and spaces
reviews_split = all_text.split('\n')
all_text = ' '.join(reviews_split)

# create a list of words
words = all_text.split()

## Build a dictionary that maps words to integers
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

########### Project - Sentiment Analysis ###########
@app.route('/', methods=['POST', 'GET'])
def sentiment_page():
    return render_template("index.html")

########### Sentiment Analysis - Results ###########
@app.route('/sentiment_results', methods=['POST', 'GET'])
def sentiment_results():

    if request.method == "POST":

        class SentimentRNN(nn.Module):

            """
            The RNN model that will be used to perform Sentiment analysis.
            """

            def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
                """
                Initialize the model by setting up the layers.
                """
                super(SentimentRNN, self).__init__()

                self.output_size = output_size
                self.n_layers = n_layers
                self.hidden_dim = hidden_dim

                # embedding and LSTM layers
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                                    dropout=drop_prob, batch_first=True)

                # dropout layer
                self.dropout = nn.Dropout(0.3)

                # linear and sigmoid layers
                self.fc = nn.Linear(hidden_dim, output_size)
                self.sig = nn.Sigmoid()


            def forward(self, x, hidden):
                """
                Perform a forward pass of our model on some input and hidden state.
                """
                batch_size = x.size(0)

                # embeddings and lstm_out
                x = x.long()
                embeds = self.embedding(x)
                lstm_out, hidden = self.lstm(embeds, hidden)

                lstm_out = lstm_out[:, -1, :] # getting the last time step output

                # dropout and fully-connected layer
                out = self.dropout(lstm_out)
                out = self.fc(out)
                # sigmoid function
                sig_out = self.sig(out)

                # return last sigmoid output and hidden state
                return sig_out, hidden


            def init_hidden(self, batch_size):
                ''' Initializes hidden state '''
                # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
                # initialized to zero, for hidden state and cell state of LSTM
                weight = next(self.parameters()).data

                hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(), weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

                return hidden

        # Instantiate the model w/ hyperparams
        vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding + our word tokens
        output_size = 1
        embedding_dim = 400
        hidden_dim = 256
        n_layers = 2

        model = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

        # Load trained model
        model.load_state_dict(torch.load('model/sentiment_model.pt'))
        model.eval()

        def tokenize_review(test_review):

            test_review = test_review.lower() # lowercase
            # get rid of punctuation
            test_text = ''.join([c for c in test_review if c not in punctuation])

            # splitting by spaces
            test_words = test_text.split()

            # tokens
            test_ints = []
            test_ints.append([vocab_to_int.get(word, 0) for word in test_words])

            return test_ints

        def pad_features(reviews_ints, seq_length):
            ''' Return features of review_ints, where each review is padded with 0's
                or truncated to the input seq_length.
            '''

            # getting the correct rows x cols shape
            features = np.zeros((len(reviews_ints), seq_length), dtype=int)

            # for each review, I grab that review and
            for i, row in enumerate(reviews_ints):
                features[i, -len(row):] = np.array(row)[:seq_length]

            return features

        def predict(model, test_review, sequence_length=200):

            # tokenize review
            test_ints = tokenize_review(test_review)

            # pad tokenized sequence
            seq_length=sequence_length
            features = pad_features(test_ints, seq_length)

            # convert to tensor to pass into your model
            feature_tensor = torch.from_numpy(features)

            batch_size = feature_tensor.size(0)

            # initialize hidden state
            h = model.init_hidden(batch_size)

            # get the output from the model
            output, h = model(feature_tensor, h)

            return output.item()

    UserInput = request.form["textArea1"]

    seq_length=200 # good to use the length that was trained on
    result = predict(model, UserInput, seq_length)
    rounded = round(result)

    return render_template("results.html", UserInput=UserInput, result=result, rounded=rounded)