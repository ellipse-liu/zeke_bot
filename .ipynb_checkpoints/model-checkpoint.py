import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, Concatenate, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import FastText


class discord_chatbot_300():
    def __init__(self, max_encoder_len, max_decoder_len, vocab_size, words_to_index, sentences):
        self.latent_dim = 256
        self.embedding_dim = 100
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len
        self.vocab_size = vocab_size
        
        self.embedding_matrix = self.get_embedding_vector(words_to_index=words_to_index, sentences=sentences)
        self.build_model()
        
        
    def get_embedding_vector(self, words_to_index, sentences,file_path= "glove.6B/glove.6B.100d.txt"):
        fasttext_model = FastText(sentences, vector_size=self.embedding_dim, min_count=1)

        # Load the pre-trained GloVe embeddings
        embeddings_index = {}
        with open(file_path, encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        # Define the embedding matrix using the GloVe embeddings
        embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))
        for word,i in words_to_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                embedding_matrix[i] = fasttext_model.wv[word]
                
        return embedding_matrix
    
    def build_model(self):
        self.encoder_input = Input(shape=(self.max_encoder_len,))
        self.encoder_embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, weights=[self.embedding_matrix], trainable=True)(self.encoder_input)
        self.encoder_LSTM1 = LSTM(units=self.latent_dim ,return_sequences=True, return_state=True, dropout = 0.4, recurrent_dropout = 0.3)
        self.encoder_output1, self.state_h1, self.state_c1 = self.encoder_LSTM1(self.encoder_embedding)
        
        self.encoder_LSTM2 = LSTM(self.latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.3)
        self.encoder_output2, self.state_h2, self.state_c2 = self.encoder_LSTM2(self.encoder_output1) # encoder LSTMs feed into each other

        self.encoder_LSTM3 = LSTM(self.latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.3)
        self.encoder_output, self.state_h, self.state_c = self.encoder_LSTM3(self.encoder_output2)
        
        self.decoder_input = Input(shape=(None,))
    
        self.decoder_embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, weights=[self.embedding_matrix], trainable=False)(self.decoder_input)

        # decoder LSTM layer
        self.decoder_LSTM = LSTM(self.latent_dim, return_sequences=True, return_state= True, dropout=0.4, recurrent_dropout=0.2)
        self.decoder_outputs, self.decoder_fwd_state, self.decoder_back_state = self.decoder_LSTM(self.decoder_embedding, initial_state=[self.state_h, self.state_c])

        self.decoder_dense = TimeDistributed(Dense(self.vocab_size, activation='softmax'))
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)
        
        self.training_model = Model([self.encoder_input, self.decoder_input], self.decoder_outputs)
        
    def compile(self):
        self.training_model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics = ['acc'])
        
    def fit(self, x_tr, y_tr_in, y_tr_out, x_test, y_test_in, y_test_out, ep, batch_size):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        ck = ModelCheckpoint(filepath='zeke_bot_best_weights.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        Callbacks = [es, ck]
        self.training_model.fit([x_tr,y_tr_in], y_tr_out, epochs = ep, callbacks=Callbacks, batch_size = batch_size, validation_data=(([x_test,y_test_in]), y_test_out))
        
    def build_inference_model(self):
        self.inference_encoder_model = Model(inputs= self.encoder_inputs, outputs=[self.encoder_output, self.state_h, self.state_c])

        # decoder setup
        self.decoder_state_input_h = Input(shape=(self.latent_dim,))
        self.decoder_state_input_c = Input(shape=(self.latent_dim,))
        self.decoder_hidden_state_input = Input(shape=(self.max_encoder_len, self.latent_dim))

        self.decoder_embed_i = self.decoder_embed_layer(self.decoder_inputs)

        self.decoder_output_i, self.state_h_i, self.state_c_i = self.decoder_LSTM(self.decoder_embed_i, initial_state = [self.decoder_state_input_h, self.decoder_state_input_c])

        self.decoder_output_i = self.decoder_dense(self.decoder_output_i)

        # final decoder inference model
        self.inference_decoder_model = Model([self.decoder_inputs] + [self.decoder_hidden_state_input, self.decoder_state_input_h, self.decoder_state_input_c], [self.decoder_output_i] + [self.state_h_i, self.state_c_i])


    def decode_sequence(self, input_seq, i2o, o2i):
        e_out,e_h, e_c = self.inference_encoder_model.predict(input_seq, verbose = 0)
        target_seq = np.zeros((1,1))
        target_seq[0,0] = o2i['<']

        stop_condition = False
        decoded_sentence = []

        while not stop_condition:
            (output_tokens, h, c) = self.inference_decoder_model.predict([target_seq] + [e_out, e_h, e_c], verbose = 0)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = i2o[sampled_token_index]   

            if sampled_token != '>':
                decoded_sentence += [sampled_token]

            # Exit condition: either hit max length or find the stop word.
            if (sampled_token == '>') or (len(decoded_sentence) >= self.max_decoder_len):
                stop_condition = True

            # Update the target sequence (of length 1)
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update internal states
            (e_h, e_c) = (h, c)
        return decoded_sentence
    def sentence2seq(self, a2i, input_word):
        final_seq = []
        for c in input_word:
            final_seq += [a2i[c]]
        final_seq = pad_sequences([final_seq], maxlen=self.max_encoder_len, padding='post')[0]
        return final_seq
    
    def translate(self, input_word, a2i, i2o, o2i):
        seq = self.sentence2seq(a2i, input_word).reshape(1, self.max_encoder_len)
        return self.decode_sequence(seq, i2o, o2i)
        
        