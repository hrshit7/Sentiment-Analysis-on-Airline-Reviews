import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D, Embedding

df = pd.read_csv("./Tweets.csv")
tweet_df = df[['text', 'airline_sentiment']]
tweet_df = tweet_df[tweet_df['airline_sentiment'] != 'neutral']

sentiment_label = tweet_df.airline_sentiment.factorize()

tweet = tweet_df.text.values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweet)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(tweet)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)


embedding_vector_length = 32
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length=200))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(padded_sequence, sentiment_label[0], validation_split=0.2, epochs=5, batch_size=32)
print(history.history.keys())

def predict_sentiment():
    text = user_input.get()
    if text.strip() == "":
        messagebox.showerror("Error", "Please enter a sentence.")
        return
    
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw, maxlen=200)
    prediction = int(model.predict(tw).round().item())
    predicted_label = sentiment_label[1][prediction]
    messagebox.showinfo("Prediction", f"Predicted label: {predicted_label}")

def plot():
    plt.plot(history.history['accuracy'], label='acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.show()
    plt.savefig("Accuracy plot.jpg")
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()
    plt.savefig("Loss plot.jpg")

root = tk.Tk()
root.title("Sentiment Analysis")
root.geometry("400x250")

input_label = tk.Label(root, text="Enter a sentence:")
input_label.pack()

user_input = tk.Entry(root, width=50)
user_input.pack()

predict_button = tk.Button(root, text="Predict", command=predict_sentiment)
predict_button.pack()

accuracy_button = tk.Button(root, text="Show Plots", command=plot)
accuracy_button.pack()

root.mainloop()
