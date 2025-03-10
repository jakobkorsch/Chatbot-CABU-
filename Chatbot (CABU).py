# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 14:35:28 2025                                       Chatbot (CABU)

@author: jakob
"""





import random
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, Embedding
from tensorflow.keras.optimizers import SGD
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import threading
import queue
import time
import speech_recognition as sr
import pyttsx3
import openai  # Für GPT-Fallback
from farasa.segmentation import segmentLine  # Farasa für arabische Texte

# OpenAI API-Schlüssel
openai.api_key = "your Open API Key"

# NLTK-Downloads
nltk.download('punkt')
nltk.download('wordnet')

# Initialisiere Spracherkennung und Sprachausgabe
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Daten laden und vorbereiten
lemmatizer = WordNetLemmatizer()
with open('arabic_intents.json', encoding='utf-8') as f:
    intents = json.load(f)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]), dtype=np.float32)
train_y = np.array(list(training[:, 1]), dtype=np.float32)

# Modell erstellen (RNN)
model = Sequential()
model.add(Embedding(input_dim=len(words), output_dim=128, input_length=len(train_x[0])))
model.add(SimpleRNN(128, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Modell im .keras-Format speichern
model.save('chatbot_model.keras')

# Wörter und Klassen speichern
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)


















# GPT-Fallback-Funktion
def ask_gpt(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()  
    except Exception as e:
        print(f"Fehler bei der GPT-Anfrage: {e}")
        return None

# Funktionen zur Verarbeitung der Eingabe
def clean_up_sentence(sentence):
    sentence_words = segmentLine(sentence)
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def classify_local(sentence):
    ERROR_THRESHOLD = 0.25
    input_data = bow(sentence, words, show_details=False)
    res = model.predict(np.array([input_data]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Debugging-Ausgabe der erkannten Intents und ihrer Wahrscheinlichkeiten
    print("Erkannte Intents und Wahrscheinlichkeiten:", results)
    
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Speichern unbekannter Eingaben
def save_unknown_input(sentence):
    with open('unknown_inputs.txt', 'a') as f:
        f.write(sentence + "\n")

def response(sentence, userID='123', show_details=True):
    # Klassifikation der Eingabe, um den Intent zu erkennen
    results = classify_local(sentence)
    
    if results:
        # Den besten Treffer aus den Ergebnissen auswählen
        intent = results[0]['intent']
        
        # Suchen Sie den entsprechenden Intent in den JSON-Daten
        for i in intents['intents']:
            if i['tag'] == intent:
                if show_details:
                    print(f"Intent erkannt: {i['tag']} mit Wahrscheinlichkeit: {results[0]['probability']}")
                # Antwort aus den JSON-Daten zurückgeben
                return random.choice(i['responses'])
    
    # Unbekannte Eingabe speichern, falls keine passende Antwort gefunden wurde
    save_unknown_input(sentence)
    
    # Wenn kein Intent gefunden wurde, Fall-back auf GPT
    gpt_response = ask_gpt(f"Der Benutzer hat gefragt: '{sentence}'. Wie soll ich antworten?")
    if gpt_response:
        return gpt_response
    else:
        return "عذرًا، لم أتمكن من إنشاء رد. يرجى المحاولة مرة أخرى لاحقًا."

# Funktionen für Spracherkennung und Texteingabe
def listen_for_speech(q):
    global stop_threads
    while not stop_threads:
        try:
            with sr.Microphone() as source:
                print("Sprich jetzt...")
                audio = recognizer.listen(source, timeout=20)
                message = recognizer.recognize_google(audio, language='ar-SA')
                print(f"Du hast gesagt: {message}")
                q.put(message)
        except sr.UnknownValueError:
            print("Konnte die Spracheingabe nicht erkennen.")
        except sr.WaitTimeoutError:
            print("Es wurde 20 Sekunden lang nichts gehört. Stopping.")
            stop_threads = True
            break
        
        if stop_threads or (message and message.lower() in ["مع السلامة", "إلى اللقاء", "وداعاً", "باي"]):
            break

def listen_for_text(q):
    global stop_threads
    while not stop_threads:
        message = input("أدخل رسالة (أو اكتب 'وداعاً' للإيقاف): ")
        q.put(message)
        if stop_threads or message.lower() in ["مع السلامة", "إلى اللقاء", "وداعاً", "باي"]:
            break

# Benutzer wählt, ob er Text oder Sprache verwenden möchte
def choose_input_method():
    while True:
        choice = input("Möchtest du mit 'Text' oder 'Sprache' eingeben? (text/sprache): ").strip().lower()
        if choice == 'text':
            return 'text'
        elif choice == 'sprache':
            return 'sprache'
        else:
            print("Ungültige Eingabe. Bitte wähle 'text' oder 'sprache'.")

# Hauptprogramm
stop_threads = False
q = queue.Queue()

# Benutzer wählt, ob er Text oder Sprache verwenden möchte
input_method = choose_input_method()

# Je nach Auswahl wird der entsprechende Thread gestartet
if input_method == 'sprache':
    speech_thread = threading.Thread(target=listen_for_speech, args=(q,))
    speech_thread.daemon = True
    speech_thread.start()
elif input_method == 'text':
    text_thread = threading.Thread(target=listen_for_text, args=(q,))
    text_thread.daemon = True
    text_thread.start()

last_input_time = time.time()

while not stop_threads:
    try:
        message = q.get(timeout=20)  # Timeout nach 10 Sekunden
        if stop_threads or message.lower() in ["مع السلامة", "إلى اللقاء", "وداعاً", "باي"]:
            break
        last_input_time = time.time()
        chatbot_response = response(message)
        
        # Wenn Eingabemethode 'sprache', dann Text und Sprache ausgeben
        if input_method == 'sprache':
            print("Chatbot: ", chatbot_response)
            engine.say(chatbot_response)
            engine.runAndWait()
        elif input_method == 'text':
            print("Chatbot: ", chatbot_response)
    except queue.Empty:
        if time.time() - last_input_time >= 10:
            print("لا توجد مدخلات في آخر 10 ثوان")
            stop_threads = True
            break

print("تم إنهاء الجلسة.")

