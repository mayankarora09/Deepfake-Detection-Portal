import os
import librosa
import numpy as np
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# 1. CONFIG
REAL_DIR = 'Audio_Dataset/REAL'
FAKE_DIR = 'Audio_Dataset/FAKE'
CHUNK_DIR = 'chunks'
CHUNK_DURATION = 3  # in seconds
SAMPLE_RATE = 22050

os.makedirs(f"{CHUNK_DIR}/real", exist_ok=True)
os.makedirs(f"{CHUNK_DIR}/fake", exist_ok=True)

# 2. AUDIO CHUNKING 
def split_audio(file_path, output_dir, chunk_length=CHUNK_DURATION):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    total_samples = len(y)
    samples_per_chunk = chunk_length * sr
    filename = os.path.splitext(os.path.basename(file_path))[0]

    for i in range(0, total_samples, samples_per_chunk):
        chunk = y[i:i + samples_per_chunk]
        if len(chunk) == samples_per_chunk:
            chunk_name = f"{filename}_chunk{i//samples_per_chunk}.wav"
            sf.write(os.path.join(output_dir, chunk_name), chunk, sr)

# Chunk all real and fake files
for fname in os.listdir(REAL_DIR):
    split_audio(os.path.join(REAL_DIR, fname), f"{CHUNK_DIR}/real")

for fname in os.listdir(FAKE_DIR):
    split_audio(os.path.join(FAKE_DIR, fname), f"{CHUNK_DIR}/fake")

# 3. METADATA CREATION 
data = []
for label, folder in enumerate(['real', 'fake']):
    folder_path = os.path.join(CHUNK_DIR, folder)
    for fname in os.listdir(folder_path):
        data.append((os.path.join(folder_path, fname), label))

metadata = pd.DataFrame(data, columns=['filepath', 'label'])
metadata.to_csv("metadata.csv", index=False)
 
#4. MFCC EXTRACTION 
def extract_mfcc(file_path, max_pad_len=174):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=CHUNK_DURATION)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc

X = []
y = []

for i, row in metadata.iterrows():
    mfcc = extract_mfcc(row['filepath'])
    X.append(mfcc)
    y.append(row['label'])

X = np.array(X)
X = X[..., np.newaxis]  # Add channel dimension
y = np.array(y)

# 5. TRAIN-TEST SPLIT  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. CNN MODEL 
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(40, 174, 1)),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.3),

    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 7. TRAINING
checkpoint = ModelCheckpoint("deepfake_audio_model.h5", save_best_only=True, monitor='val_accuracy', mode='max')
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint])

# 8. EVALUATION 
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc*100:.2f}%")

# 9. SAVE FINAL MODEL
model.save("final_deepfake_audio_model.h5")
