import csv
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

CSV_PATH = "C:\\Users\\batla\\OneDrive\\Desktop\\SignToSpeech\\sign_interpreter\\dataset.csv"
MODEL_PATH = "saved_model.h5"
SCALER_PATH = "scaler.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"


def read_and_normalize_csv(csv_path):
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if len(r) > 0]

    if len(rows) == 0:
        raise ValueError("CSV is empty.")

    # Determine max columns across rows
    max_cols = max(len(r) for r in rows)
    if max_cols < 2:
        raise ValueError("CSV must contain at least one feature column and one label column per row.")

    # number of feature columns = max_cols - 1 (last column is label)
    n_features = max_cols - 1

    normalized_rows = []
    for i, r in enumerate(rows):
        # ignore empty or too-short rows
        if len(r) < 2:
            continue
        feats = r[:-1]
        label = r[-1]

        # pad or truncate features to length n_features
        if len(feats) < n_features:
            feats = feats + ["0"] * (n_features - len(feats))
        elif len(feats) > n_features:
            feats = feats[:n_features]

        normalized_rows.append(feats + [label])

    # Build DataFrame
    col_names = [f"f{i}" for i in range(n_features)] + ["label"]
    df = pd.DataFrame(normalized_rows, columns=col_names)

    # Convert all feature columns to numeric, coerce errors to NaN, then fill with 0
    feature_cols = col_names[:-1]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Trim rows where label is empty
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["label"] != ""]

    df = df.reset_index(drop=True)
    return df


def build_model(input_dim, num_classes):
    model = Sequential(
        [
            Dense(256, activation="relu", input_dim=input_dim),
            Dropout(0.4),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    print("Loading and normalizing dataset...")
    df = read_and_normalize_csv(CSV_PATH)
    print(f"Dataset shape (rows, cols): {df.shape}  (features + label)")

    X = df.drop("label", axis=1).values.astype(np.float32)
    y = df["label"].values

    print(f"Features shape: {X.shape}   Labels: {len(np.unique(y))} classes")

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = to_categorical(y_enc)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_cat, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Build and train
    model = build_model(X_train.shape[1], y_cat.shape[1])
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    print("Starting training...")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nModel evaluation -> loss: {loss:.4f}  accuracy: {acc*100:.2f}%")

    # Save artifacts
    print("Saving model and preprocessing artifacts...")
    model.save(MODEL_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    print("Saved:", MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
