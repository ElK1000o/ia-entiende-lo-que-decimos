
import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score

# --- CONFIGURACIÓN ---
MODEL_NAME = "bert-base-multilingual-cased"
OUTPUT_DIR = "models/emotion_model" # Sobrescribiremos el modelo anterior
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

# --- MAPEO DE ETIQUETAS ---
emotion_map = {
    "admiration": "amor", "amusement": "alegría", "anger": "enojo",
    "annoyance": "enojo", "approval": "alegría", "caring": "amor",
    "confusion": "sorpresa", "curiosity": "sorpresa", "desire": "amor",
    "disappointment": "tristeza", "disapproval": "enojo", "disgust": "enojo",
    "embarrassment": "miedo", "excitement": "alegría", "fear": "miedo",
    "gratitude": "alegría", "grief": "tristeza", "joy": "alegría",
    "love": "amor", "nervousness": "miedo", "optimism": "alegría",
    "pride": "alegría", "realization": "sorpresa", "relief": "alegría",
    "remorse": "tristeza", "sadness": "tristeza", "surprise": "sorpresa",
    "neutral": "neutral",
}
new_labels = ["alegría", "tristeza", "enojo", "miedo", "amor", "sorpresa", "neutral"]
id2label = {i: label for i, label in enumerate(new_labels)}
label2id = {label: i for i, label in enumerate(new_labels)}

def preprocess_data():
    """
    Carga y combina el dataset go_emotions con nuestro dataset de aumentación.
    """
    # 1. Cargar el dataset principal
    print("Cargando dataset 'go_emotions'...")
    dataset_go = load_dataset("go_emotions", "simplified")
    df_go = pd.concat([
        dataset_go["train"].to_pandas(),
        dataset_go["validation"].to_pandas(),
        dataset_go["test"].to_pandas()
    ], ignore_index=True)
    print(f"Cargadas {len(df_go)} filas de go_emotions.")

    # Preprocesar go_emotions
    df_go = df_go[df_go["labels"].apply(len) == 1].copy()
    df_go["labels"] = df_go["labels"].apply(lambda x: x[0])
    original_labels = dataset_go["train"].features["labels"].feature.names
    df_go["emotion_name"] = df_go["labels"].apply(lambda x: original_labels[x])
    df_go["new_label_name"] = df_go["emotion_name"].map(emotion_map)
    df_go = df_go.dropna(subset=["new_label_name"])
    df_go["label"] = df_go["new_label_name"].apply(lambda x: label2id[x])
    df_go = df_go[["text", "label"]]
    print(f"{len(df_go)} filas procesadas de go_emotions.")

    # 2. Cargar el dataset de aumentación
    print("Cargando dataset de aumentación...")
    dataset_aug = load_dataset("json", data_files="data/emotion_augmentation.json", split="train")
    df_aug = dataset_aug.to_pandas()
    # La columna 'label' ya contiene el nombre de la emoción, la mapeamos a su ID
    df_aug["label"] = df_aug["label"].map(label2id)
    print(f"Cargadas {len(df_aug)} filas de aumentación.")

    # 3. Combinar ambos datasets
    df_combined = pd.concat([df_go, df_aug], ignore_index=True)
    print(f"Dataset final combinado con {len(df_combined)} filas.")

    return Dataset.from_pandas(df_combined)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

if __name__ == "__main__":
    dataset = preprocess_data()
    
    if dataset is None:
        print("El preprocesamiento falló. Abortando.")
    else:
        print("Cargando tokenizador...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        print("Tokenizando dataset...")
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]
        
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        print("Cargando modelo base para clasificación...")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(new_labels),
            id2label=id2label,
            label2id=label2id
        )

        print("Configurando argumentos de entrenamiento...")
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            warmup_steps=500,
            weight_decay=0.01,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        print("--- ¡Comenzando el re-entrenamiento del modelo de emoción! ---")
        trainer.train()
        print("--- Re-entrenamiento finalizado. ---")

        print(f"Guardando el modelo mejorado en '{OUTPUT_DIR}'...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("¡Nuevo modelo de emoción guardado con éxito!")
