
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
import inspect

# --- CONFIGURACIÓN ---
# Modelo base a utilizar
MODEL_NAME = "bert-base-multilingual-cased"
# Directorio donde se guardará el modelo afinado
OUTPUT_DIR = "models/emotion_model"
# Número de épocas de entrenamiento
NUM_EPOCHS = 3
# Tamaño del lote
BATCH_SIZE = 16
# Tasa de aprendizaje
LEARNING_RATE = 2e-5

# --- MAPEO DE ETIQUETAS ---
# Mapeo de las 28 emociones de go_emotions a nuestras 7 categorías simplificadas
emotion_map = {
    "admiration": "amor",
    "amusement": "alegría",
    "anger": "enojo",
    "annoyance": "enojo",
    "approval": "alegría",
    "caring": "amor",
    "confusion": "sorpresa",
    "curiosity": "sorpresa",
    "desire": "amor",
    "disappointment": "tristeza",
    "disapproval": "enojo",
    "disgust": "enojo",
    "embarrassment": "miedo",
    "excitement": "alegría",
    "fear": "miedo",
    "gratitude": "alegría",
    "grief": "tristeza",
    "joy": "alegría",
    "love": "amor",
    "nervousness": "miedo",
    "optimism": "alegría",
    "pride": "alegría",
    "realization": "sorpresa",
    "relief": "alegría",
    "remorse": "tristeza",
    "sadness": "tristeza",
    "surprise": "sorpresa",
    "neutral": "neutral",
}

# Lista final de etiquetas
new_labels = ["alegría", "tristeza", "enojo", "miedo", "amor", "sorpresa", "neutral"]
id2label = {i: label for i, label in enumerate(new_labels)}
label2id = {label: i for i, label in enumerate(new_labels)}

def preprocess_data():
    """
    Carga, filtra y preprocesa el dataset go_emotions.
    """
    print("Cargando dataset 'go_emotions' (configuración simplificada)...")
    dataset = load_dataset("go_emotions", "simplified")
    
    # Concatenar todos los splits para tener un único dataset grande
    df_train = dataset["train"].to_pandas()
    df_validation = dataset["validation"].to_pandas()
    df_test = dataset["test"].to_pandas()
    df = pd.concat([df_train, df_validation, df_test], ignore_index=True)
    
    print(f"Dataset completo cargado con {len(df)} filas.")

    # 1. Filtrar por entradas con una sola emoción (la columna 'labels' es una lista)
    df = df[df["labels"].apply(len) == 1].copy()
    # Extraer el único ID de la lista
    df["labels"] = df["labels"].apply(lambda x: x[0])
    print(f"Filtrado a {len(df)} filas con una sola emoción.")

    # 2. Mapear la etiqueta original (ID) a su nombre
    # Para un feature de tipo List(ClassLabel(...)), la ruta correcta es .feature.names
    original_labels = dataset["train"].features["labels"].feature.names
    df["emotion_name"] = df["labels"].apply(lambda x: original_labels[x])

    # 3. Mapear el nombre de la emoción a nuestra nueva categoría simplificada
    df["new_label_name"] = df["emotion_name"].map(emotion_map)

    # 4. Eliminar filas que no correspondan a ninguna de nuestras categorías
    df = df.dropna(subset=["new_label_name"])
    
    # 5. Renombrar 'labels' a 'original_label' para evitar confusión
    df = df.rename(columns={"labels": "original_label"})

    # 6. Asignar el ID numérico final a la nueva etiqueta en la columna 'label'
    df["label"] = df["new_label_name"].apply(lambda x: label2id[x])
    
    print(f"Dataset final listo con {len(df)} filas y {len(new_labels)} etiquetas.")
    
    # Convertir de nuevo a un objeto Dataset de Hugging Face
    final_dataset = Dataset.from_pandas(df[["text", "label"]])
    
    return final_dataset

def tokenize_function(examples):
    """
    Tokeniza el texto.
    """
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)

def compute_metrics(eval_pred):
    """
    Calcula métricas de evaluación (accuracy y F1-score).
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

if __name__ == "__main__":
    # --- 1. Carga y Preprocesamiento de Datos ---
    dataset = preprocess_data()
    
    if dataset is None:
        print("El preprocesamiento falló. Abortando el script.")
    else:
        # --- 2. Tokenización ---
        print("Cargando tokenizador...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        print("Tokenizando dataset...")
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Dividir en entrenamiento y evaluación (90/10 split)
        train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]
        
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # --- 3. Carga del Modelo ---
        print("Cargando modelo base para clasificación...")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(new_labels),
            id2label=id2label,
            label2id=label2id
        )

        # --- 4. Configuración del Entrenamiento ---
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
            report_to="none" # Desactiva reportes a wandb/tensorboard etc.
        )

        # --- 5. Creación del Trainer ---
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # --- 6. Entrenamiento ---
        print("--- ¡Comenzando el entrenamiento del modelo de emoción! ---")
        trainer.train()
        print("--- Entrenamiento finalizado. ---")

        # --- 7. Guardado del Modelo ---
        print(f"Guardando el mejor modelo en '{OUTPUT_DIR}'...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("¡Modelo de emoción guardado con éxito!")
