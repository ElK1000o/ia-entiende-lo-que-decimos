
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
# Directorio del dataset local
DATASET_PATH = "data/valor_social_dataset.json"
# Directorio donde se guardará el modelo afinado
OUTPUT_DIR = "models/social_value_model"
# Número de épocas de entrenamiento
NUM_EPOCHS = 5 # Más épocas porque el dataset es pequeño y de alta calidad
# Tamaño del lote
BATCH_SIZE = 8 # Lote más pequeño para datasets pequeños
# Tasa de aprendizaje
LEARNING_RATE = 5e-5 # Tasa de aprendizaje un poco más alta

# --- MAPEO DE ETIQUETAS ---
new_labels = ["positivo", "neutral", "negativo"]
id2label = {i: label for i, label in enumerate(new_labels)}
label2id = {label: i for i, label in enumerate(new_labels)}

def preprocess_data():
    """
    Carga y preprocesa el dataset local de valor social.
    """
    print(f"Cargando dataset local desde '{DATASET_PATH}'...")
    
    # Cargar directamente desde el archivo JSON
    dataset = load_dataset('json', data_files=DATASET_PATH, split='train')
    
    # Renombrar columna para consistencia
    dataset = dataset.rename_column("valor_social", "label_name")
    
    print(f"Dataset original cargado con {len(dataset)} filas.")

    def map_labels(example):
        example['label'] = label2id[example['label_name']]
        return example

    # Mapear los nombres de las etiquetas a sus IDs numéricos
    dataset = dataset.map(map_labels)
    
    print(f"Dataset final con {len(dataset)} filas y {len(new_labels)} etiquetas.")
    
    return dataset

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
    
    # --- 2. Tokenización ---
    print("Cargando tokenizador...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("Tokenizando dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Dividir en entrenamiento y evaluación (85/15 split)
    train_test_split = tokenized_dataset.train_test_split(test_size=0.15, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- 3. Carga del Modelo ---
    print("Cargando modelo base para clasificación...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(new_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
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
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        warmup_steps=50,
        weight_decay=0.01,
        report_to="none"
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
    print("--- ¡Comenzando el entrenamiento del modelo de valor social! ---")
    trainer.train()
    print("--- Entrenamiento finalizado. ---")

    # --- 7. Guardado del Modelo ---
    print(f"Guardando el mejor modelo en '{OUTPUT_DIR}'...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("¡Modelo de valor social guardado con éxito!")