
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

# Modelo base a utilizar. Aunque 'liar' está en inglés, usaremos el mismo
# modelo base para mantener consistencia. Para un rendimiento óptimo en producción,
# se podría usar un modelo en inglés y luego traducir la entrada del usuario.
MODEL_NAME = "bert-base-multilingual-cased"
# Directorio donde se guardará el modelo afinado
OUTPUT_DIR = "models/veracity_model"
# Número de épocas de entrenamiento
NUM_EPOCHS = 4 # Un poco más de épocas ya que el dataset es más pequeño
# Tamaño del lote
BATCH_SIZE = 16
# Tasa de aprendizaje
LEARNING_RATE = 3e-5

# --- MAPEO DE ETIQUETAS ---
# Mapeo de las 6 etiquetas de 'liar' a nuestras 3 categorías
veracity_map = {
    0: "falso",      # 'false'
    1: "dudoso",     # 'half-true'
    2: "verdadero",  # 'mostly-true'
    3: "verdadero",  # 'true'
    4: "dudoso",     # 'barely-true'
    5: "falso"       # 'pants-fire'
}

# Lista final de etiquetas
new_labels = ["verdadero", "dudoso", "falso"]
id2label = {i: label for i, label in enumerate(new_labels)}
label2id = {label: i for i, label in enumerate(new_labels)}

def preprocess_data():
    """
    Carga, filtra y preprocesa el dataset 'liar'.
    """
    print("Cargando dataset 'chengxuphd/liar2'...")

    # Usamos una versión moderna del dataset 'liar' en formato Parquet
    dataset = load_dataset("chengxuphd/liar2")
    
    # Concatenar todos los splits (train, validation, test)
    df_train = dataset["train"].to_pandas()
    df_validation = dataset["validation"].to_pandas()
    df_test = dataset["test"].to_pandas()
    df = pd.concat([df_train, df_validation, df_test])

    # Renombrar columnas para consistencia
    df = df.rename(columns={"statement": "text", "label": "original_label"})
    print(f"Dataset original cargado con {len(df)} filas.")

    # 1. Mapear la etiqueta original a nuestra nueva categoría
    df["new_label_name"] = df["original_label"].map(veracity_map)
    # 2. Asignar el ID numérico final a la nueva etiqueta
    df["label"] = df["new_label_name"].apply(lambda x: label2id[x])
    print(f"Dataset final con {len(df)} filas y {len(new_labels)} etiquetas.")

    # Convertir de nuevo a un objeto Dataset de Hugging Face
    final_dataset = Dataset.from_pandas(df[["text", "label"]])
    return final_dataset

def tokenize_function(examples):
    """
    Tokeniza el texto.
    """
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=256)

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
        label2id=label2id,
        ignore_mismatched_sizes=True # Necesario porque el modelo base no fue pre-entrenado para 3 etiquetas
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
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        warmup_steps=100,
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
    print("--- ¡Comenzando el entrenamiento del modelo de veracidad! ---")
    trainer.train()
    print("--- Entrenamiento finalizado. ---")

    # --- 7. Guardado del Modelo ---
    print(f"Guardando el mejor modelo en '{OUTPUT_DIR}'...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("¡Modelo de veracidad guardado con éxito!")