# Evaluador de Sistema RAG - NASA Project

Este script realiza una evaluación automatizada por lotes de un sistema de Generación Aumentada por Recuperación (RAG). Utiliza la librería **Ragas** para medir la calidad de las respuestas comparándolas con un dataset de referencia.

## 📋 Funcionalidades
- **Evaluación por Frase**: Calcula métricas específicas para cada pregunta del dataset.
- **Métricas Globales**: Al finalizar el proceso, calcula el promedio total de rendimiento del modelo.
- **Exportación**: Genera un archivo CSV con todos los resultados detallados y una fila final de promedios.

## 🛠️ Requisitos Previos
Antes de ejecutar el script, asegúrate de tener configurado:
- Una clave de API de OpenAI en un archivo `.env` (`OPENAI_API_KEY=tu_clave`).
- El archivo de dataset `evaluation_dataset.txt` con el formato:
  '''text
  Q - ¿Cuál es la misión de Artemis?
  A - La misión Artemis busca llevar a la primera mujer y al próximo hombre a la Luna.

## 🚀 Ejecución
Primero de todo se debe eejcutar el script para generar los embeddings, para ello usa el siguiente comando:
'''bash
python embedding_pipeline.py

Para iniciar la evaluación del dataset, usa el siguiente comando:
'''bash
python run_evaluation_dataset.py