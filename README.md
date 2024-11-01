﻿# Data Warehouse - Videojuegos

Este proyecto fue desarrollado para la materia **Almacenes de Datos 24B** bajo la supervisión de la profesora **Armida Griselda Vázquez Curiel**. El objetivo es construir un Data Warehouse que permita analizar tendencias en videojuegos, utilizando un dataset de **Kaggle**. Con esta herramienta, se busca predecir los videojuegos y géneros que podrían interesar más al público, basándose en las reseñas y opiniones de los usuarios.

## Descripción del Proyecto
El Data Warehouse permite almacenar, organizar y analizar datos históricos de videojuegos para identificar tendencias y patrones de popularidad. A través de técnicas de aprendizaje automático, es posible predecir qué géneros o títulos podrían captar mayor atención, lo cual es valioso para estudios de mercado y toma de decisiones en la industria de videojuegos.

## Requisitos de Instalación

Para ejecutar este proyecto, se requiere instalar las siguientes librerías y herramientas:

```bash
pip install streamlit pandas numpy mariadb scikit-learn plotly
sudo apt install mariadb-server mariadb-client
```

## Instrucciones de Ejecución
Para iniciar la aplicación, usa el siguiente comando en la terminal:

```bash
streamlit run main.py
```
Este comando desplegará la interfaz del proyecto en un navegador web, donde podrás interactuar con las visualizaciones y los análisis predictivos del dataset de videojuegos.

## Dataset
El dataset utilizado fue descargado de Kaggle y contiene información detallada sobre distintos videojuegos, incluyendo reseñas, calificaciones y métricas de popularidad que permiten realizar análisis detallados.\
https://www.kaggle.com/datasets/arnabchaki/popular-video-games-1980-2023?resource=download

