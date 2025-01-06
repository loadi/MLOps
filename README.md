# MLOps

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Описание проекта
Проект реализует:
- подготовку дата-сета EMNIST
- обучение многослойного перцептрона
- экспорт модели в формате ONNX
- создание веб-приложения с использованием фреймворка Gradio

## Установка
Список зависимостей находится в файле [requirements.txt](requirements.txt).
Пример обучения и экспорта модели находится в Jupyter-ноутбуке [1.0-model-train-example.ipynb](notebooks%2F1.0-model-train-example.ipynb)

## Запуск веб-приложения
Для запуска веб-приложения достаточно запустить [app.py](app%2Fapp.py).
Веб-приложение будет запущено по адресу http://127.0.0.1:8080