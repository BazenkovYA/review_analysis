import openai
import os
import pandas as pd
import glob
from typing import Optional
from pandas import DataFrame  # Импорт класса DataFrame напрямую из pandas
import logging
# Настройка логирования
logging.basicConfig(filename=os.path.join(os.getcwd(), 'Logs', 'project.log'),
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


def get_api_key() -> Optional[str]:
    """
    Получение ключа API из переменных окружения.
    Возвращает ключ API, если он доступен, иначе выводит сообщение об ошибке.
    """
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    if api_key:
        logging.info("Текущий ключ актуальный.")
        return api_key
    else:
        logging.warning("Необходимо проверить ключ, текущий не актуален.")
        return None



def load_data(filepath: str, sheet_name: str = "Data") -> Optional[DataFrame]:
    """
    Загрузка данных из Excel файла.
    """
    try:
        return pd.read_excel(filepath, sheet_name=sheet_name, usecols=["email", "review text", "date"])
    except ValueError as e:
        logging.warning(f"В файле {filepath} нет данных: {e}")
        return None


def rate_review(review_text: str, api_key: str) -> int:
    """
    Оценка отзыва с использованием OpenAI API.
    """
    # Загрузка текста запроса из файла
    with open('prompt.txt', 'r', encoding='utf-8') as file:
        prompt_template = file.read()

    # Форматирование текста запроса с учетом текущего отзыва
    prompt = prompt_template.format(review_text=review_text)

    response = openai.Completion.create(
        api_key=api_key,
        model="text-davinci-003",  # Подставьте актуальную модель, если потребуется
        prompt=prompt,
        temperature=0.3,
        max_tokens=1
    )
    rating: str = response.choices[0].text.strip()
    return int(rating)


def process_files(api_key: str) -> None:
    """
    Обработка всех файлов Excel в директории Data.
    """
    path: str = os.path.join(os.getcwd(), 'Data', '*.xlsx')
    files: list[str] = glob.glob(path)
    if not files:
        logging.info("Нет файлов для обработки.")
        return

    for filename in files:
        dataframe: Optional[DataFrame] = load_data(filename)
        if dataframe is None or dataframe.empty:
            logging.info(f"Файл {filename} пустой или содержит неправильный формат данных.")
            continue

        dataframe["rate"] = dataframe["review text"].apply(lambda x: rate_review(x, api_key))
        base_filename: str = os.path.splitext(os.path.basename(filename))[0]
        output_filename: str = f"{base_filename}_analyzed.csv"
        output_filepath: str = os.path.join(os.getcwd(), 'Data', output_filename)

        # Создание и запись в файл результатов анализа
        dataframe = dataframe.sort_values(by=["rate"], ascending=False)
        dataframe.to_csv(output_filepath, index=False, columns=["email", "review text", "date", "rate"])
        logging.info(f"Обработан файл {filename}, результаты сохранены в {output_filepath}.")


def main() -> None:
    # Подключение к OpenAI API
    api_key: Optional[str] = get_api_key()
    if not api_key:
        return

    openai.api_key = api_key
    process_files(api_key)


if __name__ == '__main__':
    main()
