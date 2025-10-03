import logging
import os
import pypdf
import docx
import ollama
import json
import re
import asyncio
import pandas as pd
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
)
from telegram.constants import ParseMode

# --- НАСТРОЙКИ ---
# Безопасный способ хранения токена:
# 1. Создайте рядом с ботом файл .env
# 2. Напишите в нем: TELEGRAM_BOT_TOKEN="ВАШ_ТОКЕН"
# 3. Установите библиотеку: pip install python-dotenv
load_dotenv()
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Не найден токен! Создайте .env файл или установите переменную окружения TELEGRAM_BOT_TOKEN.")

OLLAMA_MODEL = 'llama3:8b'

# --- ЛОГИРОВАНИЕ ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ЗАГРУЗКА ДАТАСЕТА ПРИ СТАРТЕ БОТА ---
try:
    JOBS_DF = pd.read_csv('cleaned_jobs.csv')
    # Убедимся, что колонка для поиска текстовая и без пустых значений
    JOBS_DF['SearchText'] = JOBS_DF['SearchText'].astype(str).fillna('')
    logger.info(f"Датасет cleaned_jobs.csv успешно загружен. Найдено {len(JOBS_DF)} вакансий.")
except FileNotFoundError:
    logger.error("ОШИБКА: Файл 'cleaned_jobs.csv' не найден. Сначала запустите скрипт для очистки данных.")
    JOBS_DF = pd.DataFrame()

# --- СОСТОЯНИЯ ДИАЛОГА ---
WAITING_JD, WAITING_CV = range(2)

# --- ФУНКЦИЯ ДЛЯ ОЧИСТКИ MARKDOWN ---
def escape_markdown(text: str) -> str:
    """Экранирует специальные символы Markdown V2."""
    if not isinstance(text, str):
        return ''
    # В Markdown V2 нужно экранировать все эти символы
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

# --- УЛУЧШЕННАЯ ФУНКЦИЯ ПОИСКА ПО ЛОКАЛЬНОМУ ДАТАСЕТУ ---
def search_vacancies_dataset(query: str) -> list[dict]:
    """Ищет вакансии в загруженном DataFrame по ключевым словам."""
    if JOBS_DF.empty:
        logger.warning("Поиск невозможен, так как датафрейм пуст.")
        return []

    logger.info(f"Начинаю локальный поиск по датасету с запросом: '{query}'")

    # Разбиваем запрос на отдельные слова
    keywords = query.lower().split()
    if not keywords:
        return []

    # Создаем regex, который ищет все слова в любом порядке
    # (?=.*word1)(?=.*word2) и т.д.
    regex_query = ''.join([f'(?=.*{re.escape(word)})' for word in keywords])

    try:
        results_df = JOBS_DF[JOBS_DF['SearchText'].str.contains(regex_query, case=False, na=False, regex=True)]
    except re.error as e:
        logger.error(f"Ошибка в регулярном выражении при поиске: {e}")
        return []

    top_5_results = results_df.head(5)
    return top_5_results.to_dict('records')

# --- ИСПРАВЛЕННАЯ ФУНКЦИЯ ФОРМАТИРОВАНИЯ СООБЩЕНИЯ С ВАКАНСИЯМИ ---
def format_vacancies_message(vacancies: list[dict]) -> str:
    """Форматирует сообщение с найденными вакансиями, избегая ошибок Markdown."""
    if not vacancies:
        return escape_markdown("Анализ завершен!\n\nК сожалению, в локальной базе данных не нашлось подходящих вакансий.")

    header = escape_markdown("Анализ завершен!\n\nА вот 5 подходящих вакансий из локальной базы:")
    message_parts = [header, ""]  # Добавляем пустую строку для отступа

    for i, v in enumerate(vacancies, 1):
        # Экранируем каждую часть отдельно
        title = escape_markdown(v.get('title', 'Без названия'))
        company = escape_markdown(v.get('company', 'Не указана'))
        location = escape_markdown(v.get('location', 'Не указан'))

        # Собираем сообщение. `\\.` экранирует точку после номера. Звездочки для форматирования оставляем.
        part = f"{i}\\. *{title}*\n   Компания: {company}\n   Город: {location}"
        message_parts.append(part)

    footer = escape_markdown("\nЧтобы начать новый анализ, отправьте команду /new.")
    message_parts.append(footer)
    return "\n".join(message_parts)


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ И ДИАЛОГ ---
async def extract_text_from_file(file_path: str):
    """Извлекает текст из .pdf и .docx файлов и удаляет файл."""
    text = ""
    try:
        if file_path.lower().endswith('.pdf'):
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages: text += page.extract_text() or ""
        elif file_path.lower().endswith('.docx'):
            doc = docx.Document(file_path)
            for para in doc.paragraphs: text += para.text + "\n"
        return text
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

async def get_document_or_text(update: Update):
    """Получает текст из сообщения или из присланного документа."""
    if update.message.text:
        return update.message.text
    if update.message.document:
        doc = update.message.document
        if not doc.file_name.lower().endswith(('.pdf', '.docx')):
            await update.message.reply_text("Пожалуйста, отправьте файл в формате .pdf или .docx")
            return None
        file = await doc.get_file()
        file_path = f"temp_{doc.file_id}_{doc.file_name}"
        await file.download_to_drive(file_path)
        return await extract_text_from_file(file_path)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Начинает или перезапускает диалог."""
    context.user_data.clear()
    user = update.effective_user
    await update.message.reply_html(
        f"Привет, {user.mention_html()}!\n\nЯ готов к анализу. Отправь мне описание вакансии (JD).\n\n"
        "Чтобы начать сначала, отправь команду /new."
    )
    return WAITING_JD

async def received_jd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает полученное описание вакансии."""
    await update.message.reply_text("Обрабатываю вакансию...")
    jd_text = await get_document_or_text(update)
    if not jd_text:
        await update.message.reply_text("Не удалось прочитать текст. Попробуйте еще раз.")
        return WAITING_JD
    context.user_data['job_description'] = jd_text
    await update.message.reply_text("Вакансия принята. Теперь отправь мне твое резюме (CV).")
    return WAITING_CV

def get_analysis_from_llm(jd: str, cv: str) -> str:
    """Синхронная функция для обращения к Ollama. Запускается в отдельном потоке."""
    logger.info("Отправляю запрос в Ollama...")
    try:
        prompt = f'Проведи анализ резюме (CV) на соответствие вакансии (JD). Структурируй ответ.\n1. **Общая оценка соответствия (в %):**\n2. **Сильные стороны:**\n3. **Зоны для улучшения:**\n\n--- JD ---\n{jd}\n--- CV ---\n{cv}\n\nПосле анализа, ОБЯЗАТЕЛЬНО добавь JSON-объект с поисковым запросом.\nПРАВИЛА: запрос должен быть коротким (2-4 слова), содержать должность и ключевую технологию.\nПример: Python Backend Developer, Data Scientist Pandas, QA Automation Engineer.\nНЕ ИСПОЛЬЗУЙ описательные слова.\nJSON должен выглядеть ТОЧНО так:\nSEARCH_QUERY_JSON: {{"query": "Python Developer"}}'
        response = ollama.chat(model=OLLAMA_MODEL, messages=[{'role': 'user', 'content': prompt}])
        logger.info("Получен ответ от Ollama.")
        return response['message']['content']
    except Exception as e:
        logger.error(f"Ошибка при обращении к Ollama: {e}")
        return "Ошибка: Не удалось связаться с языковой моделью для анализа."

async def received_cv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обрабатывает резюме, запускает анализ и поиск."""
    await update.message.reply_text("Обрабатываю резюме...")
    cv_text = await get_document_or_text(update)
    jd_text = context.user_data.get('job_description')

    if not cv_text or not jd_text:
        await update.message.reply_text("Ошибка, описание вакансии или резюме потеряно. Начнем сначала? /new")
        return ConversationHandler.END

    await update.message.reply_text("Анализирую документы... Это может занять до минуты.")
    try:
        # Асинхронно запускаем тяжелую (синхронную) функцию, чтобы не блокировать бота
        loop = asyncio.get_running_loop()
        llm_response = await loop.run_in_executor(
            None, get_analysis_from_llm, jd_text, cv_text
        )

        analysis_text, search_query = llm_response, None

        if "SEARCH_QUERY_JSON:" in llm_response:
            parts = llm_response.split("SEARCH_QUERY_JSON:", 1)
            analysis_text = parts[0].strip()
            json_part = parts[1]
            match = re.search(r'{\s*".*?"\s*:\s*".*?"\s*}', json_part, re.DOTALL)
            if match:
                try:
                    search_data = json.loads(match.group(0))
                    search_query = search_data.get("query")
                except json.JSONDecodeError:
                    logger.error(f"LLM вернула некорректный JSON: {match.group(0)}")
                    analysis_text += "\n\n(Не удалось разобрать JSON для поиска)"

        safe_analysis_text = escape_markdown(analysis_text)
        await update.message.reply_text(safe_analysis_text, parse_mode=ParseMode.MARKDOWN_V2)

        if search_query:
            search_query_safe = escape_markdown(search_query)
            await update.message.reply_text(f"Ищу вакансии по ключам: *{search_query_safe}*", parse_mode=ParseMode.MARKDOWN_V2)
            vacancies = search_vacancies_dataset(search_query)
            vacancies_message = format_vacancies_message(vacancies)
            await update.message.reply_text(vacancies_message, parse_mode=ParseMode.MARKDOWN_V2, disable_web_page_preview=True)
        else:
            await update.message.reply_text(escape_markdown("Анализ завершен! Не удалось автоматически сгенерировать запрос для поиска. Чтобы начать новый анализ, отправьте /new."))

    except Exception as e:
        logger.error(f"Критическая ошибка в received_cv: {e}", exc_info=True)
        await update.message.reply_text(escape_markdown("Произошла критическая ошибка при анализе. Попробуйте снова: /new"))
    
    context.user_data.clear()
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отменяет текущий диалог."""
    await update.message.reply_text("Действие отменено. Чтобы начать новый анализ, введите /new.")
    context.user_data.clear()
    return ConversationHandler.END

def main() -> None:
    """Основная функция для запуска бота."""
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start), CommandHandler('new', start)],
        states={
            WAITING_JD: [MessageHandler(filters.TEXT & ~filters.COMMAND | filters.Document.ALL, received_jd)],
            WAITING_CV: [MessageHandler(filters.TEXT & ~filters.COMMAND | filters.Document.ALL, received_cv)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )
    application.add_handler(conv_handler)
    
    print("Бот запущен!")
    application.run_polling()

if __name__ == '__main__':
    main()