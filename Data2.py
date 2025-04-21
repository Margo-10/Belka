#!/usr/bin/env python3
import os
import re
import json
import warnings
import traceback
import logging
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    import docx
    import pytesseract
    import pandas as pd
    from PIL import Image
    from PyPDF2 import PdfReader, PdfWriter
    from pdfminer.high_level import extract_text as pdfminer_extract
    from pdf2image import convert_from_path
except ImportError as e:
    logger.error(f"Ошибка импорта зависимостей: {e}")
    raise

# Конфигурация
DATA_DIR = os.getenv('DATA_DIR', '/app/data/downloaded_files')
OUTPUT_JSON = os.getenv('OUTPUT_JSON', '/app/output/data.json')
POPPLER_PATH = os.environ.get("POPPLER_PATH", "/usr/bin")
OCR_TIMEOUT = int(os.getenv('OCR_TIMEOUT', '600'))  # 10 минут на обработку одного файла
MAX_PAGES_FOR_OCR = 500  # Максимальное количество страниц для OCR
OCR_CHUNK_SIZE = 10  # Количество страниц для одновременной обработки OCR

pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD', '/usr/bin/tesseract')

def get_page_count(file_path: str) -> int:
    """Возвращает количество страниц в PDF"""
    try:
        with open(file_path, 'rb') as f:
            return len(PdfReader(f).pages)
    except Exception as e:
        logger.warning(f"Ошибка при подсчете страниц: {e}")
        return 0

def fix_cropbox_inplace(file_path: str):
    """Исправляет CropBox прямо в файле без создания копии"""
    try:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            writer = PdfWriter()

            for page in reader.pages:
                if '/CropBox' not in page:
                    page.cropbox = page.mediabox
                writer.add_page(page)

            # Перезаписываем исходный файл
            with open(file_path, 'wb') as f_out:
                writer.write(f_out)
    except Exception as e:
        logger.warning(f"Ошибка при исправлении CropBox: {e}")

def pdf_to_text(file_path: str) -> str:
    """Извлекает текст из PDF с обработкой в памяти"""
    try:
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_pdf:
            # Копируем оригинальный файл во временный
            with open(file_path, 'rb') as original:
                temp_pdf.write(original.read())
            temp_pdf.flush()

            # Исправляем CropBox во временном файле
            fix_cropbox_inplace(temp_pdf.name)

            # Извлекаем текст
            text = pdfminer_extract(temp_pdf.name)
            
            # Если текст слишком короткий, пробуем OCR
            if not text.strip() or len(text.strip()) < 100:
                logger.info(f"Текст пустой/короткий ({len(text)} символов), запускаем OCR")
                ocr_text = pdf_to_text_ocr(temp_pdf.name)
                if ocr_text:
                    return ocr_text
            
            return text
            
    except Exception as e:
        logger.warning(f"Ошибка при обработке PDF: {e}")
        # Пробуем OCR как запасной вариант
        try:
            logger.info("Пробуем OCR из-за ошибки в основном методе")
            return pdf_to_text_ocr(file_path)
        except Exception as ocr_error:
            logger.warning(f"Ошибка при OCR PDF: {ocr_error}")
            return ""

def pdf_to_text_ocr(file_path: str) -> str:
    """Извлекает текст из PDF с помощью OCR с обработкой по частям"""
    try:
        total_pages = get_page_count(file_path)
        if total_pages == 0:
            return ""
            
        if total_pages > MAX_PAGES_FOR_OCR:
            logger.warning(f"Файл слишком большой ({total_pages} страниц), обрабатываются первые {MAX_PAGES_FOR_OCR} страниц")
            total_pages = MAX_PAGES_FOR_OCR

        text_parts = []
        
        # Обрабатываем по частям для экономии памяти
        for start_page in range(0, total_pages, OCR_CHUNK_SIZE):
            end_page = min(start_page + OCR_CHUNK_SIZE, total_pages)
            logger.info(f"Обработка страниц {start_page + 1}-{end_page} из {total_pages}")

            try:
                images = convert_from_path(
                    file_path,
                    first_page=start_page + 1,
                    last_page=end_page,
                    dpi=300,
                    poppler_path=POPPLER_PATH,
                    grayscale=True,
                    thread_count=2,
                    timeout=OCR_TIMEOUT
                )

                for i, img in enumerate(images):
                    page_num = start_page + i + 1
                    try:
                        page_text = pytesseract.image_to_string(img, lang='rus+eng')
                        text_parts.append(f"=== Страница {page_num} ===\n{page_text}\n")
                    except Exception as ocr_error:
                        logger.warning(f"OCR ошибка на странице {page_num}: {ocr_error}")
                        
            except Exception as chunk_error:
                logger.warning(f"Ошибка при обработке страниц {start_page + 1}-{end_page}: {chunk_error}")

        return "\n".join(text_parts)
        
    except Exception as e:
        logger.warning(f"Ошибка при OCR PDF: {e}")
        return ""

def doc_to_text(file_path: str) -> str:
    """Конвертирует DOC в текст через DOCX"""
    try:
        logger.info(f"Конвертация DOC в DOCX: {file_path}")
        
        # Создаем временную директорию для конвертации
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = subprocess.run(
                ["libreoffice", "--headless", "--convert-to", "docx", "--outdir", tmp_dir, file_path],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                logger.warning(f"Ошибка конвертации DOC: {result.stderr}")
                return ""

            # Ищем сконвертированный файл
            docx_file = next(Path(tmp_dir).glob("*.docx"), None)
            if not docx_file:
                logger.warning("Не удалось найти сконвертированный DOCX файл")
                return ""
                
            return docx_to_text(str(docx_file))
            
    except Exception as e:
        logger.warning(f"Ошибка при чтении DOC: {e}")
        return ""

def docx_to_text(file_path: str) -> str:
    """Извлекает текст из DOCX"""
    try:
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        logger.warning(f"Ошибка при чтении DOCX: {e}")
        return ""

def excel_to_text(file_path: str) -> str:
    """Извлекает текст из Excel"""
    try:
        text = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Пробуем разные движки для чтения Excel
            try:
                dfs = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
            except:
                dfs = pd.read_excel(file_path, sheet_name=None, engine='xlrd')

            for sheet_name, df in dfs.items():
                df = df.fillna('')
                text.append(f"=== Лист '{sheet_name}' ===\n{df.to_string(index=False)}\n")
                
        return "\n\n".join(text)
    except Exception as e:
        logger.warning(f"Ошибка при чтении Excel: {e}")
        return ""

def clean_text(text: str) -> str:
    """Очищает текст от лишних пробелов и спецсимволов"""
    if not text:
        return ""
        
    # Удаляем управляющие символы
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)
    # Заменяем множественные пробелы и переносы
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_file(file_path: str) -> str:
    """Выбирает подходящий метод извлечения текста по расширению файла"""
    ext = Path(file_path).suffix.lower()
    try:
        if ext == '.pdf':
            return pdf_to_text(file_path)
        elif ext == '.docx':
            return docx_to_text(file_path)
        elif ext == '.doc':
            return doc_to_text(file_path)
        elif ext in ('.xlsx', '.xls'):
            return excel_to_text(file_path)
        else:
            logger.info(f"Неизвестный формат файла: {file_path}")
            return ""
    except Exception as e:
        logger.error(f"Критическая ошибка при обработке {file_path}: {e}")
        return ""

def process_single_file(file_path: str, existing_docs_map: Dict) -> Dict:
    """Обрабатывает один файл и возвращает результат"""
    filename = Path(file_path).name
    logger.info(f"Обработка файла: {filename}")
    
    # Пропускаем временные и обработанные файлы
    if '.fixed.' in filename or filename in existing_docs_map:
        logger.info(f"Пропускаем файл: {filename}")
        return existing_docs_map.get(filename, {
            'name': filename,
            'text': '',
            'preview': 'Пропущен (дубликат или временный файл)',
            'status': 'skipped'
        })

    try:
        start_time = time.time()
        text = extract_text_from_file(file_path)
        processing_time = time.time() - start_time
        
        cleaned = clean_text(text)
        preview = cleaned[:200] + ('...' if len(cleaned) > 200 else '')
        
        doc_dict = {
            'name': filename,
            'text': cleaned,
            'preview': preview,
            'size': os.path.getsize(file_path),
            'processing_time': round(processing_time, 2),
            'status': 'success'
        }
        
        logger.info(f"Успешно обработан: {filename} ({doc_dict['processing_time']} сек)")
        return doc_dict
        
    except Exception as e:
        logger.error(f"Ошибка обработки файла {filename}: {e}")
        traceback.print_exc()
        return {
            'name': filename,
            'text': '',
            'preview': f'Ошибка: {str(e)}',
            'status': 'failed'
        }

def main():
    """Основная функция обработки файлов"""
    # Проверяем и создаем директории
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(OUTPUT_JSON)).mkdir(parents=True, exist_ok=True)
    
    # Получаем список файлов для обработки
    file_paths = []
    for ext in ('.pdf', '.docx', '.doc', '.xlsx', '.xls'):
        file_paths.extend(Path(DATA_DIR).glob(f'*{ext}'))
    
    if not file_paths:
        logger.error("В указанной папке нет файлов поддерживаемых форматов.")
        return

    logger.info(f"\nНайдено {len(file_paths)} файлов для обработки в {DATA_DIR}")

    # Загружаем предыдущие результаты
    documents = []
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                documents = json.load(f)
        except Exception as e:
            logger.error(f"Ошибка при чтении {OUTPUT_JSON}: {e}")

    existing_docs_map = {doc['name']: doc for doc in documents}
    processed_files = 0
    skipped_files = 0

    # Обрабатываем файлы с использованием ThreadPool
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_single_file, str(fp), existing_docs_map): fp
            for fp in file_paths
        }

        for future in as_completed(futures):
            file_path = futures[future]
            try:
                result = future.result()
                existing_docs_map[result['name']] = result
                
                if result['status'] == 'skipped':
                    skipped_files += 1
                else:
                    processed_files += 1
                
                # Периодически сохраняем промежуточные результаты
                if processed_files % 10 == 0:
                    save_results(existing_docs_map)
                    
            except Exception as e:
                logger.error(f"Ошибка при обработке {file_path}: {e}")

    # Сохраняем финальные результаты
    save_results(existing_docs_map)

    logger.info("\n" + "="*50)
    logger.info(f"Обработка завершена. Всего файлов: {len(file_paths)}")
    logger.info(f"Успешно обработано: {processed_files}")
    logger.info(f"Пропущено: {skipped_files}")
    logger.info(f"Не удалось обработать: {len(file_paths) - processed_files - skipped_files}")
    logger.info(f"Результаты сохранены в {OUTPUT_JSON}")

def save_results(docs_map: Dict):
    """Сохраняет результаты в JSON файл"""
    documents = list(docs_map.values())
    try:
        temp_file = f"{OUTPUT_JSON}.tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        
        # Атомарная замена файла
        os.replace(temp_file, OUTPUT_JSON)
        logger.info(f"Сохранено {len(documents)} записей в {OUTPUT_JSON}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении результатов: {e}")

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    logger.info(f"Общее время выполнения: {time.time() - start_time:.2f} секунд")
