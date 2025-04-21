#!/usr/bin/env python3
import os
import sys
import re  
from pathlib import Path
import concurrent.futures
from typing import List, Dict, Optional
import requests
import psycopg2
import logging
import time
import hashlib

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/file_downloader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FileDownloader:
    def __init__(self):
        self.base_url = os.getenv('BASE_DOWNLOAD_URL', 'https://hackaton.hb.ru-msk.vkcloud-storage.ru/media')
        self.download_dir = Path('/app/data/downloaded_files')
        self.max_workers = 5
        self.db_retries = 3
        self.db_wait = 2
        self.download_retries = 3
        self.download_timeout = int(os.getenv('DOWNLOAD_TIMEOUT', '30'))
        self.max_filename_length = 100
        
       
        self.download_dir.mkdir(parents=True, exist_ok=True)
        Path('/app/logs').mkdir(parents=True, exist_ok=True)

    def sanitize_filename(self, filename: str) -> str:
        """Очищает имя файла и обрезает его до допустимой длины"""
        # Удаляем недопустимые символы
        filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
        
        # Если имя слишком длинное, используем хеш
        if len(filename.encode('utf-8')) > self.max_filename_length:
            ext = Path(filename).suffix
            name_hash = hashlib.md5(filename.encode()).hexdigest()
            return f"{name_hash}{ext}"
        return filename

    def get_db_connection(self) -> Optional[psycopg2.extensions.connection]:
        """Устанавливает соединение с БД с повторными попытками"""
        for attempt in range(self.db_retries):
            try:
                conn = psycopg2.connect(
                    dbname=os.getenv('POSTGRES_DB', 'documents'),
                    user=os.getenv('POSTGRES_USER', 'admin'),
                    password=os.getenv('POSTGRES_PASSWORD', 'secret'),
                    host=os.getenv('PG_HOST', 'db'),
                    port=os.getenv('POSTGRES_PORT', '5432')
                )
                logger.info("Успешное подключение к базе данных")
                return conn
            except psycopg2.OperationalError as e:
                if attempt == self.db_retries - 1:
                    logger.error(f"Не удалось подключиться к БД после {self.db_retries} попыток: {e}")
                    return None
                logger.warning(f"Попытка {attempt + 1}: Ошибка подключения к БД, повтор через {self.db_wait} сек...")
                time.sleep(self.db_wait)

    def get_file_list(self, conn) -> List[Dict[str, str]]:
        """Получает список файлов из базы данных"""
        query = """
            SELECT sv.link, so.name
            FROM storage_storageobject AS so
            LEFT JOIN storage_version AS sv ON sv.storage_object_id = so.id
            WHERE so.type = 1
              AND sv.link IS NOT NULL
              AND sv.link <> ''
        """
        try:
            with conn.cursor() as cur:
                cur.execute(query)
                return [{'name': row[1], 'url': row[0]} for row in cur.fetchall()]
        except psycopg2.Error as e:
            logger.error(f"Ошибка при запросе к БД: {e}")
            return []

    def prepare_download_dir(self):
        """Подготавливает директорию для загрузки"""
        try:
            self.download_dir.mkdir(parents=True, exist_ok=True)
            (self.download_dir / '.keep').touch()
        except Exception as e:
            logger.error(f"Ошибка при создании директории: {e}")
            sys.exit(1)

    def download_file(self, file_info: Dict[str, str]) -> Dict[str, str]:
        """Загружает один файл с повторными попытками"""
        original_name = file_info['name']
        file_name = self.sanitize_filename(original_name)
        file_url = file_info['url']
        file_path = self.download_dir / file_name
        
        result = {
            'filename': file_name,
            'original_name': original_name,
            'status': 'failed',
            'message': '',
            'attempts': 0
        }

        # Проверка существующего файла
        if file_path.exists() and file_path.stat().st_size > 0:
            result.update({
                'status': 'skipped',
                'message': 'File already exists'
            })
            return result

        full_url = f"{self.base_url.rstrip('/')}/{file_url.lstrip('/')}"
        temp_path = file_path.with_suffix('.tmp')

        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': '*/*'
        }

        for attempt in range(self.download_retries):
            result['attempts'] += 1
            try:
                response = requests.get(
                    full_url,
                    headers=headers,
                    stream=True,
                    timeout=self.download_timeout
                )
                response.raise_for_status()

                # Скачивание во временный файл
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                # Проверка скачанного файла
                if temp_path.exists() and temp_path.stat().st_size > 0:
                    temp_path.rename(file_path)
                    result.update({
                        'status': 'success',
                        'message': f"Downloaded {file_name}"
                    })
                    return result
                else:
                    raise Exception("Downloaded file is empty")

            except requests.exceptions.RequestException as e:
                result['message'] = f"HTTP error: {str(e)}"
                if attempt == self.download_retries - 1:
                    break
                time.sleep(1)
            except Exception as e:
                result['message'] = f"Unexpected error: {str(e)}"
                if attempt == self.download_retries - 1:
                    break
                time.sleep(1)
            finally:
                if temp_path.exists():
                    temp_path.unlink()

        return result

    def run(self):
        """Основной метод выполнения"""
        self.prepare_download_dir()
        
        conn = self.get_db_connection()
        if not conn:
            sys.exit(1)

        try:
            files = self.get_file_list(conn)
            if not files:
                logger.error("Не найдено файлов для загрузки. Проверьте:")
                logger.error("1. Что БД восстановлена корректно")
                logger.error("2. Что в БД есть файлы")
                sys.exit(1)

            logger.info(f"Найдено {len(files)} файлов для загрузки")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.download_file, file): file['name']
                    for file in files
                }

                for future in concurrent.futures.as_completed(futures):
                    file_name = futures[future]
                    try:
                        result = future.result()
                        if result['status'] == 'success':
                            logger.info(f"Успешно: {result['original_name']} -> {result['filename']} (попыток: {result['attempts']})")
                        elif result['status'] == 'skipped':
                            logger.debug(f"Пропущен: {result['filename']}")
                        else:
                            logger.error(f"Ошибка: {result['original_name']} - {result['message']} (попыток: {result['attempts']})")
                    except Exception as e:
                        logger.error(f"Ошибка при загрузке {file_name}: {str(e)}")

        finally:
            conn.close()
            logger.info("Завершение работы")

if __name__ == "__main__":
    downloader = FileDownloader()
    downloader.run()
    
