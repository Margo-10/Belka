#!/usr/bin/env python3
import os
import subprocess
import logging
from pathlib import Path
import time
import psycopg2
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/restore_db.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def wait_for_postgres():
    """Ожидание готовности PostgreSQL"""
    for i in range(30):
        try:
            conn = psycopg2.connect(
                dbname="postgres",
                user="admin",
                password="secret",
                host="db",
                port="5432"
            )
            conn.close()
            logger.info("PostgreSQL готов")
            return True
        except psycopg2.OperationalError as e:
            logger.info(f"Попытка {i+1}: PostgreSQL ещё не готов...")
            time.sleep(2)
    logger.error("PostgreSQL не готов после 30 попыток")
    return False

def clean_database():
    """Очистка базы данных перед восстановлением"""
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="admin",
            password="secret",
            host="db",
            port="5432"
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            
            cur.execute("""
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = 'documents'
                  AND pid <> pg_backend_pid();
            """)
            
         
            cur.execute("DROP DATABASE IF EXISTS documents WITH (FORCE)")
            cur.execute("CREATE DATABASE documents")
            
            # Подключаемся к новой базе
            conn_db = psycopg2.connect(
                dbname="documents",
                user="admin",
                password="secret",
                host="db",
                port="5432"
            )
            conn_db.autocommit = True
            with conn_db.cursor() as cur_db:
                
                cur_db.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm CASCADE")
            conn_db.close()
            
        logger.info("База данных очищена и расширения созданы")
        return True
    except Exception as e:
        logger.error(f"Ошибка при очистке БД: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def restore_dump(dump_file: str) -> bool:
    """Восстановление одного дампа"""
    cmd = [
        'pg_restore',
        '--verbose',
        '--dbname', 'postgresql://admin:secret@db:5432/documents',
        '--no-owner',
        '--no-privileges',
        '--clean',
        '--if-exists',
        '--exit-on-error',
        '--jobs', '1',
        dump_file
    ]
    logger.info(f"Восстановление {dump_file}...")
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            logger.error(f"Ошибка при восстановлении {dump_file}: {result.stderr}")
            return False
        logger.info(f"Успешно восстановлен {dump_file}")
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"Таймаут при восстановлении {dump_file}")
        return False

def main():
    """Основная функция восстановления"""
    Path('/app/logs').mkdir(parents=True, exist_ok=True)
    
    if not wait_for_postgres():
        sys.exit(1)

    if not clean_database():
        sys.exit(1)

   
    dumps = [
        '/app/WorkWithDb/filestorage.dump',
        '/app/WorkWithDb/cms.dump',
        '/app/WorkWithDb/lists.dump'
    ]

    for dump in dumps:
        if not Path(dump).exists():
            logger.error(f"Файл дампа не найден: {dump}")
            continue
            
        if not restore_dump(dump):
            sys.exit(1)

    logger.info("Все дампы успешно восстановлены")
    
    # Проверка восстановленных данных
    try:
        conn = psycopg2.connect(
            dbname="documents",
            user="admin",
            password="secret",
            host="db",
            port="5432"
        )
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM storage_storageobject")
            count = cur.fetchone()[0]
            logger.info(f"Восстановлено {count} записей в storage_storageobject")
    except Exception as e:
        logger.error(f"Ошибка при проверке БД: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

    sys.exit(0)

if __name__ == "__main__":
    main()
