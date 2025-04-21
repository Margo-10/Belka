
# Hackathon

## Цель проекта:
Разработать прототип (MVP) поисковой системы на базе LLM  для интеграции 
его в VK HR Tek.  

## Структура проекта

```text
Hakaton/
├── WorkWithDb/
│   ├── RestoreDb.py       # Восстановление БД
│   ├── FileDownloader.py  # Скачивание файлов 
│   ├── lists.dump         # Исходные данные
│   ├── filestorage.dump   # Исходные данные
│   └── cms.dump           # Исходные данные
├── data/                  
│   └── downloaded_files   # Скачанные файлы
├── logs/                  # Логи
├── output/ 
│   └── data.json          # Обработанные данные
├── .dockerignore          # Файл для исключения файлов из Docker
├── .env                   # Файл с переменными окружения
├── .gitignore             # Файл для исключения файлов из Git
├── DS2plusTG              # Модель + тг-бот
├── Data2.py               # Обработка данных
├── Dockerfile             # Конфигурация Docker образа
├── README.md              # Описание проекта
├── docker-compose.yml     # Конфигурация Docker
└── requirements.txt       # Файл со списком необходимых библиотек и их версий
```
## Развертывание проекта

### Требуется скачать:

- [Docker](https://www.docker.com/)
- [Docker Compose](https://wiki.crowncloud.net/?How_to_Install_and_use_Docker_Compose_on_Ubuntu_24_04)
  


### Этапы запуска

1.  **Клонирование репозитория**

2.  **Добавление в переменную .env токена вашего телеграм бота**
   
3. **Выполнение команд**
 ```bash
    docker-compose build
    docker-compose up

 ```


