import re
import json
import torch
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, BitsAndBytesConfig
import logging
from nltk.tokenize import sent_tokenize
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from dotenv import load_dotenv
import os

load_dotenv()

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logging.basicConfig(level=logging.INFO)

DOCUMENTS_JSON = "/app/output/data.json"

MODEL_NAME = "tiiuae/falcon-7b-instruct"

DEVICE = torch.device("cuda")


TELEGRAMM_TOKEN= os.getenv('TELEGRAM_TOKEN')

print(TELEGRAMM_TOKEN)

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [2]                                                             # Токены остановки
        return input_ids[0][-1] in stop_ids

def split_into_chunks(text, chunk_size=300):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sent in sentences:
        if current_length + len(sent) > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sent)
        current_length += len(sent)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

class DocumentRetriever:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.retriever = SentenceTransformer(model_name, device='cpu')

    def get_relevant_docs(self, query, documents, top_k=3):
        try:
            query_embedding = self.retriever.encode(query)
            chunks = []

            for doc in documents:
                text = str(doc.get('text', ''))
                text_chunks = split_into_chunks(text)

                for i, chunk in enumerate(text_chunks):  # Без ограничения по чанкам
                    chunks.append({
                        'name': f"{doc.get('name', 'Без названия')} (чанк {i + 1})",
                        'text': chunk,
                    })

            if not chunks:
                return []

            chunk_embeddings = self.retriever.encode([c['text'] for c in chunks])
            similarities = np.dot(chunk_embeddings, query_embedding)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            best_chunks = [chunks[i] for i in top_indices]

            return best_chunks if similarities[top_indices[0]] >= 0.1 else []

        except Exception as e:
            logging.error(f"Ошибка при поиске документов: {e}")
            return []

def init_llm(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16
        ).to(DEVICE)

        return tokenizer, model

    except Exception as e:
        logging.error(f"Ошибка при загрузке модели: {e}")
        raise

def generate_response(query, documents, retriever, tokenizer, model):
    try:
        relevant_docs = retriever.get_relevant_docs(query, documents)
        if not relevant_docs:
            return "Информация отсутствует."

        context_parts = []
        for doc in relevant_docs:
            page_info = ""
            if "(страница" in doc['name']:
                page_info = f" (стр. {doc['name'].split('(страница ')[1].split(')')[0]})"
            elif "чанк" in doc['name']:
                page_info = f" (чанк {doc['name'].split('чанк ')[1].split(')')[0]})"

            cleaned_text = re.sub(r'\[\d+\.\d+\.\d+\]', '', doc['text'])
            source = f"{doc['name'].split(' (')[0]}{page_info}"
            context_parts.append(f"{source}:\n{cleaned_text}")

        context = "\n\n".join(context_parts)

        prompt = f"""
<|im_start|>system
Отвечай только на основе предоставленных документов. Если информации нет, отвечай ровно: "Информация отсутствует." Не добавляй ничего лишнего. Укажи источник (документ и страница/чанк), если данные найдены. <|im_end|>
<|im_start|>user
Контекст: {context}
Вопрос: {query}<|im_end|>
<|im_start|>assistant
"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(DEVICE)

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.9,  # Креативность
            top_p=0.3,  # Строгий или нет выбор токенов
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()])
        )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if '<|im_start|>assistant' in result:
            result = result.split('<|im_start|>assistant')[-1].strip()

        result = re.sub(r'<\|.*?\|>', '', result).strip()


        return result

    except Exception as e:
        logging.error(f"Ошибка при генерации ответа: {e}")
        return "Ошибка обработки запроса."


async def handle_message(update: Update, context: CallbackContext):
    user_query = update.message.text
    response = generate_response(
        query=user_query,
        documents=documents,
        retriever=retriever,
        tokenizer=tokenizer,
        model=model
    )
    await update.message.reply_text(response[:2000])


async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Добро пожаловать! Задавайте вопросы.")

def main():
    try:
        with open(DOCUMENTS_JSON, 'r', encoding='utf-8') as f:
            global documents
            documents = json.load(f)

    except FileNotFoundError:
        logging.error(f"Файл {DOCUMENTS_JSON} не найден.")
        return

    except json.JSONDecodeError:
        logging.error(f"Ошибка в формате файла {DOCUMENTS_JSON}.")
        return

    try:
        global retriever, tokenizer, model
        retriever = DocumentRetriever()
        tokenizer, model = init_llm(MODEL_NAME)

    except Exception as e:
        logging.error(f"Ошибка инициализации: {e}")
        return

    print("задавай тг боту вопросы")
    application = Application.builder().token(TELEGRAMM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()

if __name__ == "__main__":
    main()
    
    
 