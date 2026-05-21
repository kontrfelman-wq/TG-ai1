import os
import logging
import tempfile
import sqlite3
import base64
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import AsyncOpenAI

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
OPENAI_BASE_URL = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
OPENAI_API_KEY = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")

_raw_allowed = os.environ.get("ALLOWED_USER_IDS", "")
ALLOWED_USER_IDS: set[int] = {
    int(uid.strip()) for uid in _raw_allowed.split(",") if uid.strip().isdigit()
}

HISTORY_LIMIT = 200
CONTEXT_LIMIT = 30
DB_PATH = "chat_history.db"

# Моделі у порядку пріоритету — якщо перша не відповідає, пробує наступну
MODELS = ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"]

client = AsyncOpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "Ти корисний AI-асистент. Відповідай чітко, зрозуміло та по суті. "
    "Якщо користувач пише українською — відповідай українською. "
    "Якщо іншою мовою — відповідай тією ж мовою."
)

user_histories: dict[int, list[dict]] = {}


# --- База даних ---

def db_init() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   INTEGER NOT NULL,
                role      TEXT NOT NULL,
                content   TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
    logger.info("База даних ініціалізована")


def db_save(user_id: int, role: str, content: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO messages (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (user_id, role, content, datetime.utcnow().isoformat())
        )
        conn.commit()


def db_load(user_id: int) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, HISTORY_LIMIT)
        ).fetchall()
    rows.reverse()
    result = []
    for role, content in rows:
        norm_role = "assistant" if role in ("model", "assistant") else "user"
        result.append({"role": norm_role, "content": content})
    return result


def db_clear(user_id: int) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        conn.commit()


def db_count(user_id: int) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM messages WHERE user_id = ?", (user_id,)
        ).fetchone()[0]


# --- Допоміжні функції ---

def is_allowed(user_id: int) -> bool:
    if not ALLOWED_USER_IDS:
        return True
    return user_id in ALLOWED_USER_IDS


def ensure_history(user_id: int) -> None:
    if user_id not in user_histories:
        user_histories[user_id] = db_load(user_id)
        logger.info(f"Завантажено {len(user_histories[user_id])} повідомлень для {user_id}")


async def deny(update: Update) -> None:
    await update.message.reply_text("⛔ У тебе немає доступу до цього бота.")


# --- Ядро: запит до AI ---

async def ask_ai(user_id: int, user_message: dict, save_user_text: str) -> str:
    ensure_history(user_id)
    user_histories[user_id].append(user_message)

    recent = user_histories[user_id][-CONTEXT_LIMIT:]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + recent

    last_error = None
    for model in MODELS:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=1024,
            )
            bot_reply = response.choices[0].message.content
            if not bot_reply:
                raise ValueError("Порожня відповідь")

            user_histories[user_id].append({"role": "assistant", "content": bot_reply})
            if len(user_histories[user_id]) > HISTORY_LIMIT:
                user_histories[user_id] = user_histories[user_id][-HISTORY_LIMIT:]

            db_save(user_id, "user", save_user_text)
            db_save(user_id, "assistant", bot_reply)

            logger.info(f"Відповідь від {model}")
            return bot_reply

        except Exception as e:
            last_error = str(e)
            logger.warning(f"Модель {model} не відповіла: {last_error[:80]}")
            continue

    user_histories[user_id].pop()
    raise Exception(f"Всі моделі недоступні: {last_error}")


# --- Обробники команд ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_allowed(user.id):
        await deny(update)
        return
    user_histories[user.id] = db_load(user.id)
    total = db_count(user.id)
    await update.message.reply_text(
        f"Привіт, {user.first_name}! 👋\n\n"
        "Я твій AI-асистент. Запам'ятовую нашу розмову і підтримую безперервний діалог.\n\n"
        f"📂 Збережено повідомлень: {total}\n\n"
        "Команди:\n"
        "/start — відновити розмову\n"
        "/clear — очистити всю історію розмови\n"
        "/stats — статистика розмови\n"
        "/myid — дізнатись свій Telegram ID\n"
        "/help — довідка"
    )


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_allowed(user.id):
        await deny(update)
        return
    db_clear(user.id)
    user_histories[user.id] = []
    await update.message.reply_text("✅ Вся історія розмови видалена. Починаємо з чистого аркуша!")


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_allowed(user.id):
        await deny(update)
        return
    total = db_count(user.id)
    in_memory = len(user_histories.get(user.id, []))
    await update.message.reply_text(
        f"Статистика розмови:\n\n"
        f"Збережено: {total} повідомлень\n"
        f"В пам'яті: {in_memory} повідомлень\n"
        f"Ліміт контексту: {CONTEXT_LIMIT} повідомлень"
    )


async def myid(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_text(f"Твій Telegram ID: {user.id}")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_allowed(user.id):
        await deny(update)
        return
    await update.message.reply_text(
        "Як користуватись ботом:\n\n"
        "Просто напиши мені будь-яке запитання.\n"
        "Я пам'ятаю нашу розмову (до 200 у базі, 30 у контексті).\n\n"
        "Також підтримую:\n"
        "Голосові повідомлення\n"
        "Фотографії (з підписом або без)\n\n"
        "Команди:\n"
        "/start — відновити розмову\n"
        "/clear — очистити всю історію\n"
        "/stats — статистика розмови\n"
        "/myid — дізнатись свій Telegram ID\n"
        "/help — ця довідка"
    )


# --- Обробники повідомлень ---

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_allowed(user.id):
        await deny(update)
        return

    await update.message.chat.send_action("typing")

    try:
        user_text = update.message.text
        bot_reply = await ask_ai(
            user.id,
            user_message={"role": "user", "content": user_text},
            save_user_text=user_text
        )
        await update.message.reply_text(bot_reply)
    except Exception as e:
        logger.error(f"Помилка: {e}")
        await update.message.reply_text("Виникла помилка. Спробуй ще раз.")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_allowed(user.id):
        await deny(update)
        return

    await update.message.chat.send_action("typing")

    try:
        voice_file = await context.bot.get_file(update.message.voice.file_id)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            await voice_file.download_to_drive(tmp.name)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            transcript = await client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=("voice.ogg", f, "audio/ogg"),
            )
        user_text = transcript.text
        logger.info(f"Голос розпізнано: {user_text[:80]}")

        bot_reply = await ask_ai(
            user.id,
            user_message={"role": "user", "content": user_text},
            save_user_text=f"[Голос]: {user_text}"
        )
        await update.message.reply_text(f"Ти сказав: {user_text}\n\n{bot_reply}")

    except Exception as e:
        logger.error(f"Помилка голосу: {e}")
        await update.message.reply_text("Не вдалось обробити голосове повідомлення. Спробуй ще раз.")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_allowed(user.id):
        await deny(update)
        return

    await update.message.chat.send_action("typing")

    try:
        photo = update.message.photo[-1]
        photo_file = await context.bot.get_file(photo.file_id)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            await photo_file.download_to_drive(tmp.name)
            with open(tmp.name, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()

        caption = update.message.caption or "Опиши детально що зображено на цьому фото."

        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": caption},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        }
        bot_reply = await ask_ai(
            user.id,
            user_message=user_message,
            save_user_text=f"[Фото: {caption}]"
        )
        await update.message.reply_text(bot_reply)

    except Exception as e:
        logger.error(f"Помилка фото: {e}")
        await update.message.reply_text("Не вдалось обробити зображення. Спробуй ще раз.")


# --- Запуск ---

def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN не встановлено!")
    if not OPENAI_BASE_URL or not OPENAI_API_KEY:
        raise ValueError("AI_INTEGRATIONS змінні не встановлено!")

    db_init()

    if ALLOWED_USER_IDS:
        logger.info(f"Приватний режим: {len(ALLOWED_USER_IDS)} користувач(ів)")
    else:
        logger.warning("ALLOWED_USER_IDS не задано — бот публічний!")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("myid", myid))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info(f"Бот запущено! Моделі: {', '.join(MODELS)}")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
