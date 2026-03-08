#!/usr/bin/env python3
# telegram_ai_english_bot.py
# Миграция на OpenAI v1 client + TTS + упрощённый функционал: Учим / Повторяем / Список

import os
import logging
import sqlite3
import json
import re
import random
import datetime
import tempfile
import html

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
# OpenAI v1 client
from openai import OpenAI

# Optional TTS
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# Config / env
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
DB_PATH = os.getenv("BOT_DB_PATH", "english_bot.db")

if not TELEGRAM_TOKEN or not OPENAI_API_KEY:
    raise RuntimeError("Please set TELEGRAM_TOKEN and OPENAI_API_KEY environment variables")

# create new OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.INFO)
logging.getLogger("telegram.ext").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Database (simple) ----------------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS review_words (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        word TEXT,
        transcription TEXT,
        translation TEXT,
        examples TEXT,
        added_at TEXT
    )
    """)
    conn.commit()
    conn.close()

def add_review_word(user_id: int, word: str, transcription: str = "", translation: str = "", examples: str = ""):
    now = datetime.datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO review_words (user_id, word, transcription, translation, examples, added_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, word.strip(), transcription.strip(), translation.strip(), examples.strip(), now))
    conn.commit()
    conn.close()

def delete_word_by_id(word_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM review_words WHERE id=?", (word_id,))
    conn.commit()
    conn.close()

def get_all_words(user_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, word, transcription, translation, examples, added_at FROM review_words WHERE user_id=? ORDER BY added_at", (user_id,))
    rows = cur.fetchall()
    conn.close()
    return rows

# ---------------- OpenAI helpers (new SDK) ----------------

def generate_word_via_ai() -> dict:
    """
    Use OpenAI v1 client (client.chat.completions.create) to get a JSON object describing one word.
    Returns dict with keys: word, transcription, translation, examples (list of 3).
    """
    prompt = (
        "Generate one useful English vocabulary word for a language learner and return STRICT JSON only in the following format:\n"
        "{\n  \"word\": \"...\",\n  \"transcription\": \"...\",\n  \"translation\": \"...\",\n  \"examples\": [\"...\", \"...\", \"...\"]\n}\n\nReturn no additional text."
    )
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=250,
        )

        # try different ways to extract text depending on SDK minor variations
        text = None
        try:
            text = resp.choices[0].message.content
        except Exception:
            try:
                text = resp.choices[0].message["content"]
            except Exception:
                try:
                    text = resp["choices"][0]["message"]["content"]
                except Exception:
                    text = str(resp)

        # parse JSON from text (safe)
        try:
            data = json.loads(text)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                data = json.loads(m.group(0))
            else:
                raise

        examples = data.get("examples", [])
        if not isinstance(examples, list):
            examples = [str(examples)]
        while len(examples) < 3:
            examples.append("")
        return {
            "word": str(data.get("word", "")).strip(),
            "transcription": str(data.get("transcription", "")).strip(),
            "translation": str(data.get("translation", "")).strip(),
            "examples": examples[:3],
        }

    except Exception:
        logger.exception("AI generation failed; returning fallback word")
        return {
            "word": "example",
            "transcription": "[ˈɛɡzæmpəl]",
            "translation": "пример",
            "examples": ["This is an example sentence.", "For example, ...", "Another example."],
        }

# ---------------- TTS ----------------

def synthesize_tts(text: str, lang: str = "en") -> str | None:
    if not TTS_AVAILABLE:
        return None
    try:
        tts = gTTS(text=text, lang=lang)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp_name = tmp.name
        tmp.close()
        tts.save(tmp_name)
        return tmp_name
    except Exception:
        logger.exception("TTS generation failed")
        return None

# ---------------- Telegram handlers ----------------

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📗 Учим новые слова", callback_data="learn_words")],
        [InlineKeyboardButton("🔁 Повторяем слова", callback_data="review_words")],
        [InlineKeyboardButton("🧾 Список моих слов", callback_data="list_my_words")],
    ]
    await update.message.reply_text("Выберите действие:", reply_markup=InlineKeyboardMarkup(keyboard))

async def words_menu_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    keyboard = [
        [InlineKeyboardButton("📗 Учим новые слова", callback_data="learn_words")],
        [InlineKeyboardButton("🔁 Повторяем слова", callback_data="review_words")],
        [InlineKeyboardButton("🧾 Список моих слов", callback_data="list_my_words")],
    ]
    await query.message.reply_text("Выберите действие:", reply_markup=InlineKeyboardMarkup(keyboard))

# Learn: generate and present word (with bold HTML + tts)
async def learn_words_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = generate_word_via_ai()
    hist = context.user_data.get("generated_history", [])
    hist.append(data)
    context.user_data["generated_history"] = hist
    context.user_data["generated_index"] = len(hist) - 1
    await send_generated_word_cb(query, context, data)

async def send_generated_word_cb(query, context, data):
    word_safe = html.escape(data.get('word','') or "")
    transcription = html.escape(data.get('transcription','') or "")
    translation = html.escape(data.get('translation','') or "")
    examples = data.get('examples', [])
    text = f"<b>{word_safe}</b>\n\n"
    if transcription:
        text += f"<i>{transcription}</i>\n\n"
    if translation:
        text += f"Translation: {translation}\n\n"
    text += "Examples:\n"
    for ex in examples:
        text += f"- {html.escape(ex)}\n"

    keyboard = [
        [InlineKeyboardButton("✅ Учить (сохранить)", callback_data="save_word"),
         InlineKeyboardButton("🔊 Произношение", callback_data="tts_gen")],
        [InlineKeyboardButton("⏭ Пропустить (следующее)", callback_data="next_generated")],
        [InlineKeyboardButton("⬅ Предыдущее", callback_data="prev_generated")],
    ]
    keyboard.append([InlineKeyboardButton("➕ Добавить своё слово", callback_data="manual_add")])
    keyboard.append([InlineKeyboardButton("⬅ Главное меню", callback_data="menu")])

    await query.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")

async def save_word_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    idx = context.user_data.get("generated_index")
    hist = context.user_data.get("generated_history", [])
    if idx is None or idx >= len(hist):
        await query.message.reply_text("Нет текущего слова для сохранения")
        return
    w = hist[idx]
    add_review_word(query.from_user.id, w["word"], w.get("transcription",""), w.get("translation",""), "\n".join(w.get("examples",[])))
    await query.message.reply_text(f"Слово '{w['word']}' сохранено.")

async def next_generated_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = generate_word_via_ai()
    hist = context.user_data.get("generated_history", [])
    hist.append(data)
    context.user_data["generated_history"] = hist
    context.user_data["generated_index"] = len(hist) - 1
    await send_generated_word_cb(query, context, data)

async def prev_generated_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    idx = context.user_data.get("generated_index", 0)
    hist = context.user_data.get("generated_history", [])
    if idx > 0:
        idx -= 1
        context.user_data["generated_index"] = idx
        data = hist[idx]
        await send_generated_word_cb(query, context, data)
    else:
        await query.message.reply_text("Это первое сгенерированное слово в этой сессии")

# Manual add
async def start_manual_add_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    context.user_data["awaiting_manual_add"] = True
    await query.message.reply_text(
        "Отправьте слово в формате:\nслово — транскрипция — перевод — пример1;пример2;пример3\nили коротко: слово — перевод"
    )

async def handle_manual_add_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get("awaiting_manual_add"):
        return
    text = update.message.text.strip()
    user_id = update.effective_user.id
    parts = re.split(r'[-—–]', text, maxsplit=3)
    if len(parts) == 1:
        parts = [p.strip() for p in text.split(',', 3)]
    word = parts[0].strip() if len(parts) >= 1 else ""
    transcription = parts[1].strip() if len(parts) >= 2 else ""
    translation = parts[2].strip() if len(parts) >= 3 else ""
    examples_raw = parts[3].strip() if len(parts) >= 4 else ""
    if not translation and len(parts) >= 2 and not transcription:
        translation = parts[1].strip()
        transcription = ""
        examples_raw = ""
    examples = ""
    if examples_raw:
        ex_list = [e.strip() for e in re.split(r'[;\n]', examples_raw) if e.strip()]
        examples = "\n".join(ex_list)
    add_review_word(user_id, word, transcription, translation, examples)
    context.user_data.pop("awaiting_manual_add", None)
    await update.message.reply_text(f"Сохранено: {word} — {translation}")

# Review flow
async def review_words_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    rows = get_all_words(query.from_user.id)
    if not rows:
        await query.message.reply_text("Ваш список повторения пуст. Добавьте слова через 'Учим новые слова' или '➕ Добавить своё слово'.")
        return
    context.user_data["review_words"] = rows
    context.user_data["review_index"] = 0
    await send_review_item_cb(query, context)

async def send_review_item_cb(query, context):
    i = context.user_data.get("review_index", 0)
    rows = context.user_data.get("review_words", [])
    if i >= len(rows):
        await query.message.reply_text("Повторение закончено.")
        return
    wid, word, transcription, translation, examples, added_at = rows[i]
    text = f"{word}\n{transcription}" if word else translation
    keyboard = [
        [InlineKeyboardButton("Показать ответ", callback_data=f"show_answer_{wid}")],
        [InlineKeyboardButton("🔊 Произношение", callback_data=f"tts_{wid}")],
        [InlineKeyboardButton("Удалить слово", callback_data=f"delete_word_{wid}")],
        [InlineKeyboardButton("Следующее", callback_data="next_review_word")],
        [InlineKeyboardButton("⬅ Главное меню", callback_data="menu")],
    ]
    await query.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

async def show_answer_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    m = re.match(r"show_answer_(\d+)", query.data)
    if not m:
        await query.message.reply_text("Не удалось показать ответ.")
        return
    word_id = int(m.group(1))
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT word, transcription, translation, examples FROM review_words WHERE id=?", (word_id,))
    r = cur.fetchone()
    conn.close()
    if not r:
        await query.message.reply_text("Слово не найдено.")
        return
    word, transcription, translation, examples = r
    text = f"{word}\n{transcription}\n{translation}\n\nExamples:\n{examples if examples else '(нет примеров)'}"
    await query.message.reply_text(text)

async def delete_word_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    m = re.match(r"delete_word_(\d+)", query.data)
    if not m:
        await query.message.reply_text("Неверный запрос на удаление.")
        return
    word_id = int(m.group(1))
    rows = context.user_data.get("review_words", [])
    idx = None
    for i, row in enumerate(rows):
        if row[0] == word_id:
            idx = i
            break
    delete_word_by_id(word_id)
    if idx is not None:
        rows.pop(idx)
        context.user_data["review_words"] = rows
        if context.user_data.get("review_index", 0) >= len(rows) and len(rows) > 0:
            context.user_data["review_index"] = len(rows) - 1
    await query.message.reply_text("Слово удалено.")
    if rows:
        await send_review_item_cb(query, context)
    else:
        await query.message.reply_text("Список повторения теперь пуст.")

async def next_review_word_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    idx = context.user_data.get("review_index", 0) + 1
    context.user_data["review_index"] = idx
    rows = context.user_data.get("review_words", [])
    if idx >= len(rows):
        await query.message.reply_text("Это было последнее слово.")
        return
    await send_review_item_cb(query, context)

# TTS handlers
async def tts_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    m = re.match(r"tts_(\d+)", query.data)
    if not m:
        await query.message.reply_text("TTS: неверный запрос.")
        return
    word_id = int(m.group(1))
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT word FROM review_words WHERE id=?", (word_id,))
    r = cur.fetchone()
    conn.close()
    if not r:
        await query.message.reply_text("Слово не найдено.")
        return
    word = r[0]
    if not TTS_AVAILABLE:
        await query.message.reply_text("TTS не доступен. Установите gTTS в requirements.")
        return
    tmp = synthesize_tts(word)
    if not tmp:
        await query.message.reply_text("Ошибка генерации аудио.")
        return
    try:
        await query.message.reply_audio(audio=InputFile(tmp), caption=f"Pronunciation: {word}")
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass

async def tts_generated_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    idx = context.user_data.get("generated_index")
    hist = context.user_data.get("generated_history", [])
    if idx is None or idx >= len(hist):
        await query.message.reply_text("Нет текущего сгенерированного слова для озвучивания.")
        return
    w = hist[idx]
    word = w.get("word") or ""
    if not TTS_AVAILABLE:
        await query.message.reply_text("TTS недоступен в этом окружении. Установите gTTS в requirements.")
        return
    tmp = synthesize_tts(word)
    if not tmp:
        await query.message.reply_text("Ошибка при генерации аудио.")
        return
    try:
        await query.message.reply_audio(audio=InputFile(tmp), caption=f"Pronunciation: {word}")
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass

# List my words
async def list_my_words_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    rows = get_all_words(query.from_user.id)
    if not rows:
        await query.message.reply_text("Ваш список пуст.")
        return
    lines = []
    for r in rows:
        wid, w, tr, trans, ex, added = r
        lines.append(f"{w} — {tr or trans}")
    msg = "Ваши слова:\n" + "\n".join(lines[:200])
    await query.message.reply_text(msg)

# Generic "menu" handler
async def menu_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    keyboard = [
        [InlineKeyboardButton("📗 Учим новые слова", callback_data="learn_words")],
        [InlineKeyboardButton("🔁 Повторяем слова", callback_data="review_words")],
        [InlineKeyboardButton("🧾 Список моих слов", callback_data="list_my_words")],
    ]
    await query.message.reply_text("Выберите действие:", reply_markup=InlineKeyboardMarkup(keyboard))

# Message router (manual add)
async def message_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get("awaiting_manual_add"):
        await handle_manual_add_message(update, context)
        return
    return

# ---------------- Main ----------------

def main():
    init_db()
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # commands
    app.add_handler(CommandHandler("start", start_cmd))

    # flows
    app.add_handler(CallbackQueryHandler(words_menu_cb, pattern="^menu$|^words$"))
    app.add_handler(CallbackQueryHandler(learn_words_cb, pattern="^learn_words$"))
    app.add_handler(CallbackQueryHandler(save_word_cb, pattern="^save_word$"))
    app.add_handler(CallbackQueryHandler(next_generated_cb, pattern="^next_generated$"))
    app.add_handler(CallbackQueryHandler(prev_generated_cb, pattern="^prev_generated$"))
    app.add_handler(CallbackQueryHandler(start_manual_add_cb, pattern="^manual_add$"))

    app.add_handler(CallbackQueryHandler(review_words_cb, pattern="^review_words$"))
    app.add_handler(CallbackQueryHandler(show_answer_cb, pattern="^show_answer_"))
    app.add_handler(CallbackQueryHandler(delete_word_cb, pattern="^delete_word_"))
    app.add_handler(CallbackQueryHandler(next_review_word_cb, pattern="^next_review_word$"))
    app.add_handler(CallbackQueryHandler(list_my_words_cb, pattern="^list_my_words$"))

    app.add_handler(CallbackQueryHandler(tts_cb, pattern="^tts_"))
    app.add_handler(CallbackQueryHandler(tts_generated_cb, pattern="^tts_gen$"))

    app.add_handler(CallbackQueryHandler(menu_cb, pattern="^menu$"))

    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), message_router))

    logger.info("Bot started")
    app.run_polling()

if __name__ == "__main__":
    main()
