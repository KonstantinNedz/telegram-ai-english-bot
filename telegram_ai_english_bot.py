#!/usr/bin/env python3
# telegram_ai_english_bot.py
# Telegram AI English Tutor Bot (single-file)
# Features:
# - Учим слова (AI generate + manual add; choose CEFR + category)
# - Повторяем слова (SM-2 spaced repetition; show en->ru, ru->en, shuffle, filters)
# - TTS (gTTS) pronunciation button (optional)
# - Categories: business, travel, daily life, IT (and custom)
# - Statistics: total words, reviews today, avg quality, learned words (repetitions >= 5)

import os
import logging
import sqlite3
import json
import re
import random
import datetime
import tempfile

from typing import Optional

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputFile,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
import openai

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

openai.api_key = OPENAI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1"]
CATEGORIES = ["business", "travel", "daily life", "IT"]

# ---------------- Database ----------------

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
        cefr_level TEXT,
        category TEXT,
        efactor REAL DEFAULT 2.5,
        interval INTEGER DEFAULT 0,
        repetitions INTEGER DEFAULT 0,
        next_review TEXT,
        added_at TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS review_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        word_id INTEGER,
        quality INTEGER,
        ts TEXT
    )
    """)
    conn.commit()
    conn.close()

def add_review_word(user_id: int, word: str, transcription: str = "", translation: str = "",
                    examples: str = "", cefr: str = "A2", category: str = "daily life"):
    now = datetime.datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO review_words (user_id, word, transcription, translation, examples, cefr_level, category, next_review, added_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, word.strip(), transcription.strip(), translation.strip(), examples.strip(), cefr, category, now, now))
    conn.commit()
    conn.close()

def delete_word_by_id(word_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM review_words WHERE id=?", (word_id,))
    conn.commit()
    conn.close()

def delete_word_by_text(user_id: int, word_text: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM review_words WHERE user_id=? AND word=?", (user_id, word_text))
    conn.commit()
    conn.close()

def get_all_words(user_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, word, transcription, translation, examples, cefr_level, category, efactor, interval, repetitions, next_review FROM review_words WHERE user_id=? ORDER BY added_at", (user_id,))
    rows = cur.fetchall()
    conn.close()
    return rows

def get_due_words(user_id: int, limit: int = 50):
    now = datetime.datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, word, transcription, translation, examples, cefr_level, category, efactor, interval, repetitions, next_review FROM review_words WHERE user_id=? AND (next_review<=? OR next_review IS NULL) ORDER BY next_review LIMIT ?",
        (user_id, now, limit)
    )
    rows = cur.fetchall()
    conn.close()
    return rows

def update_word_review(word_id: int, quality: int):
    """
    SM-2 algorithm update for a single word (quality: 0..5).
    Logs the review in review_logs.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT user_id, efactor, interval, repetitions FROM review_words WHERE id=?", (word_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return
    user_id, efactor, interval, repetitions = row
    efactor = float(efactor or 2.5)
    interval = int(interval or 0)
    repetitions = int(repetitions or 0)

    if quality < 3:
        repetitions = 0
        interval = 1
    else:
        repetitions += 1
        if repetitions == 1:
            interval = 1
        elif repetitions == 2:
            interval = 6
        else:
            interval = int(round(interval * efactor)) if interval > 0 else 1

    efactor = efactor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    if efactor < 1.3:
        efactor = 1.3

    next_review = (datetime.datetime.utcnow() + datetime.timedelta(days=interval)).isoformat()
    cur.execute("UPDATE review_words SET efactor=?, interval=?, repetitions=?, next_review=? WHERE id=?",
                (efactor, interval, repetitions, next_review, word_id))
    # log the review
    cur.execute("INSERT INTO review_logs (user_id, word_id, quality, ts) VALUES (?, ?, ?, ?)",
                (user_id, word_id, quality, datetime.datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

# ---------------- OpenAI helpers ----------------

def generate_word_via_ai() -> dict:
    """
    Ask OpenAI to output a single word in strict JSON form.
    Returns dict with keys: word, transcription, translation, examples(list).
    Fallback to a safe stub if parsing fails.
    """
    prompt = (
        "Generate one useful English vocabulary word for a language learner and return STRICT JSON only in the following format:\n"
        "{\n  \"word\": \"...\",\n  \"transcription\": \"...\",\n  \"translation\": \"...\",\n  \"examples\": [\"...\", \"...\", \"...\"]\n}\n\nReturn no additional text."
    )
    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=250
        )
        text = resp.choices[0].message.content.strip()
        # try load JSON directly or extract JSON substring
        try:
            data = json.loads(text)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                data = json.loads(m.group(0))
            else:
                raise
        # Normalize fields
        examples = data.get("examples", [])
        if not isinstance(examples, list):
            examples = [str(examples)]
        while len(examples) < 3:
            examples.append("")
        return {
            "word": data.get("word", "").strip(),
            "transcription": data.get("transcription", "").strip(),
            "translation": data.get("translation", "").strip(),
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

def synthesize_tts(text: str, lang: str = "en") -> Optional[str]:
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

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📚 Слова", callback_data="words")],
        [InlineKeyboardButton("✍️ Грамматика", callback_data="grammar")],
        [InlineKeyboardButton("🗣 Разговор", callback_data="chat")],
        [InlineKeyboardButton("📊 Статистика", callback_data="stats")],
    ]
    await update.message.reply_text("Выберите раздел", reply_markup=InlineKeyboardMarkup(keyboard))

# Words menu
async def words_menu_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    keyboard = [
        [InlineKeyboardButton("📗 Учим слова", callback_data="learn_words")],
        [InlineKeyboardButton("🔁 Повторяем слова", callback_data="review_words")],
        [InlineKeyboardButton("🧾 Мои слова (список)", callback_data="list_my_words")],
        [InlineKeyboardButton("⬅ Назад", callback_data="back")],
    ]
    await query.message.reply_text("Раздел слов", reply_markup=InlineKeyboardMarkup(keyboard))

# Learn flow
async def learn_words_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = generate_word_via_ai()
    # push to session history
    hist = context.user_data.get("generated_history", [])
    hist.append(data)
    context.user_data["generated_history"] = hist
    context.user_data["generated_index"] = len(hist) - 1

    text = (
        f"Word: {data['word']}\nTranscription: {data.get('transcription','')}\nTranslation: {data.get('translation','')}\n\n"
        f"Examples:\n- {data['examples'][0]}\n- {data['examples'][1]}\n- {data['examples'][2]}"
    )
    keyboard = [
        [InlineKeyboardButton("✅ Учить (в список)", callback_data="choose_cefr")],
        [InlineKeyboardButton("⏭ Пропустить (следующее)", callback_data="next_generated")],
        [InlineKeyboardButton("⬅ Предыдущее", callback_data="prev_generated")],
    ]
    keyboard.append([InlineKeyboardButton("➕ Добавить своё слово", callback_data="manual_add")])
    keyboard.append([InlineKeyboardButton("⬅ Назад", callback_data="words")])
    await query.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

async def choose_cefr_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    buttons = [[InlineKeyboardButton(level, callback_data=f"cefr_{level}")] for level in CEFR_LEVELS]
    buttons.append([InlineKeyboardButton("Отмена", callback_data="cancel_save")])
    await query.message.reply_text("Выберите CEFR уровень:", reply_markup=InlineKeyboardMarkup(buttons))

async def cefr_selected_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    cefr = query.data.split("_", 1)[1]
    context.user_data["pending_cefr"] = cefr
    buttons = [[InlineKeyboardButton(cat, callback_data=f"cat_{cat}")] for cat in CATEGORIES]
    buttons.append([InlineKeyboardButton("Другой", callback_data="cat_other")])
    buttons.append([InlineKeyboardButton("Отмена", callback_data="cancel_save")])
    await query.message.reply_text("Выберите категорию:", reply_markup=InlineKeyboardMarkup(buttons))

async def category_selected_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    if data == "cat_other":
        context.user_data["awaiting_custom_category"] = True
        await query.message.reply_text("Отправьте название категории (например: sports)")
        return
    cat = data.split("_", 1)[1]
    cefr = context.user_data.pop("pending_cefr", "A2")
    idx = context.user_data.get("generated_index")
    hist = context.user_data.get("generated_history", [])
    if idx is None or idx >= len(hist):
        await query.message.reply_text("Нет текущего слова для сохранения")
        return
    w = hist[idx]
    add_review_word(query.from_user.id, w["word"], w.get("transcription", ""), w.get("translation", ""), "\n".join(w.get("examples", [])), cefr=cefr, category=cat)
    await query.message.reply_text(f"Слово '{w['word']}' добавлено (CEFR={cefr}, category={cat})")

async def save_with_custom_category_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get("awaiting_custom_category"):
        return
    cat = update.message.text.strip()
    context.user_data.pop("awaiting_custom_category", None)
    cefr = context.user_data.pop("pending_cefr", "A2")
    idx = context.user_data.get("generated_index")
    hist = context.user_data.get("generated_history", [])
    if idx is None or idx >= len(hist):
        await update.message.reply_text("Нет текущего слова для сохранения")
        return
    w = hist[idx]
    add_review_word(update.effective_user.id, w["word"], w.get("transcription", ""), w.get("translation", ""), "\n".join(w.get("examples", [])), cefr=cefr, category=cat)
    await update.message.reply_text(f"Слово '{w['word']}' добавлено (CEFR={cefr}, category={cat})")

async def next_generated_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = generate_word_via_ai()
    hist = context.user_data.get("generated_history", [])
    hist.append(data)
    context.user_data["generated_history"] = hist
    context.user_data["generated_index"] = len(hist) - 1
    # reuse learn_words_cb to present
    await learn_words_cb(update, context)

async def prev_generated_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    idx = context.user_data.get("generated_index", 0)
    hist = context.user_data.get("generated_history", [])
    if idx > 0:
        idx -= 1
        context.user_data["generated_index"] = idx
        w = hist[idx]
        text = (
            f"Word: {w['word']}\nTranscription: {w.get('transcription','')}\nTranslation: {w.get('translation','')}\n\n"
            f"Examples:\n- {w['examples'][0]}\n- {w['examples'][1]}\n- {w['examples'][2]}"
        )
        keyboard = [
            [InlineKeyboardButton("✅ Учить (в список)", callback_data="choose_cefr")],
            [InlineKeyboardButton("⏭ Пропустить (следующее)", callback_data="next_generated")],
            [InlineKeyboardButton("⬅ Предыдущее", callback_data="prev_generated")],
        ]
        keyboard.append([InlineKeyboardButton("➕ Добавить своё слово", callback_data="manual_add")])
        keyboard.append([InlineKeyboardButton("⬅ Назад", callback_data="words")])
        await query.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        await query.message.reply_text("Это первое сгенерированное слово в этой сессии")

# Manual add flow
async def start_manual_add_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    context.user_data["awaiting_manual_add"] = True
    await query.message.reply_text(
        "Отправьте слово в формате:\nслово — транскрипция — перевод — пример1;пример2;пример3\nили коротко: слово — перевод\n"
        "Можно добавить теги через запятую: ,CEFR=B1,cat=business"
    )

async def handle_manual_add_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get("awaiting_manual_add"):
        return
    text = update.message.text.strip()
    user_id = update.effective_user.id
    cefr = "A2"
    cat = "daily life"
    tags = re.findall(r',\s*([^,=\s]+)\s*=\s*([^,]+)', text)
    if tags:
        for k, v in tags:
            text = re.sub(r',\s*' + re.escape(k) + r'\s*=\s*' + re.escape(v), '', text)
            if k.lower() == "cefr" and v.strip().upper() in CEFR_LEVELS:
                cefr = v.strip().upper()
            if k.lower() in ("cat", "category"):
                cat = v.strip()
    parts = re.split(r'[-—–]', text, maxsplit=3)
    if len(parts) == 1:
        parts = [p.strip() for p in text.split(',', 3)]
    word = parts[0].strip() if len(parts) >= 1 else ""
    transcription = parts[1].strip() if len(parts) >= 2 else ""
    translation = parts[2].strip() if len(parts) >= 3 else ""
    examples_raw = parts[3].strip() if len(parts) >= 4 else ""
    # If user sent "word — translation"
    if not translation and len(parts) >= 2 and not transcription:
        translation = parts[1].strip()
        transcription = ""
    examples = ""
    if examples_raw:
        ex_list = [e.strip() for e in re.split(r'[;\n]', examples_raw) if e.strip()]
        examples = "\n".join(ex_list)
    add_review_word(user_id, word, transcription, translation, examples, cefr=cefr, category=cat)
    context.user_data.pop("awaiting_manual_add", None)
    await update.message.reply_text(f"Сохранено: {word} — {translation} (CEFR={cefr}, cat={cat})")

# Review menu & flow
async def review_menu_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    keyboard = [
        [InlineKeyboardButton("🇬🇧 Англ → Рус", callback_data="review_en")],
        [InlineKeyboardButton("🇷🇺 Рус → Англ", callback_data="review_ru")],
        [InlineKeyboardButton("🔀 Вперемешку", callback_data="review_mix")],
        [InlineKeyboardButton("Фильтр CEFR/категория", callback_data="filter_options")],
    ]
    await query.message.reply_text("Как показывать слова?", reply_markup=InlineKeyboardMarkup(keyboard))

async def filter_options_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    buttons = [[InlineKeyboardButton(level, callback_data=f"filter_cefr_{level}")] for level in CEFR_LEVELS]
    cat_buttons = [[InlineKeyboardButton(cat, callback_data=f"filter_cat_{cat}")] for cat in CATEGORIES]
    keyboard = buttons + cat_buttons
    keyboard.append([InlineKeyboardButton("Все", callback_data="filter_none")])
    await query.message.reply_text("Выберите фильтр:", reply_markup=InlineKeyboardMarkup(keyboard))

async def review_words_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    mode = query.data
    # Filtering
    if mode.startswith("filter_cefr_"):
        cefr = mode.split("_", 2)[2]
        rows = [r for r in get_due_words(query.from_user.id) if r[5] == cefr]
    elif mode.startswith("filter_cat_"):
        cat = mode.split("_", 2)[2]
        rows = [r for r in get_due_words(query.from_user.id) if r[6] == cat]
    else:
        rows = get_due_words(query.from_user.id)
        if mode == "review_mix":
            random.shuffle(rows)
    if not rows:
        await query.message.reply_text("Нет слов для повторения прямо сейчас (due). Добавьте слова или подождите до следующего повторения.")
        return
    context.user_data["review_words"] = rows
    context.user_data["review_index"] = 0
    context.user_data["review_mode"] = mode
    await send_review_item_cb(query, context)

async def send_review_item_cb(query_or_update, context: ContextTypes.DEFAULT_TYPE):
    # can be called with (query, context) from callback handlers
    if isinstance(query_or_update, Update):
        query = query_or_update.callback_query
    else:
        query = query_or_update
    # no need to answer here, callers do it
    i = context.user_data.get("review_index", 0)
    rows = context.user_data.get("review_words", [])
    if i >= len(rows):
        await query.message.reply_text("Повторение закончено")
        return
    row = rows[i]
    word_id, word, transcription, translation, examples, cefr, category, efactor, interval, repetitions, next_review = row
    mode = context.user_data.get("review_mode", "review_en")
    if mode in ("review_en", "review_mix", "review_en_order"):
        text = f"{word}\n{transcription}\nCEFR: {cefr} | category: {category}"
    else:
        text = f"{translation}\nCEFR: {cefr} | category: {category}"
    keyboard = [
        [InlineKeyboardButton("Показать ответ", callback_data=f"show_answer_{word_id}")],
        [InlineKeyboardButton("🔊 Произношение", callback_data=f"tts_{word_id}")],
        [InlineKeyboardButton("Удалить слово", callback_data=f"delete_word_{word_id}")],
        [InlineKeyboardButton("Следующее", callback_data="next_word")],
    ]
    await query.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

async def show_answer_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    m = re.match(r"show_answer_(\d+)", query.data)
    if not m:
        await query.message.reply_text("Не получается показать ответ")
        return
    word_id = int(m.group(1))
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT word, transcription, translation, examples FROM review_words WHERE id=?", (word_id,))
    r = cur.fetchone()
    conn.close()
    if not r:
        await query.message.reply_text("Слово не найдено")
        return
    word, transcription, translation, examples = r
    text = f"{word}\n{transcription}\n{translation}\n\nExamples:\n{examples if examples else '(нет примеров)'}"
    # rating buttons 0..5
    buttons = [[InlineKeyboardButton(str(s), callback_data=f"rate_{word_id}_{s}") for s in range(0, 6)]]
    await query.message.reply_text(text, reply_markup=InlineKeyboardMarkup(buttons))

async def rating_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    m = re.match(r"rate_(\d+)_(\d+)", query.data)
    if not m:
        await query.message.reply_text("Ошибка оценки")
        return
    word_id = int(m.group(1))
    quality = int(m.group(2))
    update_word_review(word_id, quality)
    await query.message.reply_text(f"Оценка {quality} сохранена.")
    # advance
    idx = context.user_data.get("review_index", 0) + 1
    context.user_data["review_index"] = idx
    rows = context.user_data.get("review_words", [])
    if idx < len(rows):
        await send_review_item_cb(update.callback_query, context)
    else:
        await query.message.reply_text("Повторение окончено")

async def tts_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    m = re.match(r"tts_(\d+)", query.data)
    if not m:
        await query.message.reply_text("TTS: неверный запрос")
        return
    word_id = int(m.group(1))
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT word FROM review_words WHERE id=?", (word_id,))
    r = cur.fetchone()
    conn.close()
    if not r:
        await query.message.reply_text("Слово не найдено")
        return
    word = r[0]
    if not TTS_AVAILABLE:
        await query.message.reply_text("TTS не доступен в окружении. Установите библиотеку gTTS.")
        return
    tmp_path = synthesize_tts(word)
    if not tmp_path:
        await query.message.reply_text("Ошибка генерации аудио")
        return
    try:
        await query.message.reply_audio(audio=InputFile(tmp_path), caption=f"Pronunciation: {word}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

async def delete_word_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    m = re.match(r"delete_word_(\d+)", query.data)
    if not m:
        await query.message.reply_text("Неверный запрос на удаление")
        return
    word_id = int(m.group(1))
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT word FROM review_words WHERE id=?", (word_id,))
    r = cur.fetchone()
    if not r:
        await query.message.reply_text("Слово не найдено")
        conn.close()
        return
    word = r[0]
    cur.execute("DELETE FROM review_words WHERE id=?", (word_id,))
    conn.commit()
    conn.close()
    await query.message.reply_text(f"Удалено: {word}")
    # remove from session list if present
    rows = context.user_data.get("review_words", [])
    for i, row in enumerate(rows):
        if row[0] == word_id:
            rows.pop(i)
            context.user_data["review_words"] = rows
            break

async def next_word_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    idx = context.user_data.get("review_index", 0) + 1
    context.user_data["review_index"] = idx
    rows = context.user_data.get("review_words", [])
    if idx >= len(rows):
        await query.message.reply_text("Это было последнее слово")
        return
    await send_review_item_cb(update.callback_query, context)

async def list_my_words_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    rows = get_all_words(query.from_user.id)
    if not rows:
        await query.message.reply_text("Ваш список пуст")
        return
    lines = []
    for r in rows[:200]:
        wid, w, tr, trans, ex, cefr, cat, ef, inter, rep, nr = r
        next_date = nr.split("T")[0] if nr else "—"
        lines.append(f"{w} — {tr} | {cefr} | {cat} | rep={rep} | next={next_date}")
    # send in chunks if too long
    chunk = "\n".join(lines)
    await query.message.reply_text("Ваши слова:\n" + chunk)

# Statistics
def get_stats(user_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM review_words WHERE user_id=?", (user_id,))
    total = cur.fetchone()[0]
    today = datetime.datetime.utcnow().date().isoformat()
    cur.execute("SELECT COUNT(*), AVG(quality) FROM review_logs WHERE user_id=? AND substr(ts,1,10)=?", (user_id, today))
    r = cur.fetchone() or (0, None)
    reviews_today = r[0]
    avg_quality_today = float(r[1]) if r[1] is not None else None
    cur.execute("SELECT COUNT(*) FROM review_words WHERE user_id=? AND repetitions>=5", (user_id,))
    learned = cur.fetchone()[0]
    conn.close()
    return {
        "total_words": total,
        "reviews_today": reviews_today,
        "avg_quality_today": avg_quality_today,
        "learned_words": learned,
    }

async def stats_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if query:
        await query.answer()
        user_id = query.from_user.id
    else:
        user_id = update.effective_user.id
    s = get_stats(user_id)
    text = (
        f"Статистика:\nВсего слов: {s['total_words']}\nПовторений сегодня: {s['reviews_today']}\n"
        f"Средняя оценка сегодня: {s['avg_quality_today'] if s['avg_quality_today'] is not None else '—'}\n"
        f"Изучено (repetitions>=5): {s['learned_words']}"
    )
    if query:
        await query.message.reply_text(text)
    else:
        await update.message.reply_text(text)

# Message router (manual add + custom category)
async def message_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get("awaiting_manual_add"):
        await handle_manual_add_message(update, context)
        return
    if context.user_data.get("awaiting_custom_category"):
        await save_with_custom_category_message(update, context)
        return
    # otherwise ignore or implement fallback handlers
    return

# ---------------- Main ----------------

def main():
    init_db()
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start))

    # Menu
    app.add_handler(CallbackQueryHandler(words_menu_cb, pattern="^words$"))
    app.add_handler(CallbackQueryHandler(learn_words_cb, pattern="^learn_words$"))
    app.add_handler(CallbackQueryHandler(choose_cefr_cb, pattern="^choose_cefr$"))
    app.add_handler(CallbackQueryHandler(cefr_selected_cb, pattern="^cefr_"))
    app.add_handler(CallbackQueryHandler(category_selected_cb, pattern="^cat_"))
    app.add_handler(CallbackQueryHandler(next_generated_cb, pattern="^next_generated$"))
    app.add_handler(CallbackQueryHandler(prev_generated_cb, pattern="^prev_generated$"))
    app.add_handler(CallbackQueryHandler(start_manual_add_cb, pattern="^manual_add$"))

    # Review
    app.add_handler(CallbackQueryHandler(review_menu_cb, pattern="^review_words$"))
    app.add_handler(CallbackQueryHandler(filter_options_cb, pattern="^filter_options$"))
    app.add_handler(CallbackQueryHandler(review_words_cb, pattern="^review_en$|^review_ru$|^review_mix$|^filter_cefr_|^filter_cat_|^filter_none$"))
    app.add_handler(CallbackQueryHandler(show_answer_cb, pattern="^show_answer_"))
    app.add_handler(CallbackQueryHandler(rating_cb, pattern="^rate_"))
    app.add_handler(CallbackQueryHandler(tts_cb, pattern="^tts_"))
    app.add_handler(CallbackQueryHandler(delete_word_cb, pattern="^delete_word_"))
    app.add_handler(CallbackQueryHandler(next_word_cb, pattern="^next_word$"))
    app.add_handler(CallbackQueryHandler(list_my_words_cb, pattern="^list_my_words$"))

    # Stats
    app.add_handler(CallbackQueryHandler(stats_cb, pattern="^stats$"))

    # Message handler(s)
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), message_router))

    logger.info("Bot started")
    app.run_polling()

if __name__ == "__main__":
    main()
