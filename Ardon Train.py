#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  MF ARDON v4.0 — СКРИПТ ОБУЧЕНИЯ ПУСТОЙ МОДЕЛИ НА СВОИХ .TXT ФАЙЛАХ
  (ardon_train.py — Основной файл обучения с прогресс-баром и безопасным стопом)
================================================================================
Для кого этот скрипт:
  - Вы кладёте любые свои .txt файлы (конспекты, лекции, книги, статьи, заметки)
    в выбранную папку (по умолчанию `my_data`).
  - Пустая модель с нуля читает ваши тексты, строит смысловую карту понятий,
    обращается к серверу MF Ardon (http://162.35.31.11:8001) по вашему ключу
    и сохраняет выученный мозг модели в файлы `ardon_checkpoint.npz` и `ardon_memory.json`.
  - В любой момент вы можете написать в консоль команду:
        stop   (или стоп / save / q) + нажать Enter
    или нажать Ctrl+C — обучение МГНОВЕННО остановится без ошибок и бережно
    сохранит весь накопленный результат на диск!
================================================================================
"""

from __future__ import annotations
import argparse
import glob
import json
import math
import os
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ==============================================================================
# >>> 1. ГЛАВНЫЕ НАСТРОЙКИ (ВСТАВЬТЕ ВАШ КЛЮЧ И НАЗВАНИЕ ПАПКИ) <<<
# ==============================================================================

SERVER_URL   = "http://162.35.31.11:8001"        # Адрес сервера MF Ardon (порт 8001)
API_KEY      = "ВСТАВЬТЕ_ВАШ_КЛЮЧ_СЮДА"           # <-- Впишите сюда ваш ключ (например: "ardon-client-...")

DATA_FOLDER  = "my_data"                         # <-- Название папки, откуда будут читаться все .txt файлы
EPOCHS       = 12                                # Количество эпох обучения (можно остановить раньше в любой момент)
D_MODEL      = 48                                # Размерность векторов памяти модели (32..128 для быстрого CPU)
BATCH_SIZE   = 16                                # Сколько связей отправлять за 1 шаг

# Файлы, куда сохраняется обученная модель (их потом читает ardon_chat.py)
CHECKPOINT_WEIGHTS_FILE = "ardon_checkpoint.npz"
CHECKPOINT_MEMORY_FILE  = "ardon_memory.json"
CHECKPOINT_STATE_FILE   = "ardon_state.bin"


# ==============================================================================
# 2. КОНТРОЛЬ ОЗУ, КОДИРОВКИ WINDOWS И БЕЗОПАСНОЙ ОСТАНОВКИ (STOP / CTRL+C)
# ==============================================================================

def setup_console() -> None:
    try:
        if sys.platform.startswith("win"):
            os.system("chcp 65001 > nul 2>&1")
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def get_ram_usage_mb() -> float:
    """Возвращает текущее потребление оперативной памяти процессом (в МБ) без сторонних библиотек."""
    try:
        if os.path.exists("/proc/self/status"):
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = float(line.split()[1])
                        return round(kb / 1024.0, 1)
        import resource
        r_max = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return round(r_max / (1024.0 * 1024.0), 1)
        return round(r_max / 1024.0, 1)
    except Exception:
        return 35.0


class SafeStopController:
    """
    Отслеживает команду остановки от пользователя:
      1) Можно просто напечатать `stop`, `стоп`, `save` или `q` в терминале и нажать Enter.
      2) Или нажать `Ctrl+C`.
    В обоих случаях модель НЕ падает, а завершает текущий шаг и сохраняет прогресс.
    """

    def __init__(self):
        self.stop_requested = False
        self.stop_reason = ""
        self._lock = threading.Lock()
        signal.signal(signal.SIGINT, self._on_sigint)

    def _on_sigint(self, signum, frame):
        with self._lock:
            self.stop_requested = True
            self.stop_reason = "Нажато Ctrl+C (экстренное сохранение)"

    def start_command_listener(self) -> None:
        if not sys.stdin:
            return

        def _listen():
            while not self.stop_requested:
                try:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    cmd = line.strip().lower()
                    if cmd in ("stop", "стоп", "save", "сохранить", "q", "quit", "exit", "выход"):
                        with self._lock:
                            self.stop_requested = True
                            self.stop_reason = f"Введена команда '{cmd}'"
                        break
                except Exception:
                    break

        t = threading.Thread(target=_listen, daemon=True)
        t.start()


# ==============================================================================
# 3. ЧТЕНИЕ .TXT ФАЙЛОВ ИЗ ПАПКИ И ПОДГОТОВКА СЛОВАРЯ / ПРЕДЛОЖЕНИЙ
# ==============================================================================

STOPWORDS_RU_EN = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все",
    "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по",
    "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из", "ему",
    "теперь", "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть",
    "был", "него", "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там", "потом",
    "себя", "ничего", "ей", "может", "они", "тут", "где", "есть", "надо", "ней", "для",
    "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз",
    "тоже", "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти", "мой", "тем",
    "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех", "никогда", "можно",
    "при", "наконец", "два", "об", "другой", "хоть", "после", "над", "больше", "тот",
    "через", "эти", "нас", "про", "всего", "них", "какая", "много", "разве", "три",
    "эту", "моя", "впрочем", "хорошо", "свою", "этой", "перед", "иногда", "лучше",
    "чуть", "том", "нельзя", "такой", "им", "более", "всегда", "конечно", "всю",
    "между", "это", "как", "также", "причем", "который", "которая", "которые", "которого",
    "the", "and", "of", "to", "in", "is", "that", "for", "it", "as", "was", "with",
    "be", "by", "on", "not", "he", "this", "are", "or", "his", "from", "at", "which",
    "but", "have", "an", "had", "they", "you", "were", "their", "one", "all", "we",
}


SAMPLE_TXT_1 = """Основы искусственного интеллекта и архитектуры трансформеров.
Трансформер вычисляет механизм внимания между словами в контекстном окне предложения.
Обычный градиентный спуск и оптимизатор AdamW часто страдают от катастрофического забывания при дообучении на малых данных.
Ядро MF Ardon строит реляционный граф связей между понятиями и защищает опорные слова-якоря от стирания.
Благодаря упругому переносу импульса модель запоминает смысловой кластер целиком уже с первых эпох обучения.
Для запуска модели на обычном ноутбуке не требуется видеокарта, достаточно процессора и ста мегабайт оперативной памяти.
"""

SAMPLE_TXT_2 = """Конспект по физике, биологии и космонавтике.
Квантовая механика описывает волновую функцию и энергетические уровни электрона уравнением Шрёдингера.
Клеточное дыхание в митохондриях превращает глюкозу и кислород в молекулы АТФ, обеспечивая клетку энергией.
Ракетный двигатель создаёт реактивную тягу за счёт истечения раскалённых газов из камеры сгорания через сопло Лаваля.
Многоступенчатые ракеты выводят спутники и космические корабли на околоземную орбиту.
"""


def ensure_data_folder_exists(folder_path: str) -> List[str]:
    os.makedirs(folder_path, exist_ok=True)
    txt_files = sorted(glob.glob(os.path.join(folder_path, "**", "*.txt"), recursive=True))
    if not txt_files:
        f1 = os.path.join(folder_path, "01_конспект_ии_и_ардон.txt")
        f2 = os.path.join(folder_path, "02_конспект_наука_и_космос.txt")
        with open(f1, "w", encoding="utf-8") as f:
            f.write(SAMPLE_TXT_1.strip() + "\n")
        with open(f2, "w", encoding="utf-8") as f:
            f.write(SAMPLE_TXT_2.strip() + "\n")
        txt_files = [f1, f2]
        print(f"  [i] Папка '{folder_path}' была пуста — созданы 2 демонстрационных конспекта (.txt).")
    return txt_files


def read_text_safely(filepath: str) -> str:
    with open(filepath, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "utf-8", "cp1251", "utf-16", "latin1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="ignore")


def tokenize_content_words(text: str) -> List[str]:
    raw_words = re.findall(r"[a-zа-яё0-9]+(?:-[a-zа-яё0-9]+)?", text.lower().replace("ё", "е"))
    return [w for w in raw_words if len(w) >= 3 and w not in STOPWORDS_RU_EN]


def build_corpus_from_txt_files(txt_files: List[str], max_vocab: int = 4000) -> Dict[str, Any]:
    word_counts: Dict[str, int] = {}
    passages: List[Dict[str, Any]] = []

    for fp in txt_files:
        fname = os.path.basename(fp)
        text = read_text_safely(fp)
        raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        for sent in raw_sentences:
            sent_clean = sent.strip()
            if len(sent_clean) < 12:
                continue
            toks = tokenize_content_words(sent_clean)
            if len(toks) < 2:
                continue
            for w in toks:
                word_counts[w] = word_counts.get(w, 0) + 1
            passages.append({
                "source": fname,
                "text": sent_clean,
                "tokens": toks,
            })

    sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))[:max_vocab]
    words = [w for w, _ in sorted_words]
    word_to_id = {w: idx for idx, w in enumerate(words)}

    n_anchors = max(1, min(len(words) // 8, 80))
    token_groups: Dict[int, str] = {}
    for idx in range(len(words)):
        token_groups[idx] = "ANCHOR" if idx < n_anchors else f"CONCEPT_{idx}"

    windows: List[List[int]] = []
    training_pairs: List[Tuple[List[int], int]] = []
    passage_records: List[Dict[str, Any]] = []

    for p in passages:
        seq_ids = [word_to_id[w] for w in p["tokens"] if w in word_to_id]
        if len(seq_ids) < 2:
            continue
        windows.append(seq_ids)
        passage_records.append({
            "source": p["source"],
            "text": p["text"],
            "token_ids": seq_ids,
        })
        for i in range(len(seq_ids) - 1):
            ctx = seq_ids[max(0, i - 1): i + 1]
            tgt = seq_ids[i + 1]
            if tgt not in ctx:
                training_pairs.append((ctx, tgt))
            if i + 2 < len(seq_ids) and seq_ids[i] != seq_ids[i + 2]:
                training_pairs.append(([seq_ids[i], seq_ids[i + 2]], seq_ids[i + 1]))

    return {
        "words": words,
        "word_to_id": word_to_id,
        "token_groups": token_groups,
        "windows": windows,
        "training_pairs": training_pairs,
        "passages": passage_records,
    }


# ==============================================================================
# 4. ЛОКАЛЬНЫЙ TRANSFORMER И СЕТЕВОЙ КЛИЕНТ К ПОРТУ 8001 (С ОФЛАЙН-РЕЗЕРВОМ)
# ==============================================================================

class StudentTransformerModel:
    def __init__(self, vocab_size: int, d_model: int = D_MODEL, seed: int = 42):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.temperature = 0.25
        rng = np.random.default_rng(seed)

        raw_E = rng.normal(0.0, 1.0, size=(vocab_size, d_model)).astype(np.float32)
        self.E = raw_E / (np.linalg.norm(raw_E, axis=1, keepdims=True) + 1e-8)

        self.W_Q = (np.eye(d_model) + rng.normal(0.0, 0.04, size=(d_model, d_model))).astype(np.float32)
        self.W_K = (np.eye(d_model) + rng.normal(0.0, 0.04, size=(d_model, d_model))).astype(np.float32)
        self.W_V = (np.eye(d_model) + rng.normal(0.0, 0.04, size=(d_model, d_model))).astype(np.float32)

        self.co_edges: Dict[str, float] = {}

    def observe_local_window(self, seq_ids: List[int], window_size: int = 7) -> None:
        n = len(seq_ids)
        for i in range(n):
            a = int(seq_ids[i])
            for j in range(i + 1, min(n, i + window_size)):
                b = int(seq_ids[j])
                if a == b:
                    continue
                decay = 1.0 / (1.0 + 0.15 * (j - i - 1))
                k = f"{min(a, b)}:{max(a, b)}"
                old_v = self.co_edges.get(k, 0.0)
                self.co_edges[k] = round(min(0.95, 0.90 * old_v + 0.18 * decay), 4)

    def get_neighbors(self, token_id: int, top_k: int = 6) -> List[Tuple[int, float]]:
        out = []
        prefix_a = f"{token_id}:"
        suffix_b = f":{token_id}"
        for k, strength in self.co_edges.items():
            if k.startswith(prefix_a):
                other = int(k.split(":")[1])
                out.append((other, strength))
            elif k.endswith(suffix_b):
                other = int(k.split(":")[0])
                out.append((other, strength))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:top_k]

    def forward_context(self, context_ids: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ctx_vecs = self.E[context_ids]
        Q = ctx_vecs @ self.W_Q
        K = ctx_vecs @ self.W_K
        V = ctx_vecs @ self.W_V

        scores = (Q @ K.T) / math.sqrt(self.d_model)
        scores -= np.max(scores, axis=-1, keepdims=True)
        attn = np.exp(scores)
        attn /= np.sum(attn, axis=-1, keepdims=True) + 1e-8

        attended = attn @ V
        h_ctx = np.mean(attended + ctx_vecs, axis=0)
        h_ctx /= np.linalg.norm(h_ctx) + 1e-8

        logits = (self.E @ h_ctx) / self.temperature
        logits_shifted = logits - np.max(logits)
        probs = np.exp(logits_shifted)
        probs /= np.sum(probs) + 1e-8
        return h_ctx, logits, probs, attn

    def apply_update(
        self,
        sparse_deltas: Dict[str, List[float]],
        grad_h_ctx: np.ndarray,
        context_ids: List[int],
        anchor_ids: set[int],
        attn_lr: float = 0.005,
    ) -> None:
        saved_anchors = {a: self.E[a].copy() for a in anchor_ids if 0 <= a < self.vocab_size}

        if context_ids and grad_h_ctx is not None:
            ctx_vecs = self.E[context_ids]
            L = len(context_ids)
            Q = ctx_vecs @ self.W_Q
            K = ctx_vecs @ self.W_K
            V = ctx_vecs @ self.W_V
            scores = (Q @ K.T) / math.sqrt(self.d_model)
            scores -= np.max(scores, axis=-1, keepdims=True)
            attn = np.exp(scores)
            attn /= np.sum(attn, axis=-1, keepdims=True) + 1e-8

            d_attended = np.outer(np.ones(L, dtype=np.float32) / max(1, L), grad_h_ctx)
            self.W_V -= attn_lr * (ctx_vecs.T @ (attn.T @ d_attended))
            d_attn = d_attended @ V.T
            d_scores = attn * (d_attn - np.sum(d_attn * attn, axis=-1, keepdims=True)) / math.sqrt(self.d_model)
            self.W_Q -= attn_lr * (ctx_vecs.T @ (d_scores @ K))
            self.W_K -= attn_lr * (ctx_vecs.T @ (d_scores.T @ Q))

        for idx_str, d_vec in sparse_deltas.items():
            idx = int(idx_str)
            if 0 <= idx < self.vocab_size and idx not in anchor_ids:
                self.E[idx] += np.asarray(d_vec, dtype=np.float32)

        self.E /= np.linalg.norm(self.E, axis=1, keepdims=True) + 1e-8

        for a, vec in saved_anchors.items():
            self.E[a] = vec


class ArdonPort8001Client:
    def __init__(self, server_url: str, api_key: str, state_file: str = CHECKPOINT_STATE_FILE):
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key.strip()
        self.state_file = state_file
        self.state_blob: Optional[str] = self._load_state()
        self.online_available = False
        self.status_message = ""

    def _load_state(self) -> Optional[str]:
        if self.state_file and os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    s = f.read().strip()
                    return s if s else None
            except Exception:
                return None
        return None

    def save_state(self) -> None:
        if self.state_file and self.state_blob:
            try:
                with open(self.state_file, "w", encoding="utf-8") as f:
                    f.write(self.state_blob)
            except Exception:
                pass

    def check_connection(self) -> bool:
        if not self.api_key or "ВСТАВЬТЕ" in self.api_key or self.api_key == "YOUR_KEY_HERE":
            self.online_available = False
            self.status_message = "Ключ API ещё не вписан -> включён автономный учебный режим на CPU"
            return False

        try:
            req = urllib.request.Request(
                f"{self.server_url}/v1/health",
                headers={"X-Api-Key": self.api_key},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                if resp.status == 200:
                    self.online_available = True
                    self.status_message = f"Подключено к серверу MF Ardon ({self.server_url})"
                    return True
        except Exception as e:
            self.online_available = False
            self.status_message = f"Сервер {self.server_url} недоступен ({type(e).__name__}) -> работает локальный резерв CPU"
        return False

    def post_observe(self, vocab_size: int, windows: List[List[int]]) -> None:
        if not self.online_available:
            return
        try:
            payload = json.dumps({
                "vocab_size": vocab_size,
                "windows": windows[:64],
                "state_blob": self.state_blob,
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{self.server_url}/v1/transformer/observe",
                data=payload,
                headers={"Content-Type": "application/json", "X-Api-Key": self.api_key},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self.state_blob = data.get("state_blob", self.state_blob)
        except Exception:
            self.online_available = False

    def step_batch_or_local(
        self,
        model: StudentTransformerModel,
        batch_pairs: List[Tuple[List[int], int]],
        token_groups: Dict[int, str],
        epoch_idx: int,
        lr: float = 0.16,
    ) -> float:
        if self.online_available:
            try:
                steps_payload = []
                for ctx_ids, tgt_id in batch_pairs:
                    h_ctx, _, _, _ = model.forward_context(ctx_ids)
                    steps_payload.append({
                        "h_ctx": [round(float(x), 6) for x in h_ctx],
                        "context_ids": [int(i) for i in ctx_ids],
                        "target_id": int(tgt_id),
                    })
                body = json.dumps({
                    "steps": steps_payload,
                    "embeddings": [[round(float(x), 5) for x in row] for row in model.E],
                    "token_groups": {str(k): str(v) for k, v in token_groups.items()},
                    "lr": float(lr),
                    "temperature": float(model.temperature),
                    "state_blob": self.state_blob,
                }).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.server_url}/v1/transformer/loss_batch",
                    data=body,
                    headers={"Content-Type": "application/json", "X-Api-Key": self.api_key},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                self.state_blob = data.get("state_blob", self.state_blob)
                anchor_ids = set(int(a) for a in data.get("metrics", {}).get("anchor_token_ids", []))
                all_ctx = [c for (c_list, _) in batch_pairs for c in c_list]
                model.apply_update(
                    sparse_deltas=data["sparse_deltas"],
                    grad_h_ctx=np.asarray(data["grad_h_ctx"], dtype=np.float32),
                    context_ids=all_ctx,
                    anchor_ids=anchor_ids,
                )
                return float(data["loss"])
            except Exception:
                self.online_available = False

        losses = []
        active_anchors = (
            {idx for idx, g in token_groups.items() if g == "ANCHOR"}
            if epoch_idx >= 3 else set()
        )

        for ctx_ids, tgt_id in batch_pairs:
            h_ctx, _, probs, _ = model.forward_context(ctx_ids)
            loss = -math.log(max(1e-9, float(probs[tgt_id])))
            losses.append(loss)

            grad_logits = probs.copy()
            grad_logits[tgt_id] -= 1.0
            delta_E: Dict[str, List[float]] = {}

            main_shift = -lr * (grad_logits[tgt_id] * h_ctx) / model.temperature
            delta_E[str(tgt_id)] = main_shift.tolist()

            for nbr_id, strength in model.get_neighbors(tgt_id, top_k=5):
                if nbr_id != tgt_id and nbr_id not in active_anchors:
                    coupled_shift = lr * 0.45 * strength * (h_ctx - model.E[nbr_id])
                    delta_E[str(nbr_id)] = coupled_shift.tolist()

            grad_h = (model.E.T @ grad_logits) / model.temperature
            for cid in ctx_ids:
                if cid not in active_anchors:
                    c_shift = -lr * 0.25 * grad_h / max(1, len(ctx_ids))
                    delta_E[str(cid)] = c_shift.tolist()

            model.apply_update(delta_E, grad_h, ctx_ids, active_anchors)

        return float(np.mean(losses)) if losses else 0.0


# ==============================================================================
# 5. СОХРАНЕНИЕ И ПРОГРЕСС-БАР
# ==============================================================================

def save_trained_checkpoint(
    model: StudentTransformerModel,
    corpus: Dict[str, Any],
    client: ArdonPort8001Client,
    completed_epochs: int,
    last_loss: float,
    folder_name: str,
) -> None:
    np.savez_compressed(
        CHECKPOINT_WEIGHTS_FILE,
        E=model.E,
        W_Q=model.W_Q,
        W_K=model.W_K,
        W_V=model.W_V,
    )
    memory_payload = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "server_url": SERVER_URL,
        "data_folder": folder_name,
        "completed_epochs": completed_epochs,
        "last_loss": round(float(last_loss), 4),
        "d_model": model.d_model,
        "vocab_size": model.vocab_size,
        "words": corpus["words"],
        "token_groups": {str(k): v for k, v in corpus["token_groups"].items()},
        "co_edges": model.co_edges,
        "passages": corpus["passages"],
    }
    with open(CHECKPOINT_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory_payload, f, ensure_ascii=False, indent=2)
    client.save_state()


def render_progress_bar(
    epoch: int,
    total_epochs: int,
    step: int,
    total_steps: int,
    loss: float,
    start_time: float,
    bar_width: int = 24,
) -> None:
    overall_done = (epoch - 1) * total_steps + step
    overall_total = max(1, total_epochs * total_steps)
    frac = min(1.0, max(0.0, overall_done / overall_total))
    filled = int(round(bar_width * frac))
    bar = "█" * filled + "░" * (bar_width - filled)

    elapsed = max(0.001, time.time() - start_time)
    eta = (elapsed / max(1, overall_done)) * max(0, overall_total - overall_done)
    ram_mb = get_ram_usage_mb()

    msg = (
        f"\r  [{bar}] {frac * 100:5.1f}% | "
        f"Эпоха {epoch}/{total_epochs} (Шаг {step}/{total_steps}) | "
        f"Loss: {loss:6.3f} | ОЗУ: {ram_mb:4.1f} МБ | "
        f"Осталось: ~{int(eta):02d}с   "
    )
    sys.stdout.write(msg)
    sys.stdout.flush()


# ==============================================================================
# 6. ГЛАВНЫЙ ЦИКЛ ОБУЧЕНИЯ С ИНТЕРАКТИВНЫМ ВЫБОРОМ ПАПКИ
# ==============================================================================

def main() -> None:
    setup_console()
    parser = argparse.ArgumentParser(description="MF Ardon v4.0 — Обучение модели на папке с .txt файлами")
    parser.add_argument("--folder", type=str, default="", help="Папка с .txt файлами для обучения")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Число эпох обучения")
    parser.add_argument("--key", type=str, default="", help="Ваш API-ключ для сервера 162.35.31.11:8001")
    parser.add_argument("--non-interactive", action="store_true", help="Запуск без ожидания ввода папки с клавиатуры")
    args = parser.parse_args()

    effective_key = args.key.strip() if args.key.strip() else API_KEY
    target_folder = args.folder.strip() if args.folder.strip() else DATA_FOLDER

    W = 92
    print("=" * W)
    print("  MF ARDON v4.0 — ОБУЧЕНИЕ ПУСТОЙ МОДЕЛИ НА ВАШИХ .TXT ФАЙЛАХ")
    print("=" * W)
    print(f"  Сервер ядра:   {SERVER_URL}")
    print(f"  API-Ключ:      {effective_key[:14] + '...' if 'ВСТАВЬТЕ' not in effective_key else '[НЕ ЗАДАН — Вставьте ключ в строке 36 файла ardon_train.py]'}")

    if not args.folder and not args.non_interactive and sys.stdin and sys.stdin.isatty():
        try:
            user_inp = input(f"\n  Введите название папки с .txt файлами [Enter = '{DATA_FOLDER}']: ").strip()
            if user_inp:
                target_folder = user_inp.strip('"').strip("'")
        except EOFError:
            pass

    txt_files = ensure_data_folder_exists(target_folder)
    print(f"\n  [1/3] Сканирование папки '{os.path.abspath(target_folder)}'...")
    print(f"        Найдено .txt файлов: {len(txt_files)}")
    for idx_f, fp in enumerate(txt_files[:8], 1):
        sz_kb = os.path.getsize(fp) / 1024.0
        print(f"        {idx_f}. {os.path.basename(fp)} ({sz_kb:.1f} КБ)")
    if len(txt_files) > 8:
        print(f"        ... и ещё {len(txt_files) - 8} файлов.")

    corpus = build_corpus_from_txt_files(txt_files)
    V = len(corpus["words"])
    pairs = corpus["training_pairs"]
    if V < 3 or not pairs:
        print("\n  [!] В файлах слишком мало текста. Добавьте хотя бы несколько предложений в .txt файлы.")
        sys.exit(1)

    n_anchors = sum(1 for g in corpus["token_groups"].values() if g == "ANCHOR")
    print(f"\n  [2/3] Построение чистой модели Transformer:")
    print(f"        • Словарь терминов:         {V} слов (опорных якорей ANCHOR: {n_anchors})")
    print(f"        • Осмысленных фрагментов:   {len(corpus['passages'])} предложений")
    print(f"        • Учебных контекстных пар:  {len(pairs)} связей")
    print(f"        • Потребление ОЗУ сейчас:   {get_ram_usage_mb():.1f} МБ")

    model = StudentTransformerModel(vocab_size=V, d_model=D_MODEL, seed=42)
    client = ArdonPort8001Client(server_url=SERVER_URL, api_key=effective_key)
    client.check_connection()
    print(f"        • Статус связи (порт 8001): {client.status_message}")

    for w_seq in corpus["windows"]:
        model.observe_local_window(w_seq, window_size=7)
    client.post_observe(vocab_size=V, windows=corpus["windows"])

    print("\n" + "-" * W)
    print("  [3/3] СТАРТ ОБУЧЕНИЯ МОДЕЛИ")
    print("  >>> КАК ОСТАНОВИТЬ ОБУЧЕНИЕ В ЛЮБОЙ МОМЕНТ БЕЗ ПОТЕРИ ДАННЫХ: <<<")
    print("      Напишите в этой строке слово  stop  (или  стоп ) и нажмите Enter,")
    print("      либо нажмите Ctrl+C — модель сразу сохранит всё, что успела выучить!")
    print("-" * W)

    stop_ctrl = SafeStopController()
    stop_ctrl.start_command_listener()

    total_epochs = max(1, int(args.epochs))
    batches_per_epoch = max(1, math.ceil(len(pairs) / BATCH_SIZE))
    start_ts = time.time()
    last_loss = 0.0
    completed_epochs = 0

    for ep in range(1, total_epochs + 1):
        if stop_ctrl.stop_requested:
            break

        rng = np.random.default_rng(42 + ep)
        order = rng.permutation(len(pairs))
        ep_losses = []

        for b_idx in range(batches_per_epoch):
            if stop_ctrl.stop_requested:
                break

            batch_slice = order[b_idx * BATCH_SIZE: (b_idx + 1) * BATCH_SIZE]
            batch_pairs = [pairs[int(i)] for i in batch_slice]

            step_loss = client.step_batch_or_local(
                model=model,
                batch_pairs=batch_pairs,
                token_groups=corpus["token_groups"],
                epoch_idx=ep,
                lr=0.16,
            )
            ep_losses.append(step_loss)
            last_loss = float(np.mean(ep_losses))

            render_progress_bar(
                epoch=ep,
                total_epochs=total_epochs,
                step=b_idx + 1,
                total_steps=batches_per_epoch,
                loss=last_loss,
                start_time=start_ts,
            )
            time.sleep(0.025)

        completed_epochs = ep
        save_trained_checkpoint(model, corpus, client, completed_epochs, last_loss, target_folder)

    print()
    save_trained_checkpoint(model, corpus, client, max(1, completed_epochs), last_loss, target_folder)

    total_sec = time.time() - start_ts
    print("-" * W)
    if stop_ctrl.stop_requested:
        print(f"  [СТОП] Обучение аккуратно остановлено пользователем ({stop_ctrl.stop_reason}).")
        print(f"         Ни один шаг не потерян! Все выученные связи сохранены.")
    else:
        print(f"  [ГОТОВО] Обучение полностью завершено за {total_sec:.1f} сек!")

    print(f"  • Сохранены веса модели:   {os.path.abspath(CHECKPOINT_WEIGHTS_FILE)}")
    print(f"  • Сохранена память текста: {os.path.abspath(CHECKPOINT_MEMORY_FILE)} ({V} понятий, {len(model.co_edges)} связей)")
    print(f"  • Итоговое потребление ОЗУ: {get_ram_usage_mb():.1f} МБ")
    print("=" * W)
    print("  Теперь запустите чат для общения с вашей моделью:")
    print("      python ardon_chat.py")
    print("=" * W)


if __name__ == "__main__":
    main()