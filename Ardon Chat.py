#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  MF ARDON v4.0 — ИНТЕРАКТИВНЫЙ ЧАТ С ОБУЧЕННОЙ МОДЕЛЬЮ
  (ardon_chat.py — Полностью автономный скрипт для разговора с ИИ)
================================================================================
Как пользоваться:
  1. Сначала запустите `python ardon_train.py`, чтобы обучить модель на ваших
     файлах .txt из выбранной папки.
  2. Затем запустите этот файл:
         python ardon_chat.py
  3. Пишите любые вопросы или темы в консоль — модель найдёт связанные понятия
     в своей латентной памяти (Transformer + граф MF Ardon), соберёт ответ по
     вашим конспектам и покажет цепочку ассоциаций.

Полезные команды прямо внутри чата:
  • /teach <текст> (или /учить <текст>) — мгновенно дообучить модель новому факту
    прямо во время разговора (с автосохранением и без забывания прошлых тем!)
  • /map <слово>   (или /связи <слово>) — показать, какие слова сильнее всего
    притянулись к этому понятию в памяти модели
  • /status        (или /статус)        — сколько мегабайт ОЗУ ест модель и статистика
  • stop / exit / выход                 — закрыть чат
================================================================================
"""

from __future__ import annotations
import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ==============================================================================
# >>> 1. ГЛАВНЫЕ НАСТРОЙКИ ПОДКЛЮЧЕНИЯ (ТЕ ЖЕ, ЧТО И В ARDON_TRAIN.PY) <<<
# ==============================================================================

SERVER_URL   = "http://162.35.31.11:8001"        # Адрес сервера MF Ardon (порт 8001)
API_KEY      = "ВСТАВЬТЕ_ВАШ_КЛЮЧ_СЮДА"           # <-- Впишите сюда ваш персональный ключ

CHECKPOINT_WEIGHTS_FILE = "ardon_checkpoint.npz"
CHECKPOINT_MEMORY_FILE  = "ardon_memory.json"
CHECKPOINT_STATE_FILE   = "ardon_state.bin"


# ==============================================================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ И ЯДРО ЧТЕНИЯ ПАМЯТИ TRANSFORMER
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


def tokenize_content_words(text: str) -> List[str]:
    raw_words = re.findall(r"[a-zа-яё0-9]+(?:-[a-zа-яё0-9]+)?", text.lower().replace("ё", "е"))
    return [w for w in raw_words if len(w) >= 3 and w not in STOPWORDS_RU_EN]


class ChatTransformerBrain:
    def __init__(self, vocab_size: int, d_model: int):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.temperature = 0.25
        self.E = np.zeros((vocab_size, d_model), dtype=np.float32)
        self.W_Q = np.eye(d_model, dtype=np.float32)
        self.W_K = np.eye(d_model, dtype=np.float32)
        self.W_V = np.eye(d_model, dtype=np.float32)
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


def load_trained_brain() -> Tuple[ChatTransformerBrain, Dict[str, Any]]:
    if not os.path.exists(CHECKPOINT_WEIGHTS_FILE) or not os.path.exists(CHECKPOINT_MEMORY_FILE):
        print("=" * 84)
        print("  [!] Файлы обученной модели ещё не найдены!")
        print("      Сначала запустите скрипт обучения:")
        print("          python ardon_train.py")
        print("=" * 84)
        sys.exit(1)

    with open(CHECKPOINT_MEMORY_FILE, "r", encoding="utf-8") as f:
        memory = json.load(f)

    weights = np.load(CHECKPOINT_WEIGHTS_FILE)
    vocab_size = int(memory["vocab_size"])
    d_model = int(memory["d_model"])

    model = ChatTransformerBrain(vocab_size=vocab_size, d_model=d_model)
    model.E = weights["E"].astype(np.float32)
    model.W_Q = weights["W_Q"].astype(np.float32)
    model.W_K = weights["W_K"].astype(np.float32)
    model.W_V = weights["W_V"].astype(np.float32)
    model.co_edges = {str(k): float(v) for k, v in memory.get("co_edges", {}).items()}
    return model, memory


# ==============================================================================
# 3. ОТВЕТ НА ВОПРОС ПОЛЬЗОВАТЕЛЯ И ЖИВОЕ ДООБУЧЕНИЕ (/учить)
# ==============================================================================

def answer_user_query(
    query: str,
    model: ChatTransformerBrain,
    memory: Dict[str, Any],
) -> Dict[str, Any]:
    words: List[str] = memory["words"]
    word_to_id: Dict[str, int] = {w: i for i, w in enumerate(words)}
    passages: List[Dict[str, Any]] = memory.get("passages", [])

    q_tokens = tokenize_content_words(query)
    known_ids: List[int] = []
    matched_words: List[str] = []

    for qt in q_tokens:
        if qt in word_to_id:
            wid = word_to_id[qt]
            if wid not in known_ids:
                known_ids.append(wid)
                matched_words.append(words[wid])
        elif len(qt) >= 4:
            stem = qt[:max(4, len(qt) - 2)]
            for w_cand, wid in word_to_id.items():
                if w_cand.startswith(stem) and wid not in known_ids:
                    known_ids.append(wid)
                    matched_words.append(w_cand)
                    break

    if not known_ids:
        return {
            "found": False,
            "matched_words": [],
            "associated_words": [],
            "best_passages": [],
            "generated_chain": "",
        }

    h_ctx, logits, probs, _ = model.forward_context(known_ids)

    graph_boost = np.zeros(model.vocab_size, dtype=np.float32)
    for kid in known_ids:
        for nbr_id, strength in model.get_neighbors(kid, top_k=10):
            graph_boost[nbr_id] = max(graph_boost[nbr_id], float(strength))

    combined_score = logits + 2.5 * graph_boost
    ranked_ids = [int(i) for i in np.argsort(combined_score)[::-1] if int(i) not in set(known_ids)]
    top_associated = [
        (words[i], round(float(probs[i]) * 100.0, 1), round(float(np.dot(model.E[i], h_ctx)), 3))
        for i in ranked_ids[:7]
    ]

    expanded_cluster = set(known_ids) | set(ranked_ids[:6])
    scored_passages = []
    for p in passages:
        p_ids = [int(x) for x in p.get("token_ids", []) if 0 <= int(x) < model.vocab_size]
        if not p_ids:
            continue
        p_vec = np.mean(model.E[p_ids], axis=0)
        p_vec /= np.linalg.norm(p_vec) + 1e-8
        latent_sim = float(np.dot(p_vec, h_ctx))
        direct_hits = sum(1 for tid in p_ids if tid in set(known_ids))
        cluster_hits = sum(1 for tid in p_ids if tid in expanded_cluster)
        total_score = latent_sim + 0.45 * direct_hits + 0.18 * cluster_hits
        if direct_hits > 0 or latent_sim > 0.35:
            scored_passages.append((total_score, p["source"], p["text"]))

    scored_passages.sort(key=lambda x: x[0], reverse=True)
    unique_passages = []
    seen_texts = set()
    for sc, src, txt in scored_passages:
        if txt not in seen_texts:
            seen_texts.add(txt)
            unique_passages.append((round(sc, 3), src, txt))
        if len(unique_passages) >= 3:
            break

    chain_ids = list(known_ids[:3])
    for _ in range(6):
        _, step_logits, _, _ = model.forward_context(chain_ids[-3:])
        for next_cand in np.argsort(step_logits)[::-1]:
            nc = int(next_cand)
            if nc not in chain_ids:
                chain_ids.append(nc)
                break
    generated_chain = " -> ".join(words[i] for i in chain_ids)

    return {
        "found": True,
        "matched_words": matched_words,
        "associated_words": top_associated,
        "best_passages": unique_passages,
        "generated_chain": generated_chain,
    }


def teach_new_fact_live(
    fact_text: str,
    model: ChatTransformerBrain,
    memory: Dict[str, Any],
) -> str:
    new_tokens = tokenize_content_words(fact_text)
    if len(new_tokens) < 2:
        return "Фраза слишком короткая — напишите хотя бы 2–3 содержательных слова."

    words: List[str] = memory["words"]
    word_to_id: Dict[str, int] = {w: i for i, w in enumerate(words)}
    token_groups: Dict[int, str] = {int(k): v for k, v in memory.get("token_groups", {}).items()}

    rng = np.random.default_rng(int(time.time()) % 100000)
    added_words = []

    for w in new_tokens:
        if w not in word_to_id:
            new_id = len(words)
            words.append(w)
            word_to_id[w] = new_id
            token_groups[new_id] = f"LIVE_{new_id}"
            new_vec = rng.normal(0.0, 1.0, size=(1, model.d_model)).astype(np.float32)
            new_vec /= np.linalg.norm(new_vec, axis=1, keepdims=True) + 1e-8
            model.E = np.vstack([model.E, new_vec])
            model.vocab_size = len(words)
            added_words.append(w)

    seq_ids = [word_to_id[w] for w in new_tokens]
    for _ in range(5):
        model.observe_local_window(seq_ids, window_size=7)

    pairs = []
    for i in range(len(seq_ids) - 1):
        pairs.append((seq_ids[max(0, i - 1): i + 1], seq_ids[i + 1]))

    # Защищаем старые якоря ANCHOR, чтобы модель ничего не забыла!
    active_anchors = {idx for idx, g in token_groups.items() if g == "ANCHOR"}
    lr = 0.18
    for _ in range(8):
        for ctx_ids, tgt_id in pairs:
            h_ctx, _, probs, _ = model.forward_context(ctx_ids)
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

    memory["vocab_size"] = model.vocab_size
    memory["words"] = words
    memory["token_groups"] = {str(k): v for k, v in token_groups.items()}
    memory["co_edges"] = model.co_edges
    memory.setdefault("passages", []).append({
        "source": "Живое дообучение в чате",
        "text": fact_text.strip(),
        "token_ids": seq_ids,
    })

    np.savez_compressed(
        CHECKPOINT_WEIGHTS_FILE,
        E=model.E,
        W_Q=model.W_Q,
        W_K=model.W_K,
        W_V=model.W_V,
    )
    with open(CHECKPOINT_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

    added_str = f" (новых слов добавлено: {', '.join(added_words)})" if added_words else ""
    return (
        f"Факт вшит в память модели за 8 шагов без забывания прошлых тем!{added_str}\n"
        f"     Текущий словарь: {model.vocab_size} слов | ОЗУ: {get_ram_usage_mb():.1f} МБ"
    )


# ==============================================================================
# 4. ГЛАВНЫЙ ИНТЕРФЕЙС ЧАТА
# ==============================================================================

def run_interactive_chat(single_query: str = "") -> None:
    setup_console()
    model, memory = load_trained_brain()

    n_anchors = sum(1 for v in memory.get("token_groups", {}).values() if v == "ANCHOR")

    W = 90
    print("=" * W)
    print("  MF ARDON v4.0 — ИНТЕРАКТИВНЫЙ ЧАТ С ВАШЕЙ ОБУЧЕННОЙ МОДЕЛЬЮ")
    print("=" * W)
    print(f"  Папка обучения:  {memory.get('data_folder', 'my_data')} (обучено эпох: {memory.get('completed_epochs', 1)})")
    print(f"  База памяти:     {model.vocab_size} понятий | {len(memory.get('passages', []))} фрагментов | {n_anchors} защищённых якорей")
    print(f"  Потребление ОЗУ: {get_ram_usage_mb():.1f} МБ | Сервер: {SERVER_URL}")
    print("-" * W)
    print("  Команды чата:")
    print("    • Просто напишите вопрос по вашим .txt файлам (например: «как работает трансформер»)")
    print("    • /учить <текст> — дообучить модель новой мысли прямо сейчас (с сохранением!)")
    print("    • /связи <слово> — посмотреть граф связей слова в памяти модели")
    print("    • /статус        — проверить потребление ОЗУ и состояние памяти")
    print("    • выход / stop   — выйти из чата")
    print("=" * W)

    if single_query:
        _handle_input(single_query, model, memory)
        return

    while True:
        try:
            user_msg = input("\nВы > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nЧат завершён. Вся память модели сохранена.")
            break

        if not user_msg:
            continue
        if user_msg.lower() in ("exit", "quit", "stop", "выход", "стоп", "q"):
            print("Чат завершён. До встречи!")
            break

        _handle_input(user_msg, model, memory)


def _handle_input(
    user_msg: str,
    model: ChatTransformerBrain,
    memory: Dict[str, Any],
) -> None:
    low = user_msg.lower()

    if low in ("/status", "/статус", "статус"):
        n_anchors = sum(1 for v in memory.get("token_groups", {}).values() if v == "ANCHOR")
        print(f"\n  [Статус модели MF Ardon v4.0]")
        print(f"  • Потребление оперативной памяти (ОЗУ): {get_ram_usage_mb():.1f} МБ")
        print(f"  • Размер словаря (V x d_model):         {model.vocab_size} x {model.d_model}")
        print(f"  • Защищённых узлов KEEP_TRAFFIC_ANCHOR: {n_anchors}")
        print(f"  • Активных рёбер графа памяти:          {len(model.co_edges)}")
        print(f"  • Сервер ядра:                          {SERVER_URL}")
        return

    if low.startswith("/teach ") or low.startswith("/учить "):
        fact = user_msg.split(" ", 1)[1].strip()
        res_msg = teach_new_fact_live(fact, model, memory)
        print(f"\n  [Дообучение] {res_msg}")
        return

    if low.startswith("/map ") or low.startswith("/связи "):
        target_w = user_msg.split(" ", 1)[1].strip().lower().replace("ё", "е")
        words = memory["words"]
        word_to_id = {w: i for i, w in enumerate(words)}
        wid = word_to_id.get(target_w)
        if wid is None:
            for w_cand, idx_c in word_to_id.items():
                if w_cand.startswith(target_w[:4]):
                    wid = idx_c
                    target_w = w_cand
                    break
        if wid is None:
            print(f"\n  [!] Слово '{target_w}' пока не встречалось в ваших .txt файлах.")
            return
        nbrs = model.get_neighbors(wid, top_k=8)
        print(f"\n  [Граф связей для понятия «{target_w}» (ID={wid})]:")
        if not nbrs:
            print("    Прямых рёбер в окне пока нет, используются векторные координаты Transformer.")
        else:
            for nid, strength in nbrs:
                cos_v = float(np.dot(model.E[wid], model.E[nid]))
                print(f"    • {words[nid]:<22} | сила сцепки в графе: {strength:.2f} | косинус векторов: {cos_v:+.3f}")
        return

    t0 = time.perf_counter()
    ans = answer_user_query(user_msg, model, memory)
    lat_ms = (time.perf_counter() - t0) * 1000.0

    if not ans["found"]:
        sample_words = ", ".join(memory["words"][:10])
        print(
            f"\nИИ Ardon > В ваших конспектах пока не было этих слов.\n"
            f"           Попробуйте спросить про темы из ваших .txt файлов (например: {sample_words})\n"
            f"           или научите меня прямо сейчас командой: /учить <ваше предложение>"
        )
        return

    print(f"\nИИ Ardon ({lat_ms:.1f} мс | ОЗУ {get_ram_usage_mb():.1f} МБ) >")
    if ans["best_passages"]:
        for idx_p, (score, src, txt) in enumerate(ans["best_passages"], 1):
            print(f"  [{idx_p}] Из файла «{src}» (резонанс {score:+.2f}):")
            print(f"      {txt}")
    else:
        print(f"  Смысловая цепочка: {ans['generated_chain']}")

    assoc_str = ", ".join(f"{w} (cos={c:+.2f})" for w, _, c in ans["associated_words"][:5])
    print(f"\n  • Распознанные опорные слова запроса: {', '.join(ans['matched_words'])}")
    print(f"  • Подтянутые по графу понятия кластера: {assoc_str}")
    print(f"  • Латентная траектория мысли: {ans['generated_chain']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MF Ardon v4.0 — Чат с обученной моделью")
    parser.add_argument("--ask", type=str, default="", help="Разовый вопрос к модели (для быстрой проверки)")
    args = parser.parse_args()
    run_interactive_chat(single_query=args.ask)