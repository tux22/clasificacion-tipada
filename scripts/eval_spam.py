#!/usr/bin/env python3
"""
Evalua un GGUF servido por llama.cpp como clasificador binario al estilo "System One":
n_predict=1 + n_probs, y P(spam) se lee de los logprobs del PRIMER token.
No se parsea texto: la probabilidad ES la salida.

Metricas:
  - accuracy / precision / recall / F1   -> desempeño al umbral elegido
  - AUROC                                -> discriminacion, independiente del umbral
  - Brier                                -> error cuadratico medio de la probabilidad
  - ECE (10 bins)                        -> calibracion: ¿un 0.8 acierta el 80% de las veces?

Uso: eval_spam.py <dataset> [n_muestras] [url_servidor]
"""
import json, random, sys, time, urllib.request

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/SMSSpamCollection"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
URL = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8080/completion"
SEED = 42

PROMPT = (
    "<|im_start|>system\nYou are a spam classifier. Answer with exactly one character: "
    "1 if the message is spam, 0 if it is not spam.<|im_end|>\n"
    "<|im_start|>user\nMessage:\n{msg}\n\nIs this spam? Answer 1 or 0.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def load(path, n, seed=SEED):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "\t" not in line:
                continue
            label, text = line.split("\t", 1)
            label = label.strip()
            if label in ("ham", "spam"):
                rows.append((1 if label == "spam" else 0, text.strip()))
    random.Random(seed).shuffle(rows)
    return rows[:n]


def p_spam(msg, retries=2):
    """P(spam) desde los logprobs del primer token. None si el servidor no responde."""
    body = json.dumps({
        "prompt": PROMPT.format(msg=msg[:2000]),
        "n_predict": 1,
        "n_probs": 20,
        "temperature": 0.0,
        "cache_prompt": False,
    }).encode()
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(URL, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
            break
        except Exception as e:
            if attempt == retries:
                print("  fallo en request: %s" % e, file=sys.stderr)
                return None
            time.sleep(2)

    # el formato de completion_probabilities varia entre builds de llama.cpp
    probs = d.get("completion_probabilities") or []
    if not probs:
        return None
    entry = probs[0]
    cands = entry.get("probs") or entry.get("top_logprobs") or []

    p1 = p0 = 0.0
    for c in cands:
        tok = (c.get("tok_str") or c.get("token") or "").strip()
        pr = c.get("prob")
        if pr is None and "logprob" in c:
            import math
            pr = math.exp(c["logprob"])
        if pr is None:
            continue
        if tok == "1":
            p1 += pr
        elif tok == "0":
            p0 += pr
    tot = p1 + p0
    if tot <= 0:
        return None              # el modelo no puso masa en 1 ni en 0
    return p1 / tot              # normalizado: P(spam | respondio 1 o 0)


def auroc(ys, ps):
    """AUROC por conteo de pares concordantes, con empates a media."""
    pos = [p for y, p in zip(ys, ps) if y == 1]
    neg = [p for y, p in zip(ys, ps) if y == 0]
    if not pos or not neg:
        return float("nan")
    pairs = 0.0
    for a in pos:
        for b in neg:
            pairs += 1.0 if a > b else (0.5 if a == b else 0.0)
    return pairs / (len(pos) * len(neg))


def ece(ys, ps, bins=10):
    """Expected Calibration Error: |confianza - acierto| ponderado por bin."""
    tot = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        sel = [(y, p) for y, p in zip(ys, ps) if (lo < p <= hi or (i == 0 and p == 0))]
        if not sel:
            continue
        conf = sum(p for _, p in sel) / len(sel)
        acc = sum(y for y, _ in sel) / len(sel)
        tot += (len(sel) / len(ys)) * abs(conf - acc)
    return tot


def main():
    rows = load(DATA, N)
    print("dataset: %s  muestras: %d  (spam=%d ham=%d)"
          % (DATA, len(rows), sum(y for y, _ in rows), sum(1 - y for y, _ in rows)))
    ys, ps, skipped = [], [], 0
    t0 = time.time()
    for i, (y, msg) in enumerate(rows, 1):
        p = p_spam(msg)
        if p is None:
            skipped += 1
            continue
        ys.append(y); ps.append(p)
        if i % 50 == 0:
            el = time.time() - t0
            print("  %d/%d  %.1f msg/s" % (i, len(rows), i / el), flush=True)
    el = time.time() - t0

    if not ys:
        print("Sin resultados: el servidor no devolvio probabilidades utilizables.")
        return 1

    # barrido de umbral para reportar el mejor F1, ademas del 0.5 por defecto
    def metrics_at(th):
        tp = sum(1 for y, p in zip(ys, ps) if p >= th and y == 1)
        fp = sum(1 for y, p in zip(ys, ps) if p >= th and y == 0)
        fn = sum(1 for y, p in zip(ys, ps) if p < th and y == 1)
        tn = sum(1 for y, p in zip(ys, ps) if p < th and y == 0)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return (tp + tn) / len(ys), prec, rec, f1, (tp, fp, fn, tn)

    acc, prec, rec, f1, cm = metrics_at(0.5)
    best_th, best_f1 = max(((t / 100, metrics_at(t / 100)[3]) for t in range(1, 100)),
                           key=lambda x: x[1])
    brier = sum((p - y) ** 2 for y, p in zip(ys, ps)) / len(ys)

    print("\n=== resultados (n=%d, %d descartados) ===" % (len(ys), skipped))
    print("  velocidad        %.2f msg/s  (%.1f s total)" % (len(ys) / el, el))
    print("  -- al umbral 0.5 --")
    print("  accuracy         %.4f" % acc)
    print("  precision        %.4f" % prec)
    print("  recall           %.4f" % rec)
    print("  F1               %.4f" % f1)
    print("  confusion        TP=%d FP=%d FN=%d TN=%d" % cm)
    print("  -- independiente del umbral --")
    print("  AUROC            %.4f   (0.5=azar, 1.0=perfecto)" % auroc(ys, ps))
    print("  mejor F1         %.4f al umbral %.2f" % (best_f1, best_th))
    print("  -- calibracion --")
    print("  Brier            %.4f   (menor es mejor)" % brier)
    print("  ECE (10 bins)    %.4f   (0=perfecta)" % ece(ys, ps))
    print("\n  referencia publicada en este dataset: SVM ~0.976 acc | BERT-G3CN 0.9928")
    return 0


if __name__ == "__main__":
    sys.exit(main())
