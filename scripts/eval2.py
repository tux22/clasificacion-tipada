#!/usr/bin/env python3
"""
Evalua un GGUF como clasificador binario con P(spam) leida de los logprobs del primer token.

Anade sobre la v1:
  --fewshot N     N ejemplos en el prompt (tomados del split de calibracion, nunca del test)
  --calibrate N   ajusta temperature scaling y Platt sobre N ejemplos held-out y reporta ambos

Los splits nunca se solapan: calibracion/fewshot salen del final de la lista barajada,
el test del principio. Misma semilla => mismo split.
"""
import json, math, random, sys, time, urllib.request, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/SMSSpamCollection")
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--url", default="http://127.0.0.1:8081/completion")
ap.add_argument("--fewshot", type=int, default=0)
ap.add_argument("--calibrate", type=int, default=0)
ap.add_argument("--label", default="")
ap.add_argument("--template", default="qwen", choices=["qwen", "qwen3-nothink", "gemma"])
A = ap.parse_args()

SYS = ("You are a spam classifier. Answer with exactly one character: "
       "1 if the message is spam, 0 if it is not spam.")


def load(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "\t" not in line:
                continue
            lab, txt = line.split("\t", 1)
            lab = lab.strip()
            if lab in ("ham", "spam"):
                rows.append((1 if lab == "spam" else 0, txt.strip()))
    random.Random(42).shuffle(rows)
    return rows


def build_prompt(msg, shots):
    ask = "Message:\n%s\n\nIs this spam? Answer 1 or 0."

    if A.template == "gemma":
        # gemma-3-it no tiene rol system: la instruccion va en el primer turno de user
        p = ""
        for y, m in shots:
            p += ("<start_of_turn>user\n%s\n\n%s<end_of_turn>\n<start_of_turn>model\n%d<end_of_turn>\n"
                  % (SYS, ask % m[:600], y))
        p += ("<start_of_turn>user\n%s\n\n%s<end_of_turn>\n<start_of_turn>model\n"
              % (SYS, ask % msg[:2000]))
        return p

    # ChatML (qwen2.5 y qwen3)
    p = "<|im_start|>system\n%s<|im_end|>\n" % SYS
    for y, m in shots:
        p += ("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % (ask % m[:600]))
        if A.template == "qwen3-nothink":
            p += "<think>\n\n</think>\n\n"
        p += "%d<|im_end|>\n" % y
    tail = ask % msg[:2000]
    if A.template == "qwen3-nothink":
        # /no_think + bloque think ya cerrado: fuerza que el PRIMER token generado sea 1 o 0
        tail += " /no_think"
        p += ("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" % tail)
    else:
        p += ("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % tail)
    return p


def logit_spam(msg, shots, retries=2):
    """Devuelve el LOGIT (no la prob) para poder aplicar temperature scaling despues."""
    body = json.dumps({"prompt": build_prompt(msg, shots), "n_predict": 1,
                       "n_probs": 20, "temperature": 0.0, "cache_prompt": True}).encode()
    for att in range(retries + 1):
        try:
            req = urllib.request.Request(A.url, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.load(r)
            break
        except Exception as e:
            if att == retries:
                print("  fallo: %s" % e, file=sys.stderr)
                return None
            time.sleep(2)
    probs = d.get("completion_probabilities") or []
    if not probs:
        return None
    cands = probs[0].get("probs") or probs[0].get("top_logprobs") or []
    p1 = p0 = 0.0
    for c in cands:
        tok = (c.get("tok_str") or c.get("token") or "").strip()
        pr = c.get("prob")
        if pr is None and "logprob" in c:
            pr = math.exp(c["logprob"])
        if pr is None:
            continue
        if tok == "1":
            p1 += pr
        elif tok == "0":
            p0 += pr
    if p1 + p0 <= 0:
        return None
    # clamp para evitar log(0) cuando el modelo satura
    p = min(max(p1 / (p1 + p0), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + math.exp(-z)) if z > -700 else 0.0


def auroc(ys, ps):
    pos = [p for y, p in zip(ys, ps) if y == 1]
    neg = [p for y, p in zip(ys, ps) if y == 0]
    if not pos or not neg:
        return float("nan")
    s = sum(1.0 if a > b else (0.5 if a == b else 0.0) for a in pos for b in neg)
    return s / (len(pos) * len(neg))


def ece(ys, ps, bins=10):
    tot = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        sel = [(y, p) for y, p in zip(ys, ps) if lo < p <= hi or (i == 0 and p <= 0)]
        if not sel:
            continue
        conf = sum(p for _, p in sel) / len(sel)
        acc = sum(y for y, _ in sel) / len(sel)
        tot += len(sel) / len(ys) * abs(conf - acc)
    return tot


def f1_at(ys, ps, th):
    tp = sum(1 for y, p in zip(ys, ps) if p >= th and y == 1)
    fp = sum(1 for y, p in zip(ys, ps) if p >= th and y == 0)
    fn = sum(1 for y, p in zip(ys, ps) if p < th and y == 1)
    tn = sum(1 for y, p in zip(ys, ps) if p < th and y == 0)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return ((2 * pr * rc / (pr + rc)) if pr + rc else 0.0,
            (tp + tn) / len(ys), pr, rc, (tp, fp, fn, tn))


def report(tag, ys, ps):
    f1, acc, pr, rc, cm = f1_at(ys, ps, 0.5)
    bf1, bth = max(((f1_at(ys, ps, t / 100)[0], t / 100) for t in range(1, 100)))
    brier = sum((p - y) ** 2 for y, p in zip(ys, ps)) / len(ys)
    print("  %-14s AUROC=%.4f  acc=%.4f P=%.4f R=%.4f F1=%.4f | bestF1=%.4f@%.2f | Brier=%.4f ECE=%.4f"
          % (tag, auroc(ys, ps), acc, pr, rc, f1, bf1, bth, brier, ece(ys, ps)))
    print("                 TP=%d FP=%d FN=%d TN=%d" % cm)


def fit_temperature(ys, zs):
    """Busca T que minimiza log-loss. T>1 suaviza (menos sobreconfianza)."""
    best_t, best_loss = 1.0, float("inf")
    t = 0.05
    while t <= 10.0:
        loss = 0.0
        for y, z in zip(ys, zs):
            p = min(max(sigmoid(z / t), 1e-9), 1 - 1e-9)
            loss -= y * math.log(p) + (1 - y) * math.log(1 - p)
        if loss < best_loss:
            best_loss, best_t = loss, t
        t += 0.05
    return best_t


def fit_platt(ys, zs):
    """p = sigmoid(a*z + b). El sesgo b corrige el desbalance que la temperatura sola no puede."""
    a, b = 1.0, 0.0
    lr, n = 0.05, len(ys)
    for _ in range(3000):
        ga = gb = 0.0
        for y, z in zip(ys, zs):
            p = sigmoid(a * z + b)
            d = p - y
            ga += d * z; gb += d
        a -= lr * ga / n; b -= lr * gb / n
    return a, b


def main():
    rows = load(A.data)
    test = rows[:A.n]
    held = rows[-(A.calibrate + A.fewshot * 4):] if (A.calibrate or A.fewshot) else []

    shots = []
    if A.fewshot:
        # ejemplos balanceados: mitad spam, mitad ham
        sp = [r for r in held if r[0] == 1][:A.fewshot // 2]
        hm = [r for r in held if r[0] == 0][:A.fewshot - len(sp)]
        shots = [x for pair in zip(hm, sp) for x in pair][:A.fewshot]
        held = [r for r in held if r not in shots]

    tag = A.label or ("fewshot=%d cal=%d" % (A.fewshot, A.calibrate))
    print("\n### %s  (test n=%d, spam=%d)" % (tag, len(test), sum(y for y, _ in test)))

    t0 = time.time()
    ys, zs = [], []
    for y, m in test:
        z = logit_spam(m, shots)
        if z is None:
            continue
        ys.append(y); zs.append(z)
    el = time.time() - t0
    ps = [sigmoid(z) for z in zs]
    print("  velocidad      %.1f msg/s (%.1fs)" % (len(ys) / el, el))
    report("sin calibrar", ys, ps)

    if A.calibrate:
        cal = held[:A.calibrate]
        cys, czs = [], []
        for y, m in cal:
            z = logit_spam(m, shots)
            if z is None:
                continue
            cys.append(y); czs.append(z)
        if cys:
            T = fit_temperature(cys, czs)
            print("  T ajustada     %.2f  (sobre %d held-out, sin solape con test)" % (T, len(cys)))
            report("temp-scaling", ys, [sigmoid(z / T) for z in zs])
            a, b = fit_platt(cys, czs)
            print("  Platt          a=%.3f b=%.3f" % (a, b))
            report("platt", ys, [sigmoid(a * z + b) for z in zs])


main()
