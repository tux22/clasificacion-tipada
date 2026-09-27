#!/usr/bin/env python3
"""
Evalua Laya (encoder de decisiones, Convai) en la MISMA tarea y el MISMO split que los GGUF,
para que la comparativa sea valida: dataset UCI SMS Spam, semilla 42, n=400 test + 200 calib.

Laya no lee logprobs: devuelve P(true) directamente via la primitiva `noul`.
Las metricas (AUROC/Brier/ECE) y la calibracion (Platt) son identicas a eval2.py.
"""
import json, math, random, sys, time, warnings, argparse
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/SMSSpamCollection")
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--calibrate", type=int, default=200)
ap.add_argument("--device", default="cpu")          # cpu | cuda
ap.add_argument("--engine", default="laya", choices=["laya", "jev"])
ap.add_argument("--model", default="jev-latest")    # solo --engine jev
ap.add_argument("--label", default="")
A = ap.parse_args()

INSTR = "This SMS message is spam, a scam, or unsolicited marketing."


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
    random.Random(42).shuffle(rows)       # misma semilla que eval2.py => mismo split
    return rows


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


def fit_platt(ys, zs):
    a, b, lr, n = 1.0, 0.0, 0.05, len(ys)
    for _ in range(3000):
        ga = gb = 0.0
        for y, z in zip(ys, zs):
            d = sigmoid(a * z + b) - y
            ga += d * z; gb += d
        a -= lr * ga / n; b -= lr * gb / n
    return a, b


def main():
    t_load = time.time()
    if A.engine == "jev":                 # misma interfaz system_one, via API
        import jev_client
        agent = jev_client.Agent(model=A.model)
    else:
        import laya
        agent = laya.Agent(device=A.device)
    load_s = time.time() - t_load

    rows = load(A.data)
    test = rows[:A.n]
    calib = rows[-A.calibrate:] if A.calibrate else []

    print("\n### %s  (test n=%d, spam=%d) device=%s"
          % (A.label or A.engine, len(test), sum(y for y, _ in test), A.device))
    print("  carga modelo   %.1fs" % load_s)

    def p_spam(msg):
        r = agent.system_one(msg[:2000], {"is_spam": {"type": "noul", "instructions": INSTR}})
        ans = r.get("answers", {}).get("is_spam", {})
        p = ans.get("noul")
        return None if p is None else min(max(float(p), 1e-6), 1 - 1e-6)

    t0 = time.time()
    ys, ps = [], []
    for y, m in test:
        p = p_spam(m)
        if p is None:
            continue
        ys.append(y); ps.append(p)
    el = time.time() - t0
    print("  velocidad      %.1f msg/s (%.1fs)" % (len(ys) / el, el))
    report("sin calibrar", ys, ps)

    if calib:
        cys, czs = [], []
        for y, m in calib:
            p = p_spam(m)
            if p is None:
                continue
            cys.append(y); czs.append(math.log(p / (1 - p)))
        if cys:
            a, b = fit_platt(cys, czs)
            print("  Platt          a=%.3f b=%.3f" % (a, b))
            zs = [math.log(p / (1 - p)) for p in ps]
            report("platt", ys, [sigmoid(a * z + b) for z in zs])


main()
