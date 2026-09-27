#!/usr/bin/env python3
"""
Evaluacion MULTICLASE comparando los dos finalistas en la misma tarea y split:

  --engine qwen : llama.cpp, logprobs del primer token sobre letras A/B/C/... (n_predict=1)
  --engine laya : primitiva `choice` de Laya, una opcion por clase
  --engine jev  : la misma pregunta `choice` contra la API de Jev (jev_client.py)

Datasets: agnews (4 clases) | banking77 (77 clases).

Metricas multiclase:
  accuracy, macro-F1  -> desempeno
  ECE, Brier          -> calibracion sobre la probabilidad de la clase predicha
  top-3 accuracy      -> si la respuesta correcta esta entre las 3 mas probables
"""
import argparse, csv, json, math, os, random, sys, time, urllib.request, warnings
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=["agnews", "banking77"])
ap.add_argument("--engine", required=True, choices=["qwen", "laya", "jev"])
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--url", default="http://127.0.0.1:8084/completion")
ap.add_argument("--device", default="cuda")
ap.add_argument("--model", default="jev-latest")    # solo --engine jev
ap.add_argument("--template", default="qwen3-nothink")
ap.add_argument("--label", default="")
ap.add_argument("--max-classes", type=int, default=0,
                help="quedarse con las N clases mas frecuentes (0 = todas)")
ap.add_argument("--seeds", default="42",
                help="semillas separadas por coma; se reporta media +- desviacion")
A = ap.parse_args()

DATA = os.environ.get("EVAL_DIR", "data")   # agnews.csv y banking77.csv
SEED = 42
AGNEWS = ["World", "Sports", "Business", "Science and Technology"]
# A-Z, luego a-z: hasta 52 clases con una sola letra; mas alla se trunca el set
LETTERS = [chr(65 + i) for i in range(26)] + [chr(97 + i) for i in range(26)]


def load_agnews(n):
    rows = []
    with open("%s/agnews.csv" % DATA, encoding="utf-8", errors="replace") as fh:
        for r in csv.reader(fh):
            if len(r) >= 3:
                rows.append((int(r[0]) - 1, (r[1] + ". " + r[2]).strip()))
    random.Random(SEED).shuffle(rows)
    return rows[:n], AGNEWS


def load_banking77(n):
    rows, labels = [], []
    with open("%s/banking77.csv" % DATA, encoding="utf-8", errors="replace") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            lab = (r.get("category") or "").strip()
            txt = (r.get("text") or "").strip()
            if lab and txt:
                if lab not in labels:
                    labels.append(lab)
                rows.append((lab, txt))
    labels.sort()
    idx = {l: i for i, l in enumerate(labels)}
    rows = [(idx[l], t) for l, t in rows]
    random.Random(SEED).shuffle(rows)
    return rows[:n], [l.replace("_", " ") for l in labels]


def qwen_probs(text, labels, retries=2):
    """P por clase desde los logprobs del PRIMER token.

    <=52 clases: una letra por clase, lectura directa.
    >52 clases : numeracion 1..N. El primer token es solo el digito inicial, asi que la
                 masa se reparte entre las clases que comparten prefijo (1 -> 1,10..19).
                 Es una limitacion real del enfoque de un solo token, no un bug.
    """
    use_letters = len(labels) <= len(LETTERS)
    if use_letters:
        tags = [LETTERS[i] for i in range(len(labels))]
    else:
        tags = [str(i + 1) for i in range(len(labels))]
    opts = "\n".join("%s. %s" % (tags[i], l) for i, l in enumerate(labels))
    sys_p = ("You are a text classifier. Answer with exactly one %s: the label that best "
             "fits. Answer only that, nothing else." % ("letter" if use_letters else "number"))
    ask = ("Text:\n%s\n\nOptions:\n%s\n\nAnswer with one %s."
           % (text[:1500], opts, "letter" if use_letters else "number"))
    if A.template == "qwen3-nothink":
        ask += " /no_think"
        prompt = ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n"
                  "<|im_start|>assistant\n<think>\n\n</think>\n\n" % (sys_p, ask))
    else:
        prompt = ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n"
                  "<|im_start|>assistant\n" % (sys_p, ask))

    body = json.dumps({"prompt": prompt, "n_predict": 1, "n_probs": 60,
                       "temperature": 0.0, "cache_prompt": True}).encode()
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
    pr = d.get("completion_probabilities") or []
    if not pr:
        return None
    cands = pr[0].get("probs") or pr[0].get("top_logprobs") or []
    out = [0.0] * len(labels)
    for c in cands:
        tok = (c.get("tok_str") or c.get("token") or "").strip()
        p = c.get("prob")
        if p is None and "logprob" in c:
            p = math.exp(c["logprob"])
        if p is None or not tok:
            continue
        if use_letters:
            if tok in tags:
                out[tags.index(tok)] += p
        else:
            # el token es un prefijo numerico: repartir entre las clases que empiezan asi
            hits = [i for i, t in enumerate(tags) if t.startswith(tok)]
            if hits:
                for i in hits:
                    out[i] += p / len(hits)
    tot = sum(out)
    return [x / tot for x in out] if tot > 0 else None


_agent = None


def laya_probs(text, labels):
    """P por clase con la primitiva `choice` de Laya (o de Jev: misma interfaz)."""
    global _agent
    if _agent is None:
        if A.engine == "jev":
            import jev_client
            _agent = jev_client.Agent(model=A.model)
        else:
            import laya
            _agent = laya.Agent(device=A.device)
    q = {"cls": {"type": "choice",
                 "instructions": "Classify the text into exactly one category.",
                 "criteria": {l: None for l in labels}}}
    try:
        r = _agent.system_one(text[:1500], q)
    except Exception as e:
        print("  fallo: %s" % e, file=sys.stderr)
        return None
    ans = r.get("answers", {}).get("cls", {})
    probs = ans.get("probabilities") or ans.get("probs") or {}
    if isinstance(probs, dict) and probs:
        out = [float(probs.get(l, 0.0)) for l in labels]
    elif isinstance(probs, list) and len(probs) == len(labels):
        out = [float(x) for x in probs]
    else:                                  # solo vino la etiqueta ganadora
        ch = ans.get("choice")
        if ch is None:
            return None
        out = [1.0 if l == ch else 0.0 for l in labels]
    tot = sum(out)
    return [x / tot for x in out] if tot > 0 else None


def macro_f1(ys, preds, k):
    tot = 0.0
    for c in range(k):
        tp = sum(1 for y, p in zip(ys, preds) if y == c and p == c)
        fp = sum(1 for y, p in zip(ys, preds) if y != c and p == c)
        fn = sum(1 for y, p in zip(ys, preds) if y == c and p != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        tot += (2 * pr * rc / (pr + rc)) if pr + rc else 0.0
    return tot / k


def ece_multi(correct, conf, bins=10):
    tot = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        sel = [(c, p) for c, p in zip(correct, conf) if lo < p <= hi or (i == 0 and p <= 0)]
        if not sel:
            continue
        tot += len(sel) / len(conf) * abs(sum(p for _, p in sel) / len(sel)
                                          - sum(c for c, _ in sel) / len(sel))
    return tot


def run_once():
    # se cargan de mas para poder filtrar clases y aun asi llegar a --n
    load_n = A.n * 4 if A.max_classes else A.n
    rows, labels = (load_agnews if A.dataset == "agnews" else load_banking77)(load_n)

    if A.max_classes and len(labels) > A.max_classes:
        from collections import Counter
        keep = [c for c, _ in Counter(y for y, _ in rows).most_common(A.max_classes)]
        remap = {c: i for i, c in enumerate(sorted(keep))}
        labels = [labels[c] for c in sorted(keep)]
        rows = [(remap[y], t) for y, t in rows if y in remap][:A.n]
    else:
        rows = rows[:A.n]
    if A.engine == "qwen" and len(labels) > len(LETTERS):
        print("  NOTA: %d clases > %d letras -> se numeran 1..N; el primer token solo da el"
              " digito inicial, asi que la resolucion es limitada (ver informe)."
              % (len(labels), len(LETTERS)))
    print("\n### %s | %s | %d clases | n=%d"
          % (A.label or A.engine, A.dataset, len(labels), len(rows)))

    fn = qwen_probs if A.engine == "qwen" else laya_probs
    t0 = time.time()
    ys, preds, conf, top3 = [], [], [], []
    for y, txt in rows:
        p = fn(txt, labels)
        if p is None:
            continue
        best = max(range(len(p)), key=lambda i: p[i])
        order = sorted(range(len(p)), key=lambda i: p[i], reverse=True)[:3]
        ys.append(y); preds.append(best); conf.append(p[best]); top3.append(y in order)
        if len(ys) % 100 == 0:           # progreso: una API que se frena se ve aqui
            print("    %d/%d  %.2f it/s" % (len(ys), len(rows), len(ys) / (time.time() - t0)))
    el = time.time() - t0
    if not ys:
        print("  sin resultados")
        return

    acc = sum(1 for y, p in zip(ys, preds) if y == p) / len(ys)
    correct = [1 if y == p else 0 for y, p in zip(ys, preds)]
    brier = sum((c - p) ** 2 for c, p in zip(correct, conf)) / len(conf)
    m = {"acc": acc, "f1": macro_f1(ys, preds, len(labels)),
         "top3": sum(top3) / len(top3), "brier": brier,
         "ece": ece_multi(correct, conf), "conf": sum(conf) / len(conf),
         "ips": len(ys) / el, "k": len(labels)}
    print("  seed=%-4d acc=%.4f f1=%.4f top3=%.4f ece=%.4f brier=%.4f n=%d %.2f it/s"
          % (SEED, m["acc"], m["f1"], m["top3"], m["ece"], m["brier"], len(ys), m["ips"]))
    return m


def mean_sd(vals):
    n = len(vals)
    mu = sum(vals) / n
    if n < 2:
        return mu, 0.0
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    return mu, math.sqrt(var)


seeds = [int(x) for x in A.seeds.split(",") if x.strip()]
print("\n### %s | %s | n=%d | semillas: %s"
      % (A.label or A.engine, A.dataset, A.n, seeds))
res = []
for sd in seeds:
    SEED = sd
    r = run_once()
    if r:
        res.append(r)

if len(res) > 1:
    print("  --- media +- sd sobre %d semillas (%d clases) ---" % (len(res), res[0]["k"]))
    for key, name in (("acc", "accuracy"), ("f1", "macro-F1"), ("top3", "top-3 acc"),
                      ("ece", "ECE"), ("brier", "Brier"), ("ips", "item/s")):
        mu, sd_ = mean_sd([r[key] for r in res])
        print("  %-12s %.4f +- %.4f" % (name, mu, sd_))
