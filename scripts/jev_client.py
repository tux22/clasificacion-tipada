#!/usr/bin/env python3
"""
Cliente minimo de la API de Jev (TypeSafe) con la MISMA interfaz que laya.Agent:

    Agent().system_one(state, questions) -> {"model", "answers", "usage"}

Asi eval_laya.py y eval_multi.py evaluan Jev con el mismo prompt, split y metricas que Laya.
Solo stdlib (urllib), como el resto de los scripts. Lee TYPESAFE_API_KEY del entorno.
Al salir imprime la version que respondio, las llamadas y los tokens consumidos.
"""
import atexit, json, os, sys, time, urllib.error, urllib.request

# TYPESAFE_BASE_URL es la variable documentada del SDK oficial (default https://api.typesafe.ai)
URL = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai").rstrip("/") + "/v1/systemone"
UA = "clasificacion-tipada-benchmark/1.0"   # sin esto urllib manda Python-urllib, que los WAF marcan
PAUSA = float(os.environ.get("JEV_PAUSA", "0.25"))   # s entre llamadas: ~2.5 req/s, lejos de 20/s
USD_POR_MTOK_IN = 0.042          # precio publicado de entrada; la salida no se cobra aparte
REINTENTABLES = (429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529)   # 52x: Cloudflare


class Agent:
    def __init__(self, model="jev-latest", retries=6, timeout=60):
        self.key = os.environ.get("TYPESAFE_API_KEY")
        if not self.key:
            sys.exit("falta TYPESAFE_API_KEY en el entorno")
        self.model, self.retries, self.timeout = model, retries, timeout
        self.served = set()      # versiones reales que respondieron (p.ej. jev-1.13.0)
        self.calls = self.retried = self.tok_in = self.tok_out = 0
        atexit.register(self._resumen)

    def system_one(self, state, questions):
        body = json.dumps({"state": state, "model": self.model,
                           "questions": questions}).encode()
        time.sleep(PAUSA)
        for att in range(self.retries + 1):
            req = urllib.request.Request(URL, data=body, headers={
                "Authorization": "Bearer " + self.key, "Content-Type": "application/json",
                "User-Agent": UA})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d = json.load(r)
                break
            except urllib.error.HTTPError as e:
                detalle = e.read()[:300].decode(errors="replace")
                if e.code in (401, 402, 403):   # key o saldo: abortar todo, no saltar muestras
                    sys.exit("Jev HTTP %d, se aborta la corrida: %s" % (e.code, detalle))
                if e.code not in REINTENTABLES or att == self.retries:
                    raise RuntimeError("Jev HTTP %d: %s" % (e.code, detalle))
                wait = float(e.headers.get("retry-after") or 2 ** att)
            except (urllib.error.URLError, TimeoutError) as e:
                if att == self.retries:
                    raise RuntimeError("Jev sin respuesta: %s" % e)
                wait = 2 ** att
            self.retried += 1
            time.sleep(wait)
        self.calls += 1
        self.served.add(d.get("model", "?"))
        u = d.get("usage") or {}
        self.tok_in += u.get("input_tokens", 0)
        self.tok_out += u.get("output_tokens", 0)
        return d

    def _resumen(self):
        if not self.calls:
            return
        print("  jev            modelo=%s llamadas=%d reintentos=%d tokens in=%d out=%d (~$%.4f)"
              % (",".join(sorted(self.served)), self.calls, self.retried, self.tok_in,
                 self.tok_out, self.tok_in / 1e6 * USD_POR_MTOK_IN))
