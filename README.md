# Clasificación tipada: Jev contra Laya y LLM locales

Scripts para repetir una comparación de clasificadores que devuelven **decisiones con
probabilidades** en vez de texto:

- **Jev** (TypeSafe AI), por su API de pago.
- **Laya 421M**, un encoder abierto (Apache-2.0) con la misma interfaz que Jev.
- **LLM abiertos** servidos con llama.cpp, leyendo las probabilidades del primer token.

Tres datasets públicos, zero-shot, hasta 5 semillas × 500 ejemplos. Los modelos locales
corrieron en una RTX 3060 Ti de 8 GB.

## Resultados

**Spam (UCI SMS Spam), 400 mensajes, calibrado con Platt sobre otros 200:**

| Modelo | AUROC | F1 | ECE |
|---|---|---|---|
| Jev 1.13 (API) | **0.9980** | **0.9474** | **0.0193** |
| Qwen3-4B | 0.9806 | 0.9153 | 0.0484 |
| Laya 421M | 0.9881 | 0.8704 | 0.0500 |
| Qwen2.5-3B | 0.9825 | 0.8485 | 0.0658 |
| Qwen2.5-7B | 0.9455 | 0.7703 | 0.0365 |
| Gemma-3-4B | 0.9654 | 0.0000 | 0.1306 |
| GLiClass 151M | 0.7297 | 0.0000 | 0.0189 |

**Multiclase, sin calibrar. Accuracy, media ± desviación sobre 5 semillas × 500:**

| Tarea | Jev | Laya | Qwen3-4B |
|---|---|---|---|
| AG News, 4 clases | 0.8644 ± 0.0134 | **0.9260 ± 0.0058** | 0.8412 ± 0.0205 |
| Banking77, 40 clases | **0.8612 ± 0.0424** | 0.5652 ± 0.0774 | 0.4376 ± 0.0619 |
| Banking77, 77 clases *(1 × 300)* | **0.8067** | 0.3600 | 0.0833 |

Todas las corridas, con precisión, recall, Brier, ECE y velocidad, están en
[`resultados.csv`](./resultados.csv).

> **Nota sobre Laya en Banking77 (2026-09-30).** Estas cifras son con la configuración de fábrica
> de `laya 0.3.4`, que reparte 192 tokens entre todas las opciones y, si no caben, recorta cada
> una a 3 tokens. En Banking77-77 pasa siempre; en Banking77-40, en 3 de las 5 semillas. Con
> `--head-max-len 512` (lo que recomienda el README de Laya), sin describir las categorías:
> **Banking77-40 0.6060 ± 0.0398** y **Banking77-77 0.4433**. Jev sigue arriba en las dos.

**Lectura corta:** con pocas clases bien distintas, Laya gana y corre en local. Con muchas clases
casi sinónimas, Jev gana por mucho y sigue calibrado. Entre los modelos locales, calibrar con 200
ejemplos mueve más que cambiar de modelo.

## Método

- **Mismo split para todos.** Semilla fija; el test sale del principio de la lista barajada y
  la calibración del final, sin solaparse. Semillas de validación: `42,7,123,2024,555`.
- **Mismas instrucciones para Jev y Laya**, con las categorías sin describir (solo el nombre).
  Es la comparación pareja, no el mejor resultado posible de ninguno.
- **Jev no es determinista.** La misma llamada repetida da 0.97 o 0.98, y la misma corrida de 500
  dio 0.8720 y 0.8760 en días distintos. Para medir un cambio contra Jev hay que correr la base
  el mismo día y con varias semillas; no comparar contra números viejos.
- **Calibración:** Platt scaling, `p = sigmoid(a·z + b)` sobre el logit, ajustado con 200
  ejemplos etiquetados que no están en el test.
- **Métricas:** AUROC (¿separa las clases?), F1, accuracy, macro-F1, top-3, Brier y ECE con 10
  bins (¿su 0.8 acierta el 80 % de las veces?).
- **LLM locales:** se pide un solo token (`n_predict=1`) con los logprobs del top-k y se
  normaliza la masa entre las etiquetas. Con más de 52 clases no alcanzan las letras A-Z/a-z y se
  numera 1..N, lo que degrada mucho el resultado: es un límite del método.

## Cómo repetirlo

Requisitos: Python 3.9+ (los scripts solo usan la librería estándar, salvo Laya y GLiClass),
`curl` y, para los modelos locales, Docker con GPU NVIDIA.

```bash
./get_data.sh        # baja los tres datasets a data/
```

### Jev

Hace falta una API key de TypeSafe en `TYPESAFE_API_KEY`. `run_jev.sh` corre el set completo y
se puede relanzar: salta lo que ya terminó.

```bash
export TYPESAFE_API_KEY=...
./scripts/run_jev.sh logs-jev
```

⚠️ **Va lento a propósito**: ~0.8 llamadas/s, una semilla por proceso y 60 s entre bloques, en
total unas 2 horas y ~5 M tokens. Una primera corrida a ~4 llamadas/s sin pausa terminó en un
bloqueo temporal de la cuenta y de la IP. Se ajusta con `JEV_PAUSA` y `DESCANSO`.

### Laya

Los resultados son con **`laya==0.3.4`**; versiones posteriores cambiaron las temperaturas.

```bash
pip install laya==0.3.4
python3 scripts/eval_laya.py --device cuda                              # spam
python3 scripts/eval_multi.py --engine laya --dataset agnews --n 500 --seeds 42,7,123,2024,555
python3 scripts/eval_multi.py --engine laya --dataset banking77 --max-classes 40 --n 500 --seeds 42,7,123,2024,555
python3 scripts/eval_multi.py --engine laya --dataset banking77 --n 300
# con muchas clases, darle espacio a las opciones para que no se recorten:
python3 scripts/eval_multi.py --engine laya --dataset banking77 --n 300 --head-max-len 512 --max-len 1024
```

### LLM locales con llama.cpp

Todos en cuantización Q4_K_M:

| Modelo | Repositorio en Hugging Face | Archivo | Plantilla |
|---|---|---|---|
| Qwen2.5-3B | `Qwen/Qwen2.5-3B-Instruct-GGUF` | `qwen2.5-3b-instruct-q4_k_m.gguf` | `qwen` |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B-Instruct-GGUF` | `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf` (y la parte 2) | `qwen` |
| Qwen3-4B | `unsloth/Qwen3-4B-Instruct-2507-GGUF` | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | `qwen3-nothink` |
| Gemma-3-4B | `unsloth/gemma-3-4b-it-GGUF` | `gemma-3-4b-it-Q4_K_M.gguf` | `gemma` |

```bash
docker run -d --gpus all -p 8080:8080 -v "$PWD/models:/models" \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/qwen2.5-3b-instruct-q4_k_m.gguf -ngl 99 -c 4096 --host 0.0.0.0 --port 8080

U=http://127.0.0.1:8080/completion
python3 scripts/eval2.py --url $U --template qwen --calibrate 200            # spam, sin calibrar y con Platt
python3 scripts/eval2.py --url $U --template qwen --calibrate 200 --fewshot 8
python3 scripts/eval_multi.py --engine qwen --url $U --template qwen3-nothink \
  --dataset agnews --n 500 --seeds 42,7,123,2024,555
```

Con un GGUF partido basta apuntar al primer archivo. Qwen3 necesita la plantilla
`qwen3-nothink`, que cierra el bloque de razonamiento; sin ella el primer token es `<think>` y la
lectura se rompe. Gemma no tiene rol `system`.

### GLiClass (descartado)

```bash
pip install gliclass       # requiere Python 3.10+
python3 scripts/eval_gli.py --device cpu
```

## Prueba 2: describir las categorías

La documentación de Jev recomienda describir cada opción de `choice`. Mismo split, semillas y
métricas que arriba, con una línea de descripción por categoría
([`descripciones/`](./descripciones/)), escrita solo a partir del nombre, sin mirar el test.
El efecto se mide contra una base sin describir **corrida el mismo día**.

| Tarea | Jev sin describir | Jev descrito | Laya sin describir | Laya descrito |
|---|---|---|---|---|
| AG News, 4 clases | 0.8720 / 0.8760 *(s42)* | 0.8708 ± 0.0093 | 0.9268 ± 0.0061 | 0.9136 ± 0.0155 |
| Banking77, 40 clases | 0.8612 ± 0.0397 | **0.8756 ± 0.0339** | 0.6060 ± 0.0398 | **0.1672 ± 0.0325** |
| Banking77, 77 clases *(1 × 300)* | 0.8033 | 0.8300 | 0.4433 | 0.2733 |

Laya en Banking77 con `--head-max-len 512 --max-len 1024`; con descripciones sus opciones aún se
recortan (11 tokens de texto por opción en 40 clases, 5 en 77). En AG News caben completas con
la configuración de fábrica.

- **Jev:** sin cambio en noticias; en 40 clases sube en 3 semillas y queda igual en 2. Duplica
  los tokens de entrada por llamada.
- **Laya:** baja en noticias (−1.3, y su ECE pasa de 0.022 a 0.199) y se desploma con 40 clases.

```bash
./scripts/run_descripciones.sh jev                              # requiere TYPESAFE_API_KEY
PY=python3 DEVICE=cuda HML=512 ./scripts/run_descripciones.sh laya
```

## Scripts

| Script | Para qué |
|---|---|
| `eval_spam.py` | Binario con un GGUF, métricas base |
| `eval2.py` | Binario con un GGUF, con few-shot, temperature, Platt y plantillas |
| `eval_laya.py` | Binario con Laya o Jev (`--engine jev`), primitiva `noul` |
| `eval_multi.py` | Multiclase con Qwen, Laya o Jev, multi-semilla |
| `eval_gli.py` | GLiClass |
| `jev_client.py` | Cliente de la API de Jev con la interfaz de Laya, solo librería estándar |
| `run_jev.sh` | Set completo de Jev, reanudable |
| `run_descripciones.sh` | Prueba 2: con y sin descripciones (`--criteria`), con control y base del día |

## Límites

- Tres datasets, zero-shot. No generaliza a clasificación de texto en general.
- Comparación principal con categorías sin describir. Describirlas se midió aparte (Prueba 2).
- Multiclase sin calibrar para los dos modelos. El README de Laya recomienda ajustar una
  temperatura por tarea; las cifras de ECE de Laya son de fábrica.
- Un SVM entrenado sobre el dataset de spam saca 0.976 de accuracy: les gana a todos los modelos
  locales. Si hay datos etiquetados y la tarea es estable, un clasificador clásico sigue siendo
  lo más barato.
- Una sola GPU; las cifras de velocidad no trasladan a otro hardware. Las de Jev dependen de la
  red y del ritmo impuesto.
- Una versión de Jev (`jev-1.13.0`); `jev-latest` cambia con el tiempo.

## Enlaces

- Post con la primera comparación, Laya contra LLM locales: https://lnkd.in/p/gunhvGJs
- Post con la comparación Jev contra Laya: https://lnkd.in/p/gyeS-jrk
- Jev: https://docs.typesafe.ai/introduction
- Laya: https://huggingface.co/convaiinnovations/laya
- llama.cpp: https://github.com/ggml-org/llama.cpp
- UCI SMS Spam: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- AG News: https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv
- Banking77: https://github.com/PolyAI-LDN/task-specific-datasets

## Licencia

El código es MIT. Los modelos y los datasets tienen sus propias licencias.
