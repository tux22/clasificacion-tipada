#!/usr/bin/env bash
# Prueba 2: las mismas corridas multiclase que la comparación original, pero con cada categoría
# DESCRITA (descripciones/*.json) en vez de solo su nombre. Mismo split, semillas y métricas.
#
# Uso:  ./run_descripciones.sh jev|laya [dir_logs]
#   jev : necesita TYPESAFE_API_KEY en el entorno. ~5,800 llamadas a ~1 req/s.
#   laya: corre con $PY (por defecto python3) donde esté instalado `laya`; DEVICE=cpu|cuda.
#
# Empieza con un CONTROL sin describir que debe repetir lo publicado: Jev, AG News semilla 42 =
# 0.8720 (log de 2026-09-26); Laya, las 5 semillas = 0.9260 +- 0.0058 (solo se guardó la media).
# Si no lo repite, el entorno cambió y lo demás no compara.
# REANUDABLE: un bloque cuyo log ya tiene resultado no se repite.
set -euo pipefail
ENGINE=${1:?uso: run_descripciones.sh jev|laya [dir_logs]}
OUT=$(realpath -m "${2:-logs-descripciones-$ENGINE}")
cd "$(dirname "$0")"
export EVAL_DIR=${EVAL_DIR:-$(cd .. && pwd)/data}
DESC=$(cd .. && pwd)/descripciones
PY=${PY:-python3}
EXTRA=()
if [ "$ENGINE" = jev ]; then
  : "${TYPESAFE_API_KEY:?falta TYPESAFE_API_KEY}"
  export JEV_PAUSA=${JEV_PAUSA:-1.0}
  DESCANSO=${DESCANSO:-60}
  curl -sS --fail-with-body "${TYPESAFE_BASE_URL:-https://api.typesafe.ai}/v1/systemone" \
    -A clasificacion-tipada-benchmark/1.0 -H "Authorization: Bearer $TYPESAFE_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"state":"WINNER! Claim your free prize now, text CLAIM to 80082","model":"jev-latest",
         "questions":{"is_spam":{"type":"noul",
         "instructions":"This SMS message is spam, a scam, or unsolicited marketing."}}}' \
    | tee "$OUT.humo.json"; echo
else
  EXTRA=(--device "${DEVICE:-cpu}")
  DESCANSO=${DESCANSO:-0}
fi
mkdir -p "$OUT"

hecho() { [ -f "$1" ] && grep -q "seed=" "$1"; }

# bloque <prefijo> <semillas> <etiqueta> <args de eval_multi...>
bloque() {
  local pre=$1 seeds=$2 label=$3; shift 3
  for s in $seeds; do
    local log="$OUT/$pre-s$s.log"
    if hecho "$log"; then echo "$pre semilla $s ya hecha, se salta"; continue; fi
    "$PY" -u eval_multi.py --engine "$ENGINE" "${EXTRA[@]}" "$@" --seeds "$s" --label "$label" | tee "$log"
    hecho "$log" || { echo "FALLO: $log sin resultado" >&2; exit 1; }
    sleep "$DESCANSO"
  done
  "$PY" - "$OUT/$pre" <<'EOF' | tee "$OUT/$pre-resumen.log"
import glob, re, statistics as st, sys
rows = []
for f in sorted(glob.glob(sys.argv[1] + "-s*.log")):
    m = re.search(r"seed=\s*(\d+)\s+acc=([\d.]+) f1=([\d.]+) top3=([\d.]+) ece=([\d.]+)"
                  r" brier=([\d.]+) n=(\d+)\s+([\d.]+) it/s", open(f).read())
    if m:
        rows.append([float(x) for x in m.groups()])
print("  --- %s: %d semillas, n=%s ---" % (sys.argv[1], len(rows), [int(r[6]) for r in rows]))
for i, name in ((1, "accuracy"), (2, "macro-F1"), (3, "top-3 acc"), (4, "ECE"),
                (5, "Brier"), (7, "item/s")):
    v = [r[i] for r in rows]
    print("  %-10s %.4f +- %.4f" % (name, st.mean(v), st.stdev(v) if len(v) > 1 else 0.0))
EOF
}

S5="42 7 123 2024 555"
L=$( [ "$ENGINE" = jev ] && echo Jev || echo Laya )
CTRL=$( [ "$ENGINE" = jev ] && echo 42 || echo "$S5" )
bloque 00-control-agnews "$CTRL" "$L · AG News · sin describir (control)" --dataset agnews --n 500
bloque 02-agnews "$S5" "$L · AG News · descrito" --dataset agnews --n 500 --criteria "$DESC/agnews.json"
bloque 03-banking40 "$S5" "$L · Banking77-40 · descrito" --dataset banking77 --max-classes 40 --n 500 --criteria "$DESC/banking77.json"
bloque 04-banking77 "42" "$L · Banking77-77 · descrito" --dataset banking77 --n 300 --criteria "$DESC/banking77.json"
# Base del MISMO día sin describir para Banking77: Jev cambió desde septiembre (control AG News
# s42: 0.8720 -> 0.8760), así que el efecto de describir se mide contra una base de hoy.
bloque 05-base-banking40 "$S5" "$L · Banking77-40 · sin describir (base de hoy)" --dataset banking77 --max-classes 40 --n 500
bloque 06-base-banking77 "42" "$L · Banking77-77 · sin describir (base de hoy)" --dataset banking77 --n 300
# Laya con más presupuesto para las opciones (HML=512 ML=1024): con 192, Banking77 recorta cada
# opción a 3 tokens. Solo Banking77; AG News cabe completo con el default.
if [ "$ENGINE" = laya ] && [ -n "${HML:-}" ]; then
  X=(--head-max-len "$HML" --max-len "${ML:-1024}")
  bloque 07-hml$HML-base-banking40 "$S5" "$L · Banking77-40 · sin describir · hml=$HML" --dataset banking77 --max-classes 40 --n 500 "${X[@]}"
  bloque 08-hml$HML-desc-banking40 "$S5" "$L · Banking77-40 · descrito · hml=$HML" --dataset banking77 --max-classes 40 --n 500 --criteria "$DESC/banking77.json" "${X[@]}"
  bloque 09-hml$HML-base-banking77 "42" "$L · Banking77-77 · sin describir · hml=$HML" --dataset banking77 --n 300 "${X[@]}"
  bloque 10-hml$HML-desc-banking77 "42" "$L · Banking77-77 · descrito · hml=$HML" --dataset banking77 --n 300 --criteria "$DESC/banking77.json" "${X[@]}"
fi
echo "listo: $OUT"
