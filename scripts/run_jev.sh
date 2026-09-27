#!/usr/bin/env bash
# Set de pruebas de Jev: las MISMAS corridas que se hicieron con Laya (02 y 03), mismo split,
# mismas instrucciones y mismas metricas, para poner a Jev en las tablas junto a Laya.
#
# Uso:  TYPESAFE_API_KEY=... ./run_jev.sh [dir_logs]
# Datos: EVAL_DIR (por defecto data/ junto a scripts/) con SMSSpamCollection, agnews.csv y
# banking77.csv; se bajan con get_data.sh.
#
# Ritmo conservador: la primera corrida (~4 req/s sin pausa) acabo en 402 y bloqueo de IP tras
# ~1,190 llamadas. Por eso: 1 req/s, un proceso por (dataset, semilla), descanso entre bloques
# y REANUDABLE: un bloque cuyo log ya tiene su resultado no se repite. Si algo falla
# (401/402/403 abortan en jev_client.py), se relanza igual y sigue donde quedo.
set -euo pipefail
: "${TYPESAFE_API_KEY:?falta TYPESAFE_API_KEY}"
OUT=$(realpath -m "${1:-logs-jev-$(date +%F)}")   # relativo al directorio desde donde se llama
cd "$(dirname "$0")"
export EVAL_DIR=${EVAL_DIR:-$(cd .. && pwd)/data}
export JEV_PAUSA=${JEV_PAUSA:-1.0}
DESCANSO=${DESCANSO:-60}          # s entre bloques
mkdir -p "$OUT"

# 0. Humo: una peticion. Falla rapido si la key, la red o el saldo no sirven.
curl -sS --fail-with-body "${TYPESAFE_BASE_URL:-https://api.typesafe.ai}/v1/systemone" \
  -A clasificacion-tipada-benchmark/1.0 -H "Authorization: Bearer $TYPESAFE_API_KEY" -H "Content-Type: application/json" \
  -d '{"state":"WINNER! Claim your free prize now, text CLAIM to 80082","model":"jev-latest",
       "questions":{"is_spam":{"type":"noul",
       "instructions":"This SMS message is spam, a scam, or unsolicited marketing."}}}' \
  | tee "$OUT/00-humo.json"
echo

hecho() { [ -f "$1" ] && grep -q "$2" "$1"; }

# 1. Binario (spam): n=400 test + 200 calibracion, semilla 42. Sin calibrar y con Platt.
if hecho "$OUT/01-spam.log" "platt "; then echo "01-spam ya hecho, se salta"; else
  python3 -u eval_laya.py --engine jev --data "$EVAL_DIR/SMSSpamCollection" \
    --label "Jev · spam" | tee "$OUT/01-spam.log"
  sleep "$DESCANSO"
fi

# bloque <prefijo> <semillas> <etiqueta> <args de eval_multi...>: una semilla por proceso,
# luego media +- desviacion como la tabla de Laya en 03.
bloque() {
  local pre=$1 seeds=$2 label=$3; shift 3
  for s in $seeds; do
    local log="$OUT/$pre-s$s.log"
    if hecho "$log" "seed="; then echo "$pre semilla $s ya hecha, se salta"; continue; fi
    python3 -u eval_multi.py --engine jev "$@" --seeds "$s" --label "$label" | tee "$log"
    hecho "$log" "seed=" || { echo "FALLO: $log sin resultado" >&2; exit 1; }
    sleep "$DESCANSO"
  done
  python3 - "$OUT/$pre" <<'EOF' | tee "$OUT/$pre-resumen.log"
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
# 2. AG News, 4 clases, 5 semillas x 500, sin calibrar.
bloque 02-agnews "$S5" "Jev · AG News" --dataset agnews --n 500
# 3. Banking77 recortado a 40 clases, 5 semillas x 500.
bloque 03-banking40 "$S5" "Jev · Banking77-40" --dataset banking77 --max-classes 40 --n 500
# 4. Banking77 completo, 77 clases, n=300, solo semilla 42 (la corrida del techo de 52 clases).
bloque 04-banking77 "42" "Jev · Banking77-77" --dataset banking77 --n 300

echo "listo: $OUT"
