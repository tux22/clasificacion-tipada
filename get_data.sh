#!/usr/bin/env bash
# Descarga los tres datasets públicos en data/, con el formato que esperan los scripts.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data && cd data

# UCI SMS Spam (5574 mensajes, 2 clases)
curl -fsSL -o smsspam.zip https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
python3 -c "import zipfile; zipfile.ZipFile('smsspam.zip').extract('SMSSpamCollection')"
rm smsspam.zip

# AG News, split de test (7600 noticias, 4 clases)
curl -fsSL -o agnews.csv \
  https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv

# Banking77, split de test (3080 consultas, 77 intenciones)
curl -fsSL -o banking77.csv \
  https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv

wc -l SMSSpamCollection agnews.csv banking77.csv
