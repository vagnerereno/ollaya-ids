# Ollaya IDS

Este repositório compara classificadores tabulares e modelos de decisão System One na detecção de ataques IEC 61850. O primeiro experimento usa o ERENO 2.0 e mantém as divisões, métricas e representações de entrada sob controle comum.

## Modelos avaliados

- Random Forest;
- XGBoost em CPU ou CUDA;
- Kev executado pelo Ollaya, sem adaptação ao ERENO;
- Kev com fine-tuning LoRA no conjunto de treino do ERENO.

O modelo ajustado é treinado e servido pelo projeto Kev. O Ollaya continua sendo usado para a execução do checkpoint público sem adaptação. Os dois expõem a API `/v1/systemone`, portanto passam pelo mesmo cliente e pelo mesmo código de avaliação.

## Métricas

O relatório de cada execução contém:

- acurácia e acurácia balanceada;
- precisão, recall e F1 macro;
- F1 ponderado e métricas por classe;
- matriz de confusão;
- log loss, Brier multiclasse e Expected Calibration Error;
- ROC-AUC, PR-AUC, falso-positivo e falso-negativo para a decisão binária intrusão/normal;
- latência média e vazão de inferência.

## Dataset

O arquivo `data/ERENO-2.0-100K.csv` possui duas seções de dados. A primeira é CSV; a segunda aparece depois de um cabeçalho ARFF no mesmo arquivo. Cada seção contém aproximadamente 100 mil amostras e elas não são cópias uma da outra.

O CSV é versionado com Git LFS. Depois de clonar o repositório em uma máquina nova, confirme que o arquivo foi materializado:

```bash
git lfs install
git lfs pull
```

A configuração padrão usa apenas a primeira seção:

```yaml
dataset:
  section: csv
```

As opções aceitas são `csv`, `arff` e `all`. A escolha fica registrada em `artifacts/dataset_profile.json` junto com a contagem de registros e o hash do conteúdo limpo.

Antes da divisão, o carregador remove duplicatas exatas e descarta vetores de atributos que aparecem com rótulos conflitantes. Isso evita que a mesma observação apareça em treino e teste com o mesmo rótulo ou com rótulos incompatíveis.

O perfil `decision` usa 14 atributos de protocolo e tempo:

```text
SqNum, StNum, cbStatus, goID, timestampDiff, tDiff,
timeFromLastChange, delay, stDiff, sqDiff, gooseLengthDiff,
cbStatusDiff, apduSizeDiff, frameLengthDiff
```

Esse perfil mantém a entrada do modelo de decisão curta e permite uma comparação direta com os classificadores tabulares. O perfil `all` usa todos os atributos, exceto `Time`, `class` e o identificador interno.

## Instalação

Para preparar os dados e executar Random Forest:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

No Windows, a ativação do ambiente é:

```powershell
.\.venv\Scripts\Activate.ps1
```

Para incluir XGBoost e SHAP:

```bash
pip install -e ".[dev,xgboost,explain]"
```

## Preparação

```bash
ollaya-ids inspect
ollaya-ids prepare
```

Os arquivos gerados ficam em `artifacts/` e não são versionados. A divisão padrão é estratificada em 70% para treino, 15% para validação e 15% para teste.

Uma avaliação temporal também está disponível:

```bash
ollaya-ids prepare --strategy temporal
```

No ERENO, as classes não estão distribuídas uniformemente ao longo do tempo. A divisão temporal deve ser tratada como teste de mudança de distribuição, e não como substituta da avaliação estratificada.

## Baselines tabulares

Random Forest:

```bash
ollaya-ids baseline --model random-forest
```

XGBoost com CUDA:

```bash
ollaya-ids baseline --model xgboost --device cuda
```

Para uma verificação local sem GPU:

```bash
ollaya-ids baseline --model xgboost --device cpu
```

## Kev sem fine-tuning no Ollaya

Instale o Ollaya, carregue o modelo e mantenha o servidor em execução:

```bash
ollaya pull kev:4b
ollaya serve
```

Em outro terminal:

```bash
ollaya-ids evaluate-api \
  --model kev:4b \
  --base-url http://127.0.0.1:11435 \
  --run-name kev-zero-shot
```

Por padrão, essa etapa usa uma amostra estratificada de 2.000 registros do teste. O limite pode ser alterado em `configs/experiment.yaml` ou removido com `--max-samples 0`.

## Fine-tuning do Kev

Primeiro, exporte os registros no formato de treinamento do Kev:

```bash
ollaya-ids export-kev
```

Isso cria:

```text
artifacts/kev/train.jsonl
artifacts/kev/validation.jsonl
artifacts/kev/test.jsonl
```

No servidor de GPU, use Python 3.12 ou 3.13 e instale o Kev em um ambiente separado:

```bash
git clone https://github.com/jaredpalmer/kev.git external/kev
cd external/kev
uv sync --extra serve --python 3.13
```

O primeiro experimento parte do checkpoint público, preservando o conhecimento já adquirido pelo modelo:

```bash
uv run python -m kev.train \
  --data ../../artifacts/kev/train.jsonl \
  --base Qwen/Qwen3.5-4B-Base \
  --init_from jaredpalmer/kev-4b@v1.0 \
  --epochs 2 \
  --lr 2e-5 \
  --batch 1 \
  --accum 8 \
  --dtype bf16 \
  --weights_dtype bf16 \
  --checkpointing 1 \
  --device cuda \
  --out ../../runs/ereno-kev-4b
```

Avalie o checkpoint na validação antes de consultar o teste:

```bash
uv run python -m kev.benchmark \
  --run ../../runs/ereno-kev-4b \
  --data ../../artifacts/kev/validation.jsonl \
  --out ../../runs/ereno-kev-4b-validation
```

Depois de escolher a configuração final, inicie o endpoint local:

```bash
uv run --extra serve python -m kev.serve \
  --run ../../runs/ereno-kev-4b \
  --device cuda \
  --port 8009
```

Na raiz deste repositório, execute a avaliação comum:

```bash
ollaya-ids evaluate-api \
  --base-url http://127.0.0.1:8009 \
  --endpoint /v1/systemone \
  --model kev-latest \
  --run-name kev-ereno-finetuned
```

## Comparação

Depois de executar os modelos desejados:

```bash
ollaya-ids compare
```

O resumo é gravado em `artifacts/results/comparison.csv`. Cada pasta de execução também contém `metrics.json`, `per_class.csv`, `confusion_matrix.csv` e `predictions.csv`.

## Execução na H200 compartilhada

O repositório inclui uma imagem baseada em `nvidia/cuda:13.0.2-devel-ubuntu24.04`. O container usa apenas a GPU escolhida no `docker run`; dentro dele, essa GPU aparece como `cuda:0` mesmo que no host ela seja a GPU 1, 2 ou 3.

Depois de clonar o repositório no servidor, confirme qual GPU está livre:

```bash
uso_das_gpus
```

Construa a imagem:

```bash
bash scripts/h200/build_image.sh
```

Inicie o container passando o índice livre. O exemplo abaixo usa `3` apenas como ilustração:

```bash
bash scripts/h200/start_container.sh 3
```

O script monta quatro áreas:

- o repositório atual em `/workspace`;
- `~/resultados` em `/resultados`;
- `~/.cache/ollaya-ids` para pesos do Hugging Face e cache do `uv`;
- `~/.ollaya` para os modelos baixados pelo Ollaya.

Os caches evitam baixar novamente os pesos do Qwen e do Kev quando o container for recriado. O `--shm-size` é configurado em 16 GB para não limitar os processos de treinamento.

Valide CUDA, XGBoost, Ollaya e Kev antes do experimento:

```bash
docker exec -it ollaya-ids-h200-3 bash scripts/h200/verify_environment.sh
```

As etapas podem ser executadas individualmente:

```bash
docker exec -it ollaya-ids-h200-3 bash scripts/h200/run_baselines.sh
docker exec -it ollaya-ids-h200-3 bash scripts/h200/run_ollaya_zero_shot.sh
docker exec -it ollaya-ids-h200-3 bash scripts/h200/train_kev.sh
docker exec -it ollaya-ids-h200-3 bash scripts/h200/evaluate_finetuned_kev.sh
```

Para executar o fluxo completo em background e poder encerrar a sessão SSH:

```bash
docker exec -d ollaya-ids-h200-3 bash -lc \
  'bash scripts/h200/run_all.sh > /resultados/ollaya-ids/logs/run-all.log 2>&1'
```

Acompanhe o log pelo host:

```bash
tail -f ~/resultados/ollaya-ids/logs/run-all.log
```

O teste rápido usa 2.000 registros tanto no Ollaya sem ajuste quanto no Kev ajustado. Para a leitura final de todo o teste:

```bash
docker exec -d \
  -e MAX_TEST_SAMPLES=0 \
  ollaya-ids-h200-3 \
  bash -lc 'bash scripts/h200/run_all.sh > /resultados/ollaya-ids/logs/run-all-full.log 2>&1'
```

Todos os resultados, checkpoints e logs permanecem em `~/resultados/ollaya-ids` depois que o container for encerrado.

## Cuidados experimentais

- Não use o conjunto de teste para escolher épocas, taxa de aprendizado ou limiar.
- Ajuste a temperatura de calibração apenas com a validação.
- Compare o checkpoint ajustado com o checkpoint original na mesma amostra de teste.
- Registre modelo-base, revisão, configuração CUDA e versão do dataset.
- Para estudar generalização a outra subestação, mantenha todos os registros daquela subestação fora do treinamento.

## Referências

- [ERENO: A Framework for Generating Realistic IEC–61850 Intrusion Detection Datasets for Smart Grids](https://doi.org/10.1109/TDSC.2023.3336857)
- [Ollaya](https://ollaya.dev/docs)
- [Kev](https://github.com/jaredpalmer/kev)
