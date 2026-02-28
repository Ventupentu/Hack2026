# Hugging Face Integration Guide (Hack2026)

Este documento explica como publicar automaticamente el modelo en Hugging Face Hub durante entrenamiento e inferencia, con enfoque en:

- actualizacion automatica de pesos
- trazabilidad maxima
- transparencia tecnica para evaluacion Open Source AI

## 1) Objetivo

Integrar el pipeline del proyecto con Hugging Face Hub para que cada corrida deje evidencia versionada de:

- checkpoints de entrenamiento
- mejor checkpoint estable para consumo (`checkpoints/best.pt`)
- metricas
- outputs de inferencia (submission y resumen)
- metadata de reproducibilidad (hashes, config, git, entorno)

## 2) Donde se monta y donde vive cada cosa

### En tu entorno local

- Configuracion: `config/config.yaml` (`hub.*`)
- Tipo de config: `src/config.py` (`Hub` dataclass)
- Sincronizador: `src/utils/hf_hub_sync.py`
- Integracion en training: `src/models/retrieval_openclip.py`
- Integracion en inferencia:
- `src/infer.py`
- `src/new_infer.py`

### En Hugging Face Hub (repo de tipo model)

Estructura remota por default:

- `artifacts/train/<run_id>/checkpoints/epoch_*.pt`
- `artifacts/train/<run_id>/checkpoints/best.pt`
- `artifacts/train/<run_id>/metrics/metrics.jsonl`
- `artifacts/train/<run_id>/traceability/*.json`
- `artifacts/infer/<run_id>/outputs/test_submission*.csv`
- `artifacts/infer/<run_id>/outputs/val_metrics*.json`
- `artifacts/infer/<run_id>/traceability/*.json`
- `checkpoints/best.pt` (puntero estable al mejor checkpoint mas reciente)
- `README.md` (model card auto-actualizada con metadata clave)
- `latest_train.json` y `latest_inference.json`

`run_id` se genera automaticamente con timestamp UTC + commit git corto (o valor custom con `hub.run_name`).

## 3) Que hace la integracion automaticamente

## Entrenamiento

Cuando `hub.enabled=true`:

1. Crea/valida repo remoto (`create_repo(..., exist_ok=True)`).
2. Registra contexto inicial del run:
- config resuelta
- archivos de datos usados
- commit/branch/dirty de git
- versiones de librerias
3. En cada epoch (si `hub.push_every_epoch=true`):
- sube checkpoint de esa epoch
- sube/actualiza `metrics.jsonl`
- genera evento de trazabilidad por epoch
4. Cuando mejora la metrica (si `hub.push_best=true`):
- sube `best.pt` del run
- actualiza `checkpoints/best.pt` estable
- actualiza model card (`README.md`) con mejor metrica y epoch
5. Al finalizar:
- guarda evento `train_complete`

## Inferencia

Cuando `hub.enabled=true` y `hub.push_inference=true`:

1. Registra contexto del run de inferencia.
2. Sube:
- submission CSV
- metricas JSON
- evento de trazabilidad con hash y checkpoint usado
3. Actualiza `latest_inference.json`.

## 4) Como activarlo (paso a paso)

### 4.1 Requisitos

Instalar dependencias:

```bash
pip install -r requirements.txt
```

### 4.2 Autenticacion

Exportar token:

```bash
export HF_TOKEN=hf_xxx_tu_token
```

Puedes cambiar el nombre de variable con `hub.token_env`.

### 4.3 Ejecutar entrenamiento con publicacion automatica

```bash
python -m src.train \
  hub.enabled=true \
  hub.repo_id=TU_USUARIO/TU_REPO_MODELO \
  hub.push_every_epoch=true \
  hub.push_best=true \
  hub.fail_on_error=false
```

### 4.4 Ejecutar inferencia con publicacion automatica

```bash
python -m src.infer \
  hub.enabled=true \
  hub.repo_id=TU_USUARIO/TU_REPO_MODELO \
  hub.push_inference=true \
  infer.checkpoint_path=outputs/retrieval_openclip/best.pt
```

Tambien aplica para:

```bash
python -m src.new_infer \
  hub.enabled=true \
  hub.repo_id=TU_USUARIO/TU_REPO_MODELO \
  hub.push_inference=true
```

## 5) Parametros de configuracion (`hub.*`)

- `hub.enabled`: activa/desactiva sync
- `hub.repo_id`: repo en formato `owner/name`
- `hub.private`: repo privado o publico
- `hub.token_env`: nombre de variable de token
- `hub.fail_on_error`: si `true`, corta ejecucion ante error de sync
- `hub.create_pr`: crea PR en lugar de commit directo
- `hub.push_every_epoch`: publica checkpoints por epoch
- `hub.push_best`: publica mejor checkpoint y actualiza puntero estable
- `hub.push_inference`: publica outputs de inferencia
- `hub.run_name`: nombre de run (opcional)
- `hub.remote_train_dir`: raiz remota para training
- `hub.remote_infer_dir`: raiz remota para inferencia
- `hub.license`, `hub.pipeline_tag`, `hub.base_model`, `hub.datasets`, `hub.tags`: metadata de model card

## 6) Trazabilidad y reproducibilidad (maxima)

Cada evento publica JSON con:

- SHA256 de checkpoints, metricas y outputs
- tamano y fecha de modificacion
- config completa resuelta
- commit git, branch y estado dirty
- timestamp UTC
- versions de runtime y librerias criticas

Esto permite:

- auditoria tecnica reproducible
- comparacion exacta entre corridas
- evidencia clara de como se obtuvo cada resultado

## 7) Justificaciones tecnicas y para el programa Open Source AI

### Integracion del modelo

- Integracion real a nivel pipeline (training + inferencia), no solo subida manual.
- Publicacion versionada de pesos y metricas por corrida.

### Optimizacion

- Se preservan checkpoints por epoch y mejor checkpoint para analizar convergencia.
- Facilita comparar variantes de fine-tuning y post-procesamiento entre corridas.

### Transparencia

- Model card y metadatos con fuente de datos, base model, licencia y tags.
- Trazas con hash y config resuelta para reproducibilidad end-to-end.

### Originalidad

- Combina retrieval multimodal, deteccion por cajas y post-procesamiento, con publicacion automatizada abierta.
- Estructura de evidencia lista para evaluacion tecnica y colaboracion open-source.

## 8) Buenas practicas recomendadas

- Mantener `hub.repo_id` publico para evaluacion (salvo datos sensibles).
- Definir `hub.run_name` con convencion clara (ej: `exp-openclip-lr3e5`).
- Usar `hub.fail_on_error=true` en pipelines de CI para no perder trazabilidad.
- Documentar cambios importantes de hiperparametros en mensajes de commit/run name.

## 9) Troubleshooting rapido

### Error: token o permisos

- Verifica `HF_TOKEN`.
- Verifica permisos de escritura en el repo.

### Error: dependencia faltante

- Instalar `huggingface_hub` (ya incluido en `requirements.txt`).

### Quiero correr sin publicar

- Usa `hub.enabled=false` (default).
- La trazabilidad local en `outputs/.../traceability` se mantiene.

## 10) Resumen ejecutivo

Con esta integracion, el proyecto publica automaticamente pesos, metricas y evidencia reproducible en Hugging Face Hub, alineado con criterios de integracion, optimizacion, transparencia y enfoque open-source exigidos por el programa.
