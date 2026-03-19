# AI 문서검색 시스템 — 사용 가이드
[![CI](https://github.com/shoxa-mir/RAG-Powered-Local-GenAI-System/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/shoxa-mir/RAG-Powered-Local-GenAI-System/actions/workflows/ci.yml)
## 사전 요구사항

- **Docker Desktop** (Docker Compose 포함)
- **NVIDIA GPU** + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **모델 파일** (`models/` 디렉토리에 위치)

---

## 최초 설정 (한 번만)

### 1. 모델 다운로드

```bash
pip install sentence-transformers huggingface-hub
python download_models.py
```

다운로드되는 모델:
| 모델 | 용도 | 크기 |
|------|------|------|
| BAAI/bge-m3 | 임베딩 (키워드 검색) | ~2.2GB |
| SOLAR-10.7B-Instruct GGUF Q4_K_M | LLM (질의응답) | ~6.6GB |

### 2. 최초 빌드

```bash
docker compose build
```

> LLM 이미지는 CUDA 컴파일로 인해 **10~20분** 소요됩니다.
> 이후 재빌드 시에는 캐시를 사용하므로 빠릅니다.

---

## 서비스 시작 / 중지

### 시작

```bash
docker compose up -d
```

| 서비스 | 포트 | 설명 |
|--------|------|------|
| **app** | [localhost:8001](http://localhost:8001) | 웹 UI (문서 업로드, 검색, 질의응답) |
| **llm** | localhost:8000 | LLM 서버 (GPU 추론) |
| **qdrant** | localhost:6333 | 벡터 데이터베이스 |

> LLM 서버는 모델 로딩에 **30~60초** 소요됩니다.

### 중지

```bash
docker compose down
```

> 데이터(문서, 인덱스, 벡터)는 Docker 볼륨에 보존되므로 재시작 후에도 유지됩니다.

### 완전 초기화 (데이터 삭제 포함)

```bash
docker compose down -v
```

---

## 상태 확인

```bash
# 실행 중인 컨테이너 확인
docker compose ps

# 로그 확인 (전체)
docker compose logs

# 특정 서비스 로그
docker compose logs app
docker compose logs llm
docker compose logs qdrant

# 실시간 로그
docker compose logs -f
```

---

## 코드 수정 후 반영

```bash
# app.py 또는 템플릿 변경 시 (빠름, ~5초)
docker compose up -d --build app

# serve_llm.py 변경 시 (빠름, 빌더 캐시 사용)
docker compose up -d --build llm
```

---

## 문제 해결

| 증상 | 해결 |
|------|------|
| LLM 연결 안됨 | LLM 로딩 중 (30~60초 대기), `docker compose logs llm` 확인 |
| GPU 인식 안됨 | `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi` 테스트 |
| 포트 충돌 | `docker compose down` 후 재시작, 또는 `docker-compose.yml` 포트 변경 |
| 디스크 부족 | `docker system prune` 으로 미사용 이미지 정리 |
