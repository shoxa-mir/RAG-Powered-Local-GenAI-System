---
title: AI 문서검색 시스템
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8001
hardware: a10g-small
---

# AI 문서검색 시스템 (Demo)

한국어 문서 기반 RAG(Retrieval-Augmented Generation) 시스템. 로컬 LLM + 벡터 검색 + 하이브리드 검색(BM25 + 시맨틱) + 리랭킹을 통해 정확한 문서 검색 및 AI 질의응답을 제공합니다.

**라이브 데모**: [HF Spaces 링크](https://huggingface.co/spaces) (배포 후 업데이트)

---

## 📋 기능

- **📄 문서 업로드**: PDF, DOCX 파일 지원 (다중 선택 가능)
- **🔄 인덱싱**: 문서 자동 청킹 (500토큰, 100 오버랩) → 벡터 임베딩 → Qdrant 저장
- **🔍 하이브리드 검색**:
  - **시맨틱 검색** (벡터 유사도) + **BM25** (키워드 기반)
  - **RRF(Reciprocal Rank Fusion)** 결합 (가중치: 시맨틱 70%, BM25 30%)
  - **CrossEncoder 리랭킹** (상위 10개 재정렬 → 5개 반환)
- **💬 AI 질의응답**: RAG 기반 답변 생성 (SSE 스트리밍)
- **💾 다중 세션**: 대화 히스토리 자동 저장 (로컬 스토리지)

---

## 🏗️ 시스템 아키텍처

### 컨테이너 구성 (HF Spaces)

단일 Docker 컨테이너에서 3개 서비스 실행 (Supervisord):

| 서비스 | 포트 | 설명 |
|--------|------|------|
| **app** | 8001 | FastAPI 웹 UI + RAG 백엔드 |
| **llm** | 8000 | llama-cpp-python (LLM 추론 서버) |
| **qdrant** | 6333 | 벡터 데이터베이스 |

### 모델 구성

| 모델 | 용도 | 실행 환경 | 크기 |
|------|------|---------|------|
| Yarn-Solar-10B-64K (Q4_K_M) | LLM 질의응답 | CUDA (GPU) | ~6.5GB |
| BAAI/bge-m3 | 임베딩 & 키워드 토큰화 | CPU | ~2.2GB |
| dragonkue/bge-reranker-v2-m3-ko | 검색 결과 리랭킹 | CUDA (GPU) | ~560MB |

---

## 🚀 로컬 실행 (개발용)

### 전제 조건

- Docker Desktop (Docker Compose 포함)
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- 권장: VRAM 16GB 이상 (T4는 tight, A100/A6000/RTX 40시리즈 권장)

### 1️⃣ 모델 다운로드 (초회만)

```bash
pip install -r requirements.txt
python download_models.py
```

다운로드되는 모델: `BAAI/bge-m3`, `dragonkue/bge-reranker-v2-m3-ko`, `SOLAR-10.7B-Instruct GGUF`

### 2️⃣ 서비스 시작

```bash
docker compose up -d
```

**초기 로딩 시간**: ~3분 (LLM 모델 로드)

**접근 방법**:
- 웹 UI: [http://localhost:8001](http://localhost:8001)
- LLM API: [http://localhost:8000/v1/models](http://localhost:8000/v1/models)
- Qdrant API: [http://localhost:6333/health](http://localhost:6333/health)

### 3️⃣ 사용 방법

1. **문서 업로드** → 자동 인덱싱
2. **검색**: 키워드 검색으로 관련 문서 조회
3. **질의응답**: RAG 모드 토글 후 질문

### 종료

```bash
docker compose down
```

---

## 🌐 HF Spaces 배포

### 전제 조건

- Hugging Face 계정
- HF Pro ($9/mo, 선택) — 커스텀 도메인 지원

### 1️⃣ Space 생성

1. [Hugging Face Spaces](https://huggingface.co/spaces) → "Create new Space"
2. **SDK**: Docker
3. **Hardware**: A10G (24GB VRAM) 추천
4. **Visibility**: Public (또는 Private)

### 2️⃣ 저장소 클론 및 푸시

```bash
git clone https://huggingface.co/spaces/<user>/<space-name>
cd <space-name>
git remote add origin https://huggingface.co/spaces/<user>/<space-name>
```

이 저장소 파일을 복사하고 푸시:

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### 3️⃣ 첫 실행 대기

- **빌드**: 5-10분 (CUDA 컴파일, 모델 다운로드)
- **초기화**: 추가 3분 (LLM 로드)

HF Spaces **Logs** 탭에서 진행 상황 확인:

```
[...] Downloading BAAI/bge-m3
[OK] Saved embedding model
[...] Downloading bge-reranker-v2-m3-ko
[OK] Saved reranker model
[...] Loading Yarn-Solar-10B-64K
```

### 4️⃣ 커스텀 도메인 (선택사항)

HF Pro 구독 시:

1. Space **Settings** → **Custom Domain**
2. 도메인 등록 (Namecheap, Cloudflare 등)
3. DNS CNAME: `<domain> → <user>-<space>.hf.space`

---

## 🔧 설정 파일

### config.json

```json
{
  "app": { "host": "0.0.0.0", "port": 8001 },
  "llm": {
    "base_url": "http://localhost:8000/v1",
    "model_path": "./models/Yarn-Solar-10B-64K/Yarn-Solar-10b-64k.Q4_K_M.gguf",
    "n_ctx": 65536,
    "n_gpu_layers": -1
  },
  "embedding": { "model_name": "./models/bge-m3", "device": "cpu" },
  "reranker": { "model_name": "./models/bge-reranker-v2-m3-ko", "device": "cuda" },
  "search": { "max_results": 5, "weights": { "semantic": 0.7, "bm25": 0.3 } }
}
```

---

## 📊 성능 지표

로컬 테스트 (RTX 4080 Super):

- **LLM 추론**: ~11-12 토큰/초 (65K 컨텍스트)
- **임베딩**: ~1000 임베딩/초 (BAAI/bge-m3, CPU)
- **검색 지연시간**: ~200-300ms (BM25 + 시맨틱)
- **리랭킹**: ~50-100ms (10개 후보)

---

## 🔍 기술 스택

- **웹 프레임워크**: FastAPI + Jinja2
- **LLM**: llama-cpp-python (GGUF 양자화 모델)
- **임베딩 & 리랭킹**: Sentence-Transformers (HuggingFace)
- **벡터 DB**: Qdrant
- **검색**: BM25 (rank-bm25) + RRF 결합
- **한국어 NLP**: kiwipiepy (형태소 분석)
- **컨테이너화**: Docker + Supervisor

---

## ⚠️ 알려진 제약

- **VRAM**: 65K 컨텍스트 + 10B 모델 = 16GB+ VRAM 필요
- **GPU 필수**: CPU 모드 미지원 (설정 변경 필요)
- **한국어 최적화**: 한국어 문서에만 최적화 (다국어는 bge-m3으로 부분 지원)
- **세션 용량**: 브라우저 로컬 스토리지 (5-10MB)

---

## 📝 라이선스

MIT

---

## 🤝 기여

버그 보고, 기능 제안, 풀 리퀘스트 환영합니다!

---

## 📞 지원

- GitHub Issues: [RAG-Powered-Local-GenAI-System](https://github.com/shoxa-mir/RAG-Powered-Local-GenAI-System)
- HF Discussions: (Space 생성 후 활성화)
