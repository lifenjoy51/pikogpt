# Inference API 서버

Turbo 백엔드를 메모리에 상주시켜 HTTP로 호출 가능하게 한 최소 MVP. 매 요청마다 체크포인트를 다시 로드하지 않고, 단일 `TurboSampler`가 요청을 직렬화 처리한다.

## 기동

```bash
./gradlew runInferenceApi --args="<체크포인트 디렉토리> [포트=8080]"
```

예:
```bash
./gradlew runInferenceApi --args="/Users/joey51/works/pikogpt/model/stage2/main/v0009 8840"
```

체크포인트 디렉토리는 `checkpoint.json` + `model_weights.bin` + `meta.json`을 포함해야 한다. **Turbo 백엔드로 학습된** 체크포인트만 호환된다 (vec / scalar 백엔드와 포맷 다름). 체크포인트 경로는 절대 경로 권장 — 상대 경로 사용 시 Gradle 작업 디렉토리 기준이다.

기동 로그 예:
```
# Loading Turbo sampler from .../v0009
# 모델 로드 완료 (iter=2100, val loss=2.2103)
# Listening on http://0.0.0.0:8840 (POST /generate, GET /health)
```

## 엔드포인트

### `GET /health`

서버 상태와 모델 컨텍스트 한도를 확인.

```bash
curl -sS http://localhost:8840/health
```
```json
{
  "status": "ok",
  "modelDir": "/.../v0009",
  "maxContext": 32
}
```

### `POST /generate`

프롬프트를 받아 N개 샘플을 반환.

요청 필드 (모두 optional, 미지정 시 서버 기본값 사용):

| 필드 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `prompt` | string | — (필수) | 입력 텍스트 |
| `maxNewTokens` | int | 100 | 생성할 최대 새 토큰 수 |
| `temperature` | float | 0.8 | 1.0=기본, 0.5=결정론적, >1=창의적 |
| `topK` | int | 40 | 상위 K개만 샘플링 (0=비활성) |
| `topP` | float | 1.0 | nucleus 샘플링 누적 확률 (1.0=비활성) |
| `repetitionPenalty` | float | 1.0 | 반복 페널티 (1.0=비활성, 1.1~1.3 권장) |
| `numberOfSamples` | int | 1 | 같은 프롬프트로 생성할 샘플 수 |
| `stopTokenIds` | int[] | `[0]` | 생성 중단 토큰 id (EOS=0). `<\|turn\|>` 등을 추가하면 single-turn 응답에 유용 |

응답:
```json
{ "results": ["...", "...", ...] }
```

호출 예:
```bash
curl -sS -X POST http://localhost:8840/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"once upon a time","maxNewTokens":20,"temperature":0.8,"topK":40,"numberOfSamples":1}'
```
```json
{"results":["once upon a time, and a toy are all numbers we learn to see about animals.  A bronze age"]}
```

`prompt`가 빈 문자열이면 `400`이 반환된다.

## 동작 특성과 한계

- **단일 모델 / 단일 인스턴스**: 시작 시 인자로 1개 모델을 고정. 다른 모델로 바꾸려면 재시작.
- **요청 직렬화**: 내부적으로 `Mutex`로 직렬 처리 — 동시 요청은 큐에 쌓인다. 스루풋이 아니라 정확성을 우선.
- **상태 비저장**: `/generate`는 매 요청 독립. 누적 대화는 클라이언트가 프롬프트에 직접 history를 넣어야 한다.
- **인증 / CORS / 스트리밍 / 배치 / 메트릭 — 없음**. 로컬 개발용 MVP. 필요해지면 별도 작업으로.

## 코드 위치

| 파일 | 내용 |
|---|---|
| `src/main/kotlin/server/InferenceApiMain.kt` | Ktor + Netty 진입점, 라우팅, DTO |
| `src/main/kotlin/turbo/TurboSampler.kt` | `generate(prompt, overrideConfig)` 오버로드로 요청별 파라미터 적용 |
| `build.gradle.kts` | Ktor 의존성 + `runInferenceApi` JavaExec 태스크 |

의존성: `io.ktor:ktor-server-{core,netty,content-negotiation}:2.3.12`, `io.ktor:ktor-serialization-kotlinx-json:2.3.12`.
