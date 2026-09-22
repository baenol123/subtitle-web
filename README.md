# 자막공장 (MakeSubtitles)

**https://makesubtitles.com** — 영상 자막 추출 + AI 번역 웹 도구

영상이나 오디오 파일을 올리면 Whisper AI가 자막(SRT)을 추출하고, Gemini 또는 Claude가 자연스럽게 번역합니다.
**파일은 서버에 업로드되지 않고 브라우저 안에서만 처리됩니다.**

Upload a video or audio file — Whisper AI extracts SRT subtitles, then Gemini or Claude translates them naturally.
**Files never leave your browser.** English version: https://makesubtitles.com/en/

## 동작 방식 / How it works

```
영상 업로드 (브라우저에서만 처리)
  → ffmpeg.wasm으로 오디오 추출 (로컬)
  → Groq Whisper large-v3-turbo로 자막 추출 (사용자 API 키)
  → 환각 문구/반복 문장 자동 필터
  → (선택) AI 교정 — 오인식·구두점만 수정, 번역은 하지 않음
  → AI 배치 번역 — 문맥 참고, 톤/용어집/교정치환 지원 (사용자 API 키)
  → 원문 SRT + 번역 SRT 다운로드
```

- 서버 없는 순수 정적 사이트 (Cloudflare Pages 호스팅)
- API 키는 사용자 브라우저(localStorage)에만 저장
- SRT / VTT 파일을 올리면 번역만 수행

### 효과음·이펙트 없는 판의 자막 재사용

폴더째 선택했을 때 `음성/01.제목.wav`와
`음성/SE無し版/01(SE無し).제목.wav`는 같은 트랙으로 인식합니다.
같은 트랙의 `SEなし`(효과음 없음), `加工なし`(가공·이펙트 없음) 형제/중첩 폴더와
`01.제목　加工なし　SEなし.wav`처럼 표기가 여러 개 붙은 파일명도 인식합니다.
`효과음없음`, `이펙트없음`, `エフェクトなし` 등 한국어·일본어·영어 표기를 지원합니다.

우선순위는 **둘 다 없음 → 효과음 없음 → 가공 없음 → 일반판**입니다.
효과음 없음이 같으면 가공 없음, 가공 미표기, 가공 있음 순으로 고릅니다.
미표기는 가공 없음으로 확정하지 않습니다. 가장 우선하는 후보가 여러 개면 각각 처리합니다.
번호·제목·일반 말머리·확장자와 작품/디스크 폴더 경계는 유지해 다른 트랙을 합치지 않습니다.

선택한 음성만 추출·교정·번역하고 모든 판에 같은 자막을 제공합니다.
파일 목록에서 재사용하는 기준 파일 경로를 확인할 수 있습니다.
각 판의 대사 타이밍은 같아야 합니다. 자막 싱크는 자동 조정하지 않습니다.

## 번역 모델 / Translation models

기본값은 **`gemini-3.1-flash-lite`** 입니다. 모델 id가 `gemini`로 시작하면 Gemini API,
`gpt`로 시작하면 OpenAI API, 그 외에는 Anthropic API로 라우팅됩니다.

| 엔진 | 필요한 키 | 배치 방식 |
|---|---|---|
| Gemini | Google Gemini API 키 (예비 키 2개까지) | 하루 **요청 횟수** 한도가 빡빡해 파일 전체를 한 번에 전송 |
| Claude | Anthropic API 키 | 번역 20줄 / 교정 40줄 단위 배치 |
| GPT 6 Luna / GPT 5.6 Luna | OpenAI API 키 | 번역 20줄 / 교정 40줄 단위 배치 |

Claude Opus 5.5는 Anthropic API 정책상 thinking을 항상 사용하므로, 이 모델을 선택하면 해당 옵션을 끄지 않고 호출합니다.

Groq·Gemini 모두 예비 키를 넣어두면 한도 도달 시 자동 전환합니다.
무료 한도는 계정 단위라 예비 키는 **다른 계정**에서 발급해야 효과가 있습니다.

## 기술 스택 / Tech

- Vanilla JS (빌드 도구 없음), ffmpeg.wasm, Groq API (Whisper), Google Gemini API, Anthropic API
- `vendor/`에 ffmpeg 라이브러리 로컬 번들 (same-origin worker 필요)

> ⚠ **외부 스크립트는 반드시 버전을 고정할 것.** 사용자 API 키가 이 페이지의 localStorage에
> 저장되므로, 버전을 열어두면 패키지나 CDN이 오염됐을 때 전 사용자의 키가 유출될 수 있습니다.
> 현재 `@anthropic-ai/sdk@0.115.0`, `@ffmpeg/core@0.12.10`으로 고정돼 있습니다.

## 개발 / Development

```
python -m http.server 8000   # 로컬 실행 (file:// 로는 동작하지 않음)
```

새 도구 페이지는 `_template.html`을 복사해서 만듭니다. 자세한 내용은 `사용법.txt` 참고.

SE 판별 및 처리 흐름 회귀 테스트 (Node.js 24, 추가 패키지·API 호출 없음):

```
node --test tests/se-variants.test.cjs
```
