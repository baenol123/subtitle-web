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

## 번역 모델 / Translation models

기본값은 **`gemini-3.1-flash-lite`** 입니다. 모델 id가 `gemini`로 시작하면 Gemini API,
그 외에는 Anthropic API로 라우팅됩니다 (`isGeminiModel()`).

| 엔진 | 필요한 키 | 배치 방식 |
|---|---|---|
| Gemini | Google Gemini API 키 (예비 키 2개까지) | 하루 **요청 횟수** 한도가 빡빡해 파일 전체를 한 번에 전송 |
| Claude | Anthropic API 키 | 번역 20줄 / 교정 40줄 단위 배치 |

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
