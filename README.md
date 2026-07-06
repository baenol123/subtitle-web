# 자막공방 (MakeSubtitles)

**https://makesubtitles.com** — 영상 자막 추출 + AI 번역 웹 도구

영상이나 오디오 파일을 올리면 Whisper AI가 자막(SRT)을 추출하고, Claude AI가 자연스럽게 번역합니다.
**파일은 서버에 업로드되지 않고 브라우저 안에서만 처리됩니다.**

Upload a video or audio file — Whisper AI extracts SRT subtitles and Claude AI translates them naturally.
**Files never leave your browser.** English version: https://makesubtitles.com/en/

## 동작 방식 / How it works

```
영상 업로드 (브라우저에서만 처리)
  → ffmpeg.wasm으로 오디오 추출 (로컬)
  → Groq Whisper large-v3-turbo로 자막 추출 (사용자 API 키)
  → 환각 문구/반복 문장 자동 필터
  → Claude AI 배치 번역 — 문맥 참고, 톤/용어집 지원 (사용자 API 키)
  → 원문 SRT + 번역 SRT 다운로드
```

- 서버 없는 순수 정적 사이트 (Cloudflare Pages 호스팅)
- API 키는 사용자 브라우저(localStorage)에만 저장
- SRT 파일을 올리면 번역만 수행

## 기술 스택 / Tech

- Vanilla JS (빌드 도구 없음), ffmpeg.wasm, Groq API (Whisper), Anthropic API (Claude)
- `vendor/`에 ffmpeg 라이브러리 로컬 번들 (same-origin worker 필요)

## 개발 / Development

```
python -m http.server 8000   # 로컬 실행 (file:// 로는 동작하지 않음)
```

새 도구 페이지는 `_template.html`을 복사해서 만듭니다. 자세한 내용은 `사용법.txt` 참고.
