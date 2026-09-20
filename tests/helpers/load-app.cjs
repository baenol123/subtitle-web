const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const vm = require('node:vm');

const appPath = resolve(__dirname, '../../app.js');
const source = readFileSync(appPath, 'utf8');

// Run the actual dependency-free app sections without loading CDN SDKs or APIs.
function section(start, end) {
  const from = source.indexOf(start);
  const to = source.indexOf(end, from);
  if (from < 0 || to < 0) throw new Error(`Missing app section: ${start}`);
  return source.slice(from, to);
}

function loadApp(context = {}) {
  const sandbox = vm.createContext(context);
  const parts = [
    section('const SUBTITLE_EXTS =', '// ─────'),
    section('function fileExt(', 'function readEntry('),
    section('const AFFIX_MARK =', '// 제목 "안"'),
    section('function pairCompanionSubtitles(', 'async function run()'),
    section('function handleFiles(', "els.dropZone.addEventListener('click'"),
    section('async function run()', "els.startBtn.addEventListener('click'"),
  ];
  // AUDIO_DIRECT_EXTS is normally declared just before SUBTITLE_EXTS.
  vm.runInContext("const AUDIO_DIRECT_EXTS = ['.mp3', '.m4a', '.wav', '.flac'];", sandbox);
  parts.forEach((part, index) => vm.runInContext(part, sandbox, {
    filename: `app-section-${index}.js`,
  }));
  return sandbox;
}

module.exports = { loadApp };
