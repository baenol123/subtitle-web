const assert = require('node:assert/strict');
const { test } = require('node:test');
const { loadApp } = require('./helpers/load-app.cjs');

function file(path) {
  return { name: path.split('/').at(-1), webkitRelativePath: path, size: 0 };
}

function tracks() {
  // Same structure as the reported collection, with neutral synthetic titles.
  return Array.from({ length: 5 }, (_, i) => {
    const number = String(i + 1).padStart(2, '0');
    return [
      file(`Collection/02.Audio/${number}.Track ${number}.wav`),
      file(`Collection/02.Audio/SE無し版/${number}(SE無し).Track ${number}.wav`),
    ];
  });
}

test('pairs all five tracks when the child filename has an inline SE marker', () => {
  const pairs = tracks();
  for (const files of [pairs.flat(), pairs.flat().reverse()]) {
    const { primaryOf, filesToProcess } = loadApp().pairAudioVariants(files);
    assert.equal(primaryOf.size, 5);
    assert.equal(filesToProcess.length, 5);
    for (const [original, clean] of pairs) {
      assert.equal(primaryOf.get(original), clean);
      assert.ok(filesToProcess.includes(clean));
      assert.ok(!filesToProcess.includes(original));
    }
  }
});

test('keeps support for identical parent and child filenames', () => {
  const original = file('Audio/01.Opening.wav');
  const clean = file('Audio/水音SEなし/01.Opening.wav');
  assert.equal(loadApp().pairAudioVariants([original, clean]).primaryOf.get(original), clean);
});

for (const marker of ['(SE無し)', '（SEなし）', '[SEless]', '【効果音なし】', '(SE 없음)']) {
  test(`recognizes the explicit filename marker ${marker}`, () => {
    const original = file('Audio/01.Opening.wav');
    const clean = file(`Audio/SE無し版/01${marker}.Opening.wav`);
    assert.equal(loadApp().pairAudioVariants([original, clean]).primaryOf.get(original), clean);
  });
}

test('does not erase track numbers, title tags, titles, or file extensions', () => {
  const clean = file('Audio/SE無し版/01(SE無し).[Bonus]Opening.wav');
  for (const name of ['02.[Bonus]Opening.wav', '01.[Main]Opening.wav', '01.[Bonus]Ending.wav', '01.[Bonus]Opening.mp3']) {
    const original = file(`Audio/${name}`);
    const { primaryOf, filesToProcess } = loadApp().pairAudioVariants([original, clean]);
    assert.equal(primaryOf.size, 0, name);
    assert.equal(filesToProcess.length, 2);
  }
});

test('only pairs a clean child folder with its own immediate parent', () => {
  for (const folder of ['Audio/Alternative', 'Other/SE無し版', 'Audio/Extra/SE無し版']) {
    const files = [file('Audio/01.Opening.wav'), file(`${folder}/01(SE無し).Opening.wav`)];
    assert.equal(loadApp().pairAudioVariants(files).primaryOf.size, 0);
  }
});

test('does not guess between multiple clean versions', () => {
  const files = [
    file('Audio/01.Opening.wav'),
    file('Audio/SE無し版/01.Opening.wav'),
    file('Audio/SEless/01.Opening.wav'),
  ];
  const { primaryOf, filesToProcess } = loadApp().pairAudioVariants(files);
  assert.equal(primaryOf.size, 0);
  assert.equal(filesToProcess.length, 3);
});

test('each marked parent variant reuses the uniquely preferred clean file', () => {
  const files = [file('Audio/01.Opening.wav'), file('Audio/01(SEあり).Opening.wav'), file('Audio/SE無し版/01(SE無し).Opening.wav')];
  const { primaryOf } = loadApp().pairAudioVariants(files);
  assert.equal(primaryOf.get(files[0]), files[2]);
  assert.equal(primaryOf.get(files[1]), files[2]);
});

test('ignores subtitle inputs and handles a clean file without a counterpart', () => {
  const files = [file('Audio/01.Opening.srt'), file('Audio/SE無し版/01(SE無し).Opening.srt'), file('Audio/SE無し版/02(SE無し).Other.wav')];
  const { primaryOf, filesToProcess } = loadApp().pairAudioVariants(files);
  assert.equal(primaryOf.size, 0);
  assert.equal(filesToProcess.length, 3);
  assert.equal(loadApp().pairAudioVariants([]).filesToProcess.length, 0);
});

test('preserves same-folder suffix pairing', () => {
  const original = file('Audio/01.Opening.wav');
  const clean = file('Audio/01.Opening_SEless.wav');
  const app = loadApp();
  assert.equal(app.pairAudioVariants([original, clean]).primaryOf.get(original), clean);
});

function element() {
  return { checked: false, value: '', children: [], classList: { add() {}, remove() {}, toggle() {} }, append(...children) { this.children.push(...children); } };
}

function pipelineApp(files) {
  const els = Object.fromEntries(['skipTranslate', 'aiRefine', 'refineModel', 'renameKorean', 'errorBanner', 'resultPanel', 'resultsList', 'downloadAllBtn', 'progressPanel', 'startBtn', 'cancelBtn', 'resultStats', 'fileInfo'].map(key => [key, element()]));
  const calls = [];
  const errors = [];
  const app = loadApp({
    els, selectedFiles: files, extraFiles: [], running: false,
    document: { createElement: element }, AbortController,
    T: { filesSelected: count => String(count), seReuseKind: 'reuse-clean', mediaKind: 'extract', batchDone: (ok, total) => `${ok}/${total}`, done: 'done' },
    groqKeys: () => ['test'], geminiKeys: () => ['test'], isGeminiModel: () => true,
    isElevenLabsModel: () => false, isFatalApiError: () => false,
    resetSteps() {}, setProgress() {}, setStatus() {}, checkCancelled() {}, renderResultRow() {},
    showError: message => errors.push(message),
    translateRelDir: dir => dir,
    processOne: async input => {
      calls.push(input);
      return { fileName: input.name, baseName: input.name.replace(/\.[^.]+$/, ''), translatedName: input.name.replace(/\.[^.]+$/, ''), relDir: input.webkitRelativePath.slice(0, input.webkitRelativePath.lastIndexOf('/')), origRelDir: input.webkitRelativePath.slice(0, input.webkitRelativePath.lastIndexOf('/')), originalSrt: `original:${input.name}`, translatedSrt: `translated:${input.name}`, blockCount: 1, failed: 0 };
    },
  });
  return { app, els, calls, errors };
}

test('file preview labels the five parent tracks for subtitle reuse', () => {
  const { app, els } = pipelineApp(tracks().flat());
  app.handleFiles(tracks().flat(), { filterExts: true });
  const labels = els.fileInfo.children.map(child => child.textContent);
  assert.equal(labels.filter(text => text.includes('reuse-clean')).length, 5);
  assert.equal(labels.filter(text => text.includes('extract')).length, 5);
});

test('batch processes five clean tracks and exports subtitles for all ten files', async () => {
  const pairs = tracks();
  const { app, els, calls, errors } = pipelineApp(pairs.flat());
  await app.run();
  assert.deepEqual(errors, []);
  assert.equal(calls.length, 5);
  assert.ok(calls.every(input => input.webkitRelativePath.includes('/SE無し版/')));
  assert.equal(app.allResults.length, 10);
  assert.equal(els.resultStats.textContent, '10/10');
  for (const [original, clean] of pairs) {
    const result = app.allResults.find(item => item.fileName === original.name);
    assert.equal(result.originalSrt, `original:${clean.name}`);
    assert.equal(result.translatedSrt, `translated:${clean.name}`);
    assert.equal(result.origRelDir, 'Collection/02.Audio');
  }
});

function mixedTracks() {
  const files = [];
  const preferred = [];
  for (let n = 1; n <= 6; n++) {
    const title = `${String(n).padStart(2, '0')}.Track ${n}`;
    files.push(file(`Collection/Audio/${title}　加工あり.wav`));
    const raw = file(`Collection/Audio/加工なし/${title}　加工なし.wav`);
    files.push(raw);
    if ([1, 3, 4].includes(n)) {
      files.push(file(`Collection/Audio/SEなし/${title}　加工あり　SEなし.wav`));
      const clean = file(`Collection/Audio/SEなし/${title}　加工なし　SEなし.wav`);
      files.push(clean);
      preferred.push(clean);
    } else {
      let source = raw;
      if (n === 2) {
        source = file(`Collection/Audio/SEなし/${title}　加工あり.wav`);
        files.push(source);
      }
      if (n === 5) {
        files.push(file(`Collection/Audio/SEなし/${title}　加工あり　SEなし.wav`));
        source = file(`Collection/Audio/SEなし/${title}　SEなし.wav`);
        files.push(source);
      }
      preferred.push(source);
    }
  }
  return { files, preferred };
}

test('mixed SE and processing variants use six clean sources for all 21 results', async () => {
  const { files, preferred } = mixedTracks();
  for (const input of [files, [...files].reverse()]) {
    const { app, calls, errors, els } = pipelineApp(input);
    await app.run();
    assert.deepEqual(errors, []);
    assert.equal(calls.length, 6);
    assert.deepEqual(new Set(calls), new Set(preferred));
    assert.equal(app.allResults.length, 21);
    assert.equal(els.resultStats.textContent, '21/21');
    for (const original of files) {
      const result = app.allResults.find(r => `${r.origRelDir}/${r.fileName}` === original.webkitRelativePath);
      assert.ok(result, original.webkitRelativePath);
      const source = preferred.find(f => f.name.slice(0, 2) === original.name.slice(0, 2));
      assert.equal(result.originalSrt, `original:${source.name}`);
      assert.equal(result.translatedSrt, `translated:${source.name}`);
    }
  }
});

test('mixed folder preview marks 15 files for reuse and six for extraction', () => {
  const { files } = mixedTracks();
  const { app, els } = pipelineApp(files);
  app.handleFiles(files, { filterExts: true });
  const labels = els.fileInfo.children.map(child => child.textContent);
  assert.equal(labels.filter(text => text.includes('reuse-clean')).length, 15);
  assert.equal(labels.filter(text => text.includes('extract')).length, 6);
});

test('folder and filename markers cannot leave indirect reuse targets without results', async () => {
  const files = ['Audio/01.Title.wav', 'Audio/SEなし/01.Title.wav', 'Audio/SEなし/01.Title　加工なし.wav'].map(file);
  const { app, calls } = pipelineApp(files);
  await app.run();
  assert.deepEqual(calls, [files[2]]);
  assert.equal(app.allResults.length, 3);
});

test('Korean effect-free and SE-free sibling folders choose the doubly clean version', async () => {
  const files = ['Audio/01.Title 이펙트있음.wav', 'Audio/이펙트없음/01.Title 이펙트없음.wav', 'Audio/효과음없음/01.Title 이펙트없음 효과음없음.wav'].map(file);
  const { app, calls } = pipelineApp(files);
  await app.run();
  assert.deepEqual(calls, [files[2]]);
  assert.equal(app.allResults.length, 3);
});

test('nested processing and SE folders can identify a clean source without filename markers', async () => {
  const files = ['Audio/01.Title.wav', 'Audio/加工なし/01.Title.wav', 'Audio/加工なし/SEなし/01.Title.wav'].map(file);
  const { app, calls } = pipelineApp(files);
  await app.run();
  assert.deepEqual(calls, [files[2]]);
  assert.equal(app.allResults.length, 3);
});

test('equally preferred processing-free candidates stay independent', async () => {
  const files = ['Audio/01.Title 加工あり.wav', 'Audio/加工なし/01.Title 加工なし.wav', 'Audio/エフェクトなし/01.Title 加工なし.wav'].map(file);
  const { app, calls } = pipelineApp(files);
  await app.run();
  assert.equal(calls.length, 3);
  assert.equal(app.allResults.length, 3);
});

test('SE-free with unknown processing is not labeled as unprocessed', () => {
  const variant = loadApp().audioVariantOf(file('Audio/SEなし/05.Title　SEなし.wav'));
  assert.equal(variant.flags.se, false);
  assert.equal(variant.flags.processing, null);
});

test('explicit filename status takes priority over a conflicting folder label', () => {
  const variant = loadApp().audioVariantOf(file('Audio/SEなし/01.Title　SEあり.wav'));
  assert.equal(variant.flags.se, true);
});

test('contradictory filename labels and empty normalized names do not get paired', () => {
  const app = loadApp();
  const files = ['Audio/01.Title.wav', 'Audio/01(SEあり).Title_SEなし.wav', 'Audio/SEless.wav'].map(file);
  const { primaryOf, filesToProcess } = app.pairAudioVariants(files);
  assert.equal(primaryOf.size, 0);
  assert.equal(filesToProcess.length, 3);
});

test('plain duplicate names without clean labels are not silently merged', () => {
  const files = [file('Audio/01.Title.wav'), file('Audio/01.Title.wav')];
  assert.equal(loadApp().pairAudioVariants(files).filesToProcess.length, 2);
});

test('English SE and processing labels work together in one folder', () => {
  const files = ['01.Title FXon SEon.wav', '01.Title(no FX)_no SE.wav'].map(file);
  const { primaryOf, filesToProcess } = loadApp().pairAudioVariants(files);
  assert.equal(primaryOf.get(files[0]), files[1]);
  assert.equal(filesToProcess.length, 1);
});

test('does not cross collection or ordinary disc folder boundaries', () => {
  const files = ['Collection A/Audio/01.Title.wav', 'Collection B/Audio/SEなし/01.Title.wav', 'Collection A/Audio/Disc 2/加工なし/01.Title.wav'].map(file);
  assert.equal(loadApp().pairAudioVariants(files).primaryOf.size, 0);
});
