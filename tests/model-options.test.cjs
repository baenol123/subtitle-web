const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');

const indexHtml = readFileSync(resolve(__dirname, '../index.html'), 'utf8');

test('translation model selector exposes GPT-6 Luna alongside GPT-5.6 Luna', () => {
  assert.match(indexHtml, /<optgroup label="GPT — OpenAI 키 · 유료">[\s\S]*<option value="gpt-5\.6-luna">/);
  assert.match(indexHtml, /<option value="gpt-6-luna">GPT 6 Luna/);
});
