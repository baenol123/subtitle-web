const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const appSource = readFileSync(resolve(__dirname, '../app.js'), 'utf8');

function loadClaudeThinkingConfig() {
  const start = appSource.indexOf('function claudeThinkingConfig');
  const end = appSource.indexOf('async function callClaude', start);
  if (start < 0 || end < 0) throw new Error('Claude thinking config section is missing');
  const sandbox = {};
  vm.runInContext(`${appSource.slice(start, end)}\nglobalThis.claudeThinkingConfig = claudeThinkingConfig;`, vm.createContext(sandbox));
  return sandbox.claudeThinkingConfig;
}

test('keeps thinking enabled for Claude Opus 5.5 because disabling it is unsupported', () => {
  const claudeThinkingConfig = loadClaudeThinkingConfig();
  assert.equal(claudeThinkingConfig('claude-opus-5-5'), null);
});

test('continues disabling thinking for Claude models that support it', () => {
  const claudeThinkingConfig = loadClaudeThinkingConfig();
  assert.deepEqual(claudeThinkingConfig('claude-opus-5'), { type: 'disabled' });
});
