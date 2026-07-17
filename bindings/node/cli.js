#!/usr/bin/env node
// `npx @warpllm/warpllm` entry point: boots the OpenAI-compatible gateway.
// Flags go verbatim to the shared Rust parser — run with --help for options.
// Ctrl+C exits via Node's default signal handling.
const { serve } = require('./index.js')

serve(process.argv.slice(2)).then(
  () => process.exit(0), // only --help resolves; a running server never does
  (err) => {
    console.error(err.message ?? err)
    process.exit(1)
  },
)
