//-------------------------------------------------------------------------------------
// Tiny assertion harness for the vite-node GPU tests. report() exits the process
// explicitly (0 on success, 1 on failure) to avoid the Dawn teardown segfault.
//-------------------------------------------------------------------------------------

let passed = 0;
let failed = 0;

export function check(name: string, ok: boolean, detail = ""): void {
  if (ok) {
    passed++;
    console.log(`  ✓ ${name}${detail ? ` (${detail})` : ""}`);
  } else {
    failed++;
    console.log(`  ✗ ${name}${detail ? ` (${detail})` : ""}`);
  }
}

export function report(): never {
  console.log(`\n${passed} passed, ${failed} failed`);
  process.exit(failed === 0 ? 0 : 1);
}
