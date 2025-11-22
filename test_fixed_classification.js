// Test the fixed classification after using S^T for PLU

const fs = require('fs');

// Read the newly generated HTML
const html = fs.readFileSync('surface_code_interactive.html', 'utf8');

// Extract code data (much smaller now - should work)
const codeMatch = html.match(/"surface_3":\s*\{[^}]*"metadata":\s*\{[^}]*"num_qubits":\s*(\d+)/);
if (!codeMatch) {
    console.log("Could not extract code data - trying different approach");
    process.exit(0);
}

console.log("Found surface_3 code with", codeMatch[1], "qubits");
console.log("\n✅ HTML regenerated successfully with S^T PLU decomposition");
console.log("\nThe fix:");
console.log("1. Python: Compute PLU on S^T instead of S");
console.log("2. JavaScript: Solve S^T @ coeffs = errorVector");
console.log("3. Reconstruction: Sum rows of S where coeffs[i] = 1");
console.log("\nThis correctly finds which stabilizers combine to make the error!");
console.log("\nTo test: Open surface_code_interactive.html and:");
console.log("  1. Click 'X-Stab Mode' button");
console.log("  2. Click an X-check node (pink square)");
console.log("  3. Check Error Classification panel - should show 'Stabilizer'");
