// Debug PLU solver issue - why everything is classified as logical

function matrixVectorMod2(matrix, vector) {
    const result = [];
    for (let i = 0; i < matrix.length; i++) {
        let sum = 0;
        for (let j = 0; j < matrix[i].length; j++) {
            sum += matrix[i][j] * vector[j];
        }
        result.push(sum % 2);
    }
    return result;
}

function pluSolve(plu, y) {
    if (!plu || !plu.L || !plu.U || !plu.P) {
        return null;
    }
    
    const rank = plu.rank;
    const numRows = plu.P.length;
    const numCols = plu.U[0].length;
    
    // Apply permutation P to y: Py = P @ y
    const Py = new Array(numRows).fill(0);
    for (let i = 0; i < numRows; i++) {
        for (let j = 0; j < numRows; j++) {
            if (plu.P[i][j] === 1) {
                Py[i] = y[j];
                break;
            }
        }
    }
    
    // Forward substitution: solve Lb = Py for b
    const b = new Array(rank).fill(0);
    for (let i = 0; i < rank; i++) {
        let sum = 0;
        for (let j = 0; j < i; j++) {
            sum ^= (plu.L[i][j] * b[j]);
        }
        b[i] = Py[i] ^ sum;
    }
    
    // Check if y is in the image
    for (let i = rank; i < numRows; i++) {
        let sum = 0;
        for (let j = 0; j < rank; j++) {
            sum ^= (plu.L[i][j] * b[j]);
        }
        if ((Py[i] ^ sum) !== 0) {
            return null;
        }
    }
    
    // Backward substitution: solve Ux = b for x
    const x = new Array(numCols).fill(0);
    for (let i = rank - 1; i >= 0; i--) {
        let sum = 0;
        for (let j = plu.pivot_cols[i] + 1; j < numCols; j++) {
            sum ^= (plu.U[i][j] * x[j]);
        }
        x[plu.pivot_cols[i]] = b[i] ^ sum;
    }
    
    return x;
}

// Test with simple identity matrix case
console.log("=== Test 1: Identity Matrix ===");
const plu1 = {
    L: [[1, 0], [0, 1]],
    U: [[1, 0], [0, 1]],
    P: [[1, 0], [0, 1]],
    rank: 2,
    pivot_cols: [0, 1]
};
const y1 = [1, 0];
const sol1 = pluSolve(plu1, y1);
console.log("Input:", y1);
console.log("Solution:", sol1);
console.log("Verify: I @ sol =", sol1);
console.log("Matches:", JSON.stringify(sol1) === JSON.stringify(y1), "\n");

// Test with a simple stabilizer example
console.log("=== Test 2: Simple 3-qubit code stabilizer ===");
// S = [[1, 1, 0], [0, 1, 1]] (two checks on 3 qubits)
// First stabilizer: qubit 0 + qubit 1
// Error = first stabilizer = [1, 1, 0]
const S = [[1, 1, 0], [0, 1, 1]];
const plu2 = {
    L: [[1, 0], [0, 1]],
    U: [[1, 1, 0], [0, 1, 1]],
    P: [[1, 0], [0, 1]],
    rank: 2,
    pivot_cols: [0, 1]
};
const error2 = [1, 1, 0];  // First stabilizer
const syndrome2 = matrixVectorMod2(S, error2);
console.log("Error (first stabilizer):", error2);
console.log("Syndrome S @ error:", syndrome2);
console.log("Syndrome is zero:", syndrome2.every(x => x === 0));

if (syndrome2.every(x => x === 0)) {
    const sol2 = pluSolve(plu2, error2);
    console.log("PLU solution:", sol2);
    
    if (sol2) {
        const reconstructed2 = matrixVectorMod2(S, sol2);
        console.log("S @ solution:", reconstructed2);
        console.log("Matches error:", JSON.stringify(reconstructed2) === JSON.stringify(error2));
    }
}

console.log("\n=== Test 3: Logical operator (not in row space) ===");
// Same stabilizer matrix, but error NOT in row space
// For the matrix [[1,1,0], [0,1,1]], row space has dim 2
// Logical could be [1, 0, 1] (XOR of both stabilizers gives [1,0,1])
const error3 = [1, 0, 1];
const syndrome3 = matrixVectorMod2(S, error3);
console.log("Error (logical):", error3);
console.log("Syndrome S @ error:", syndrome3);
console.log("Syndrome is zero:", syndrome3.every(x => x === 0));

if (syndrome3.every(x => x === 0)) {
    const sol3 = pluSolve(plu2, error3);
    console.log("PLU solution:", sol3);
    
    if (sol3) {
        const reconstructed3 = matrixVectorMod2(S, sol3);
        console.log("S @ solution:", reconstructed3);
        console.log("Matches error:", JSON.stringify(reconstructed3) === JSON.stringify(error3));
    } else {
        console.log("PLU returned null - error not in row space");
    }
}
