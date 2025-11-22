# Interactive Surface Code Website - Updated Design

## Latest Changes ✨

The website has been updated with your requested design preferences:

### Visual Design
- **Light theme** throughout the interface
- **Black and white nodes** in the surface code
- **X and Z labels** displayed inside check nodes (X-checks show "X", Z-checks show "Z")
- **Updated error colors**:
  - X errors: **Red** (#ef4444)
  - Z errors: **Blue** (#3b82f6)
  - Both errors: **Purple** (#9333ea)
- **Syndrome highlighting**: Triggered syndrome nodes glow in **red** with pulsing animation

### Color Scheme Summary

#### Default State
- Qubits: White circles with black border
- X-checks: White squares with black border and "X" label
- Z-checks: White squares with black border and "Z" label
- Edges: Gray (#666666)

#### Error States
- Qubit with X error: **Red**
- Qubit with Z error: **Blue**
- Qubit with both errors: **Purple**

#### Syndrome States
- Triggered X-check or Z-check: **Red glow** with pulsing animation

### UI Theme
- Background: Light gray (#f9fafb)
- Control panel: White with light borders
- Text: Dark gray for readability
- Active controls: Blue accents
- Statistics: Red for X errors, Blue for Z errors, Red for syndromes

## Quick Reference

### Visual Legend

**Nodes:**
- ⚪ White circle = Data qubit (no error)
- 🔴 Red circle = X error on qubit
- 🔵 Blue circle = Z error on qubit
- 🟣 Purple circle = Both X and Z errors
- ⬜ White square with "X" = X-check stabilizer
- ⬜ White square with "Z" = Z-check stabilizer

**Syndrome:**
- 🔴 Red glowing square = Triggered syndrome (pulsing)

### Controls

1. **Show/Hide X-type**: Toggle visibility of X-checks and their edges
2. **Show/Hide Z-type**: Toggle visibility of Z-checks and their edges
3. **Inject X Errors**: Click button, then click qubits to add red X errors
4. **Inject Z Errors**: Click button, then click qubits to add blue Z errors
5. **Clear All Errors**: Reset all errors and syndromes

### Statistics Panel

Real-time display:
- X Errors: Count shown in red
- Z Errors: Count shown in blue
- X Syndrome: Triggered X-checks (red)
- Z Syndrome: Triggered Z-checks (red)

## How It Works

### Error Injection
1. Click "Inject X Errors" or "Inject Z Errors"
2. Active mode is highlighted with blue glow
3. Click any qubit to toggle error on/off
4. Error appears as red (X) or blue (Z) fill
5. Click same qubit again to remove error

### Syndrome Calculation
- **Automatic**: Calculated instantly when errors change
- **X errors trigger Z-checks**: Formula `Hz @ x_error % 2`
- **Z errors trigger X-checks**: Formula `Hx @ z_error % 2`
- **All in JavaScript**: No server needed, runs entirely in browser
- **Visual feedback**: Triggered checks pulse with red glow

### Example Workflow

```
1. Open surface_code_interactive.html
2. Click "Inject X Errors" (button glows blue)
3. Click a qubit → turns red
4. See Z-checks (labeled "Z") light up with red glow = syndrome!
5. Click "Clear All Errors" to reset
6. Try "Inject Z Errors" → qubits turn blue
7. See X-checks (labeled "X") light up = syndrome!
```

## File Information

- **File**: `surface_code_interactive.html`
- **Size**: 89KB
- **Distance**: 7
- **Qubits**: 49
- **X-checks**: 24
- **Z-checks**: 24

Open the file in any modern web browser to use it!
