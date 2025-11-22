# Modular D3.js CSS Code Visualizations

This directory contains examples of how to make CSS code visualizations **modular and embeddable** for integration into larger applications.

## 📁 Files Generated

1. **`css_code_viz.js`** - Embeddable JavaScript module
2. **`integration_example.html`** - Interactive demo showing how to use the module
3. **`code_data.json`** - Raw JSON data for custom frameworks

## 🎯 Why Modular?

The original `draw_css_code_d3.py` creates standalone HTML files, which are great for quick visualizations but difficult to:
- Embed in larger web applications
- Add custom interactive controls (error simulation, toggles, etc.)
- Integrate with modern frameworks (React, Vue, Angular)
- Reuse across multiple pages

The modular approach solves these problems by **separating data from presentation**.

## 🚀 Quick Start

### Option 1: Use the JavaScript Module

```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="css_code_viz.js"></script>
</head>
<body>
    <svg id="my-visualization"></svg>
    
    <script>
        // Render the visualization
        CSSCodeViz.render('#my-visualization');
        
        // Highlight errors on specific qubits
        CSSCodeViz.highlightErrors('#my-visualization', ['q0', 'q5', 'q12']);
        
        // Toggle edge visibility
        CSSCodeViz.toggleEdges('#my-visualization', 'x_check', false);
    </script>
</body>
</html>
```

### Option 2: Use JSON Data with Any Framework

**React Example:**
```jsx
import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import codeData from './code_data.json';

function CSSCodeVisualization() {
    const svgRef = useRef();
    
    useEffect(() => {
        const svg = d3.select(svgRef.current)
            .attr('width', codeData.config.width)
            .attr('height', codeData.config.height);
        
        // Draw edges
        svg.selectAll('line')
            .data(codeData.edges)
            .enter().append('line')
            .attr('x1', d => d.x1)
            .attr('y1', d => d.y1)
            .attr('x2', d => d.x2)
            .attr('y2', d => d.y2)
            .attr('stroke', d => d.color);
        
        // Draw nodes
        // ... (see full example in integration_example.html)
    }, []);
    
    return <svg ref={svgRef}></svg>;
}
```

### Option 3: Extend with Custom Controls

```javascript
// Add your own custom control functions
function addCustomErrorPattern() {
    // Define your error pattern
    const errorNodes = ['q0', 'q1', 'q2'];
    CSSCodeViz.highlightErrors('#my-viz', errorNodes, '#ff0000');
}

function runDecoder() {
    // Your decoder logic here
    const correctedNodes = yourDecoder(CSSCodeViz.data.nodes);
    CSSCodeViz.updateNodeColors('#my-viz', correctedNodes);
}
```

## 📚 API Reference

### `CSSCodeViz.render(selector, options)`
Render the visualization in the specified SVG element.

**Parameters:**
- `selector` (string): CSS selector for SVG element (e.g., '#my-svg')
- `options` (object, optional): Configuration options
  - `showLabels` (boolean): Show/hide node labels
  - `width` (number): Override SVG width
  - `height` (number): Override SVG height

**Returns:** Object with `{svg, nodes, edges}` D3 selections

**Example:**
```javascript
const viz = CSSCodeViz.render('#visualization', {
    showLabels: true,
    width: 1000,
    height: 800
});
```

### `CSSCodeViz.highlightErrors(selector, nodeIds, color)`
Highlight specific nodes to visualize errors.

**Parameters:**
- `selector` (string): CSS selector for SVG element
- `nodeIds` (array): Array of node IDs to highlight (e.g., ['q0', 'q5'])
- `color` (string, optional): Error color (default: '#ff0000')

**Example:**
```javascript
CSSCodeViz.highlightErrors('#viz', ['q0', 'q3', 'q7'], '#ff6b6b');
```

### `CSSCodeViz.clearErrors(selector)`
Remove all error highlights.

**Example:**
```javascript
CSSCodeViz.clearErrors('#visualization');
```

### `CSSCodeViz.toggleEdges(selector, edgeType, visible)`
Show or hide edges by type.

**Parameters:**
- `selector` (string): CSS selector for SVG element
- `edgeType` (string): 'x_check' or 'z_check'
- `visible` (boolean): true to show, false to hide

**Example:**
```javascript
// Hide X-check edges
CSSCodeViz.toggleEdges('#viz', 'x_check', false);

// Show Z-check edges
CSSCodeViz.toggleEdges('#viz', 'z_check', true);
```

### `CSSCodeViz.updateNodeColors(selector, colorMap)`
Update colors for multiple nodes at once.

**Parameters:**
- `selector` (string): CSS selector for SVG element
- `colorMap` (object): Mapping of node IDs to colors

**Example:**
```javascript
CSSCodeViz.updateNodeColors('#viz', {
    'q0': '#ff0000',  // Error on qubit 0
    'q5': '#ff0000',  // Error on qubit 5
    'x3': '#00ff00',  // Triggered X-check 3
    'z7': '#00ff00'   // Triggered Z-check 7
});
```

## 🔧 Data Structure

The JSON data file contains:

```json
{
    "nodes": [
        {
            "id": "q0",
            "x": 400,
            "y": 300,
            "type": "qubit",
            "radius": 12,
            "color": "#000000",
            "fill": "#0091ff",
            "label": "Q_0"
        },
        // ... more nodes
    ],
    "edges": [
        {
            "x1": 400,
            "y1": 300,
            "x2": 320,
            "y2": 220,
            "color": "#000000",
            "style": "solid",
            "width": 3,
            "type": "x_check"
        },
        // ... more edges
    ],
    "config": {
        "width": 800,
        "height": 600,
        "label_fontsize": 10
    },
    "metadata": {
        "num_qubits": 13,
        "num_x_checks": 12,
        "num_z_checks": 12
    }
}
```

## 💡 Use Cases

### 1. Interactive Error Correction Simulator
```javascript
// Simulate errors and show decoder corrections
function simulateErrorCorrection() {
    // Add random errors
    const errors = generateRandomErrors(3);
    CSSCodeViz.highlightErrors('#viz', errors, '#ff0000');
    
    // Run your decoder
    const syndrome = computeSyndrome(errors);
    highlightSyndrome(syndrome);
    
    // Show correction
    const correction = runDecoder(syndrome);
    setTimeout(() => {
        CSSCodeViz.updateNodeColors('#viz', correction);
    }, 1000);
}
```

### 2. Multi-Code Comparison Dashboard
```html
<div class="dashboard">
    <div class="code-panel">
        <h3>Surface Code d=5</h3>
        <svg id="surface-code"></svg>
    </div>
    <div class="code-panel">
        <h3>Color Code d=5</h3>
        <svg id="color-code"></svg>
    </div>
</div>

<script>
    // Load and render multiple codes
    SurfaceCodeViz.render('#surface-code');
    ColorCodeViz.render('#color-code');
</script>
```

### 3. Educational Interactive Tool
```javascript
// Step-by-step error correction tutorial
const tutorial = {
    step1: () => {
        CSSCodeViz.render('#viz');
        showMessage("This is a CSS code Tanner graph");
    },
    step2: () => {
        CSSCodeViz.highlightErrors('#viz', ['q5'], '#ff0000');
        showMessage("An error occurred on qubit 5");
    },
    step3: () => {
        CSSCodeViz.updateNodeColors('#viz', {
            'x3': '#ffff00',
            'z7': '#ffff00'
        });
        showMessage("These stabilizers are triggered");
    }
};
```

## 🔄 Generating Your Own Modules

Use the Python example to generate modules from your own codes:

```python
from qec.code_constructions.rotated_surface_code import RotatedSurfaceCode
from qec.utils.draw_css_code_d3 import prepare_css_code_visualization_data
from examples.d3_modular_example import generate_embeddable_js_module

# Create your code
code = RotatedSurfaceCode(7)  # Distance 7

# Prepare data
data = prepare_css_code_visualization_data(
    code,
    qubit_radius=10,
    spacing=70,
    show_labels=True
)

# Generate embeddable module
generate_embeddable_js_module(
    data, 
    "my_code_viz.js", 
    module_name="MyCodeViz"
)
```

## 🌐 Integration with Modern Frameworks

### React/Next.js
```jsx
import { useEffect, useRef } from 'react';
import codeData from './code_data.json';

export function CodeVisualization() {
    const svgRef = useRef(null);
    
    useEffect(() => {
        if (typeof window !== 'undefined') {
            // Import D3 and module
            import('d3').then(d3 => {
                import('./css_code_viz.js').then(module => {
                    module.CSSCodeViz.render(svgRef.current);
                });
            });
        }
    }, []);
    
    return <svg ref={svgRef} />;
}
```

### Vue.js
```vue
<template>
    <svg ref="visualization"></svg>
</template>

<script>
import * as d3 from 'd3';
import codeData from './code_data.json';

export default {
    mounted() {
        // Initialize visualization
        this.renderCode();
    },
    methods: {
        renderCode() {
            // Use the JSON data to render
            // See integration_example.html for full code
        }
    }
}
</script>
```

## 📖 Further Resources

- **Full Example**: Open `integration_example.html` in a browser
- **Source Code**: See `examples/d3_modular_example.py`
- **Original Function**: `src/qec/utils/draw_css_code_d3.py`

## 🤝 Contributing

To add new features to the modular API:

1. Edit the JavaScript template in `generate_embeddable_js_module()`
2. Add your new method to the module export
3. Document it in this README
4. Create an example in `integration_example.html`

## ✨ Summary

The modular approach provides:
- ✅ **Reusability**: One module, many applications
- ✅ **Flexibility**: Easy to add custom controls and features
- ✅ **Framework Agnostic**: Works with React, Vue, Angular, vanilla JS
- ✅ **Maintainability**: Separate data preparation from rendering
- ✅ **Extensibility**: Simple API for adding new functionality

**Key Takeaway**: By separating data (`code_data.json`) from rendering (`css_code_viz.js`), you can embed your D3 visualizations anywhere and add any controls you need!
