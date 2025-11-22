# Quick Reference: Modular D3.js Visualizations

## 🎯 The Problem You Had
Your D3 code creates standalone HTML files that are hard to:
- Embed in larger applications
- Add custom controls to (error simulation, toggles, etc.)
- Integrate with modern frameworks (React, Vue, etc.)
- Reuse across multiple pages

## ✅ The Solution
**Separate data from presentation** by creating modular, embeddable components.

## 📦 What Was Generated

```
d3_modular_output/
├── css_code_viz.js          # Embeddable JavaScript module
├── integration_example.html  # Interactive demo
├── code_data.json           # Raw JSON data
└── README.md                # Full documentation
```

## 🚀 Quick Start

### 1. Generate Modular Output
```bash
cd /home/joschka/github/qec
python examples/d3_modular_example.py
```

### 2. Use in Your HTML
```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="css_code_viz.js"></script>
</head>
<body>
    <svg id="my-viz"></svg>
    <button onclick="addError()">Simulate Error</button>
    
    <script>
        // Render the code
        CSSCodeViz.render('#my-viz');
        
        // Add your custom function
        function addError() {
            CSSCodeViz.highlightErrors('#my-viz', ['q0', 'q5']);
        }
    </script>
</body>
</html>
```

## 🎨 Common Use Cases

### Add Error Simulation
```javascript
function simulateRandomError() {
    const nodes = CSSCodeViz.data.nodes.filter(n => n.type === 'qubit');
    const random = nodes[Math.floor(Math.random() * nodes.length)];
    CSSCodeViz.highlightErrors('#my-viz', [random.id], '#ff0000');
}
```

### Toggle X/Z Edges
```javascript
// Hide X-check edges
CSSCodeViz.toggleEdges('#my-viz', 'x_check', false);

// Show Z-check edges
CSSCodeViz.toggleEdges('#my-viz', 'z_check', true);
```

### Visualize Decoder Output
```javascript
function showDecoding(corrections) {
    const colorMap = {};
    corrections.forEach(nodeId => {
        colorMap[nodeId] = '#00ff00';  // Green for corrected
    });
    CSSCodeViz.updateNodeColors('#my-viz', colorMap);
}
```

### Multiple Codes in One Page
```html
<div class="dashboard">
    <svg id="code1"></svg>
    <svg id="code2"></svg>
    <svg id="code3"></svg>
</div>

<script>
    Code1Viz.render('#code1');
    Code2Viz.render('#code2');
    Code3Viz.render('#code3');
</script>
```

## 🔧 API at a Glance

| Method | What It Does |
|--------|-------------|
| `.render(selector)` | Draw the code visualization |
| `.highlightErrors(selector, ids)` | Highlight nodes in red |
| `.clearErrors(selector)` | Remove all highlights |
| `.toggleEdges(selector, type, show)` | Show/hide edge types |
| `.updateNodeColors(selector, map)` | Update multiple node colors |

## 📊 Data Structure

The JSON file has this structure:
```javascript
{
    "nodes": [
        {
            "id": "q0",           // Node identifier
            "x": 100, "y": 200,   // Position
            "type": "qubit",      // qubit, x_check, z_check
            "fill": "#0091ff"     // Color
        }
        // ... more nodes
    ],
    "edges": [
        {
            "x1": 100, "y1": 200,
            "x2": 150, "y2": 250,
            "type": "x_check",    // x_check or z_check
            "color": "#000000"
        }
        // ... more edges
    ],
    "config": {
        "width": 800,
        "height": 600
    }
}
```

## 🎓 How It Works

### Step 1: Python Generates Data
```python
from qec.utils.draw_css_code_d3 import prepare_css_code_visualization_data

data = prepare_css_code_visualization_data(code)
# → Returns: {nodes, edges, config, metadata}
```

### Step 2: Export to JavaScript Module
```python
from examples.d3_modular_example import generate_embeddable_js_module

generate_embeddable_js_module(data, "viz.js", module_name="MyViz")
# → Creates reusable JavaScript file
```

### Step 3: Use Anywhere
```html
<script src="viz.js"></script>
<script>
    MyViz.render('#svg-id');
</script>
```

## 🌐 Framework Integration

### React
```jsx
import { useEffect, useRef } from 'react';

function CodeViz() {
    const svgRef = useRef();
    
    useEffect(() => {
        import('./css_code_viz.js').then(mod => {
            mod.CSSCodeViz.render(svgRef.current);
        });
    }, []);
    
    return <svg ref={svgRef} />;
}
```

### Vue
```vue
<template>
    <svg ref="viz"></svg>
</template>

<script>
export default {
    mounted() {
        import('./css_code_viz.js').then(mod => {
            mod.CSSCodeViz.render(this.$refs.viz);
        });
    }
}
</script>
```

## 📚 Documentation

- **Full Guide**: `MODULAR_VISUALIZATION_GUIDE.md`
- **API Reference**: `d3_modular_output/README.md`
- **Example Code**: `examples/d3_modular_example.py`
- **Interactive Demo**: `d3_modular_output/integration_example.html`

## ⚡ Key Benefits

1. **Reusable**: Generate once, use everywhere
2. **Flexible**: Add any controls you want
3. **Embeddable**: Works in any HTML page
4. **Framework-friendly**: Compatible with React, Vue, Angular
5. **Data-driven**: Separate data from presentation

## 🎯 Bottom Line

Instead of generating standalone HTML files, you now have:
- A JavaScript module you can embed anywhere
- Clean JSON data for any framework
- Full control over interactivity and styling
- Easy integration into larger D3.js applications

**Just include the `.js` file and call the API!** 🚀
