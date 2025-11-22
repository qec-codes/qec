# Making D3 Visualizations Modular and Embeddable

## Summary

I've refactored your CSS code D3.js visualization to be **modular and embeddable**, allowing you to:

1. **Embed in larger D3.js applications** with custom controls
2. **Add interactive features** like error simulation, toggles, etc.
3. **Integrate with modern frameworks** (React, Vue, Angular)
4. **Reuse across multiple pages** without duplication

## What Was Created

### 1. Core Components

**`examples/d3_modular_example.py`**
- Main example script showing the modular approach
- Functions to generate embeddable JavaScript modules
- Functions to export JSON data
- Comprehensive integration examples

### 2. Generated Output (`d3_modular_output/`)

**`css_code_viz.js`** - Embeddable JavaScript Module
```javascript
// Include in any HTML page
<script src="css_code_viz.js"></script>

// Then use the API
CSSCodeViz.render('#my-svg');
CSSCodeViz.highlightErrors('#my-svg', ['q0', 'q5']);
CSSCodeViz.toggleEdges('#my-svg', 'x_check', false);
```

**`integration_example.html`** - Interactive Demo
- Shows the module in action
- Includes error simulation controls
- Toggle buttons for X/Z edges
- Label visibility controls

**`code_data.json`** - Raw Data Export
- Pure JSON data structure
- Can be used with any framework
- No D3 dependencies

**`README.md`** - Complete Documentation
- API reference
- Usage examples
- Integration guides for React, Vue, etc.

## Key Architecture Changes

### Before (Monolithic)
```python
draw_css_code_tanner_graph_d3(code, "output.html")
# → Single HTML file
# → Hard to customize
# → Not reusable
```

### After (Modular)

**Step 1: Prepare Data**
```python
data = prepare_css_code_visualization_data(code)
# → Returns structured dict with nodes, edges, config
# → Separates data from presentation
# → Reusable for different outputs
```

**Step 2: Choose Output Format**
```python
# Option A: JavaScript module
generate_embeddable_js_module(data, "viz.js")

# Option B: Standalone HTML (backward compatible)
generate_standalone_html(code, "standalone.html")

# Option C: Interactive HTML with controls
generate_html_with_controls(code, "interactive.html")

# Option D: JSON export for custom use
with open("data.json", "w") as f:
    json.dump(data, f)
```

## How to Use the Modular Approach

### Basic Usage

```bash
# Run the example to generate modular outputs
cd /home/joschka/github/qec
python examples/d3_modular_example.py

# This creates:
# - d3_modular_output/css_code_viz.js
# - d3_modular_output/integration_example.html
# - d3_modular_output/code_data.json
# - d3_modular_output/README.md
```

### Embed in Your Website

```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="css_code_viz.js"></script>
</head>
<body>
    <h1>My Quantum Error Correction App</h1>
    
    <!-- Your custom controls -->
    <button onclick="simulateError()">Add Error</button>
    <button onclick="runMyDecoder()">Run Decoder</button>
    
    <!-- The visualization -->
    <svg id="code-viz"></svg>
    
    <script>
        // Initialize
        CSSCodeViz.render('#code-viz');
        
        // Your custom functions
        function simulateError() {
            const errors = ['q0', 'q5', 'q12'];
            CSSCodeViz.highlightErrors('#code-viz', errors);
        }
        
        function runMyDecoder() {
            // Your decoder logic here
            const corrections = myDecoder();
            CSSCodeViz.updateNodeColors('#code-viz', corrections);
        }
    </script>
</body>
</html>
```

### Use in a Larger D3.js Dashboard

```html
<div id="dashboard">
    <div class="panel">
        <h3>Surface Code</h3>
        <svg id="surface-code-viz"></svg>
    </div>
    <div class="panel">
        <h3>Color Code</h3>
        <svg id="color-code-viz"></svg>
    </div>
    <div class="panel">
        <h3>Syndrome Graph</h3>
        <svg id="syndrome-graph"></svg>
    </div>
</div>

<script>
    // Render multiple codes in one dashboard
    SurfaceCodeViz.render('#surface-code-viz');
    ColorCodeViz.render('#color-code-viz');
    
    // Add custom D3 visualization in the same dashboard
    renderSyndromeGraph('#syndrome-graph');
</script>
```

### Integration with React

```jsx
import { useEffect, useRef } from 'react';
import codeData from './code_data.json';
import * as d3 from 'd3';

function CSSCodeVisualization({ errors = [] }) {
    const svgRef = useRef();
    
    useEffect(() => {
        // Load the module
        import('./css_code_viz.js').then(module => {
            module.CSSCodeViz.render(svgRef.current);
        });
    }, []);
    
    useEffect(() => {
        // Update when errors change
        if (window.CSSCodeViz) {
            window.CSSCodeViz.highlightErrors(
                svgRef.current, 
                errors
            );
        }
    }, [errors]);
    
    return <svg ref={svgRef} />;
}
```

## API Reference

The generated JavaScript module provides:

| Method | Description |
|--------|-------------|
| `render(selector, options)` | Render visualization in SVG element |
| `highlightErrors(selector, nodeIds, color)` | Highlight specific nodes as errors |
| `clearErrors(selector)` | Remove all error highlights |
| `toggleEdges(selector, type, visible)` | Show/hide X or Z edges |
| `updateNodeColors(selector, colorMap)` | Update multiple node colors |

See `d3_modular_output/README.md` for detailed API documentation.

## Benefits

### ✅ Reusability
- Generate one JavaScript module
- Use it across multiple pages
- No code duplication

### ✅ Flexibility
- Add custom controls easily
- Integrate with your own logic
- Extend with new features

### ✅ Framework Agnostic
- Works with vanilla JavaScript
- Compatible with React, Vue, Angular
- Can use with any D3.js application

### ✅ Separation of Concerns
- Data preparation (Python)
- Rendering logic (JavaScript)
- Presentation (HTML/CSS)

### ✅ Progressive Enhancement
- Start with basic visualization
- Add features incrementally
- Customize as needed

## Next Steps

### For Your Project

1. **Test the Example**
   ```bash
   cd /home/joschka/github/qec/d3_modular_output
   # Open integration_example.html in your browser
   ```

2. **Generate Modules for Your Codes**
   ```python
   from examples.d3_modular_example import generate_embeddable_js_module
   from qec.utils.draw_css_code_d3 import prepare_css_code_visualization_data
   
   # Your code
   data = prepare_css_code_visualization_data(my_code)
   generate_embeddable_js_module(data, "my_viz.js", "MyViz")
   ```

3. **Customize the Module**
   - Edit the JavaScript template in `generate_embeddable_js_module()`
   - Add new methods for your specific needs
   - Extend with domain-specific functionality

4. **Integrate into Your Application**
   - Include the generated `.js` file
   - Call the API methods
   - Add your custom controls

### Recommended Enhancements

**Add to the Module:**
- Syndrome computation visualization
- Decoder animation
- Error pattern library
- Export to SVG/PNG
- Zoom and pan controls
- Node selection/highlighting

**Example Custom Feature:**
```javascript
// Add to the module template
CSSCodeViz.animateDecoding = function(selector, steps) {
    steps.forEach((step, i) => {
        setTimeout(() => {
            this.updateNodeColors(selector, step.colors);
        }, i * 1000);
    });
};
```

## Files Modified/Created

### New Files
- ✨ `examples/d3_modular_example.py` - Main example script
- ✨ `d3_modular_output/css_code_viz.js` - Embeddable module
- ✨ `d3_modular_output/integration_example.html` - Interactive demo
- ✨ `d3_modular_output/code_data.json` - JSON data export
- ✨ `d3_modular_output/README.md` - Documentation
- ✨ `MODULAR_VISUALIZATION_GUIDE.md` - This file

### Modified Files
- `src/qec/utils/draw_css_code_d3.py` - Added `prepare_css_code_visualization_data()` function

The original `draw_css_code_tanner_graph_d3()` function remains unchanged for backward compatibility.

## Questions?

The modular approach is designed to be:
- **Easy to understand**: Clear separation of data and rendering
- **Easy to use**: Simple API with sensible defaults
- **Easy to extend**: Add new features by editing the template
- **Easy to integrate**: Works with any web technology

Check the README in `d3_modular_output/` for more examples and detailed API documentation!
