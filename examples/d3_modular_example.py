"""
Example: Creating Modular and Embeddable D3.js Visualizations

This example demonstrates how to refactor the CSS code visualization
to make it modular and embeddable in larger D3.js applications.

The key concepts are:
1. Separate data preparation from rendering
2. Export JavaScript modules that can be imported
3. Provide JSON data for custom integrations
4. Create reusable rendering functions
"""

import json
from pathlib import Path

# Import the existing visualization function
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qec.code_constructions.rotated_surface_code import RotatedSurfaceCode
from qec.utils.draw_css_code_d3 import prepare_css_code_visualization_data


def generate_embeddable_js_module(data, output_file, module_name="CSSCodeViz"):
    """
    Generate a JavaScript module that can be embedded in other applications.
    
    Usage:
        1. Include in HTML: <script src="output_file"></script>
        2. Call: CSSCodeViz.render("#svg-container")
        3. Update: CSSCodeViz.highlightErrors(["q0", "q5"])
    """
    nodes_json = json.dumps(data['nodes'], indent=2)
    edges_json = json.dumps(data['edges'], indent=2)
    config_json = json.dumps(data['config'], indent=2)
    
    js_content = f"""
// {module_name} - Embeddable CSS Code Visualization Module
// This module can be included in any D3.js application

(function(global) {{
    'use strict';
    
    const {module_name} = {{}};
    
    // Visualization data
    {module_name}.data = {{
        nodes: {nodes_json},
        edges: {edges_json},
        config: {config_json}
    }};
    
    // Core rendering function
    {module_name}.render = function(svgSelector, options) {{
        options = options || {{}};
        const data = this.data;
        const config = {{ ...data.config, ...options }};
        
        // Get or create SVG element
        let svg = d3.select(svgSelector);
        if (svg.empty()) {{
            console.error('SVG element not found:', svgSelector);
            return null;
        }}
        
        // Set dimensions
        svg.attr("width", config.width || 800)
           .attr("height", config.height || 600);
        
        // Clear existing content (for re-rendering)
        svg.selectAll("*").remove();
        
        // Draw edges first (bottom layer)
        const edges = svg.selectAll(".css-edge")
            .data(data.edges)
            .enter()
            .append("line")
            .attr("class", d => `css-edge css-edge-${{d.type}}`)
            .attr("x1", d => d.x1)
            .attr("y1", d => d.y1)
            .attr("x2", d => d.x2)
            .attr("y2", d => d.y2)
            .attr("stroke", d => d.color)
            .attr("stroke-width", d => d.width)
            .attr("stroke-dasharray", d => d.style === "dashed" ? "5,5" : "none");
        
        // Create node groups
        const nodes = svg.selectAll(".css-node")
            .data(data.nodes)
            .enter()
            .append("g")
            .attr("class", d => `css-node css-node-${{d.type}}`)
            .attr("transform", d => `translate(${{d.x}},${{d.y}})`)
            .attr("data-node-id", d => d.id);
        
        // Draw node shapes
        nodes.each(function(d) {{
            const node = d3.select(this);
            if (d.type === "qubit") {{
                node.append("circle")
                    .attr("r", d.radius)
                    .attr("fill", d.fill)
                    .attr("stroke", d.color)
                    .attr("stroke-width", 2);
            }} else {{
                node.append("rect")
                    .attr("x", -d.radius)
                    .attr("y", -d.radius)
                    .attr("width", d.radius * 2)
                    .attr("height", d.radius * 2)
                    .attr("fill", d.fill)
                    .attr("stroke", d.color)
                    .attr("stroke-width", 2);
            }}
        }});
        
        // Add labels if enabled
        if (config.showLabels !== false) {{
            nodes.append("text")
                .attr("class", "css-node-label")
                .attr("x", d => d.label_dx || 10)
                .attr("y", d => d.label_dy || -10)
                .attr("font-size", config.label_fontsize || 12)
                .attr("text-anchor", d => d.label_anchor || "start")
                .text(d => d.display_label || d.label || d.id);
        }}
        
        // Add tooltips
        nodes.append("title")
            .text(d => `${{d.tooltip_label || d.label || d.id}} (${{d.type_label || d.type}})`);
        
        return {{ svg, nodes, edges }};
    }};
    
    // Update node colors (useful for error visualization)
    {module_name}.updateNodeColors = function(svgSelector, nodeColorMap) {{
        const svg = d3.select(svgSelector);
        Object.keys(nodeColorMap).forEach(nodeId => {{
            svg.select(`[data-node-id="${{nodeId}}"]`)
                .select("circle, rect")
                .transition()
                .duration(300)
                .attr("fill", nodeColorMap[nodeId]);
        }});
    }};
    
    // Highlight errors on specific nodes
    {module_name}.highlightErrors = function(svgSelector, errorNodeIds, errorColor) {{
        errorColor = errorColor || "#ff0000";
        const colorMap = {{}};
        errorNodeIds.forEach(id => {{
            colorMap[id] = errorColor;
        }});
        this.updateNodeColors(svgSelector, colorMap);
    }};
    
    // Clear all error highlights
    {module_name}.clearErrors = function(svgSelector) {{
        const svg = d3.select(svgSelector);
        const data = this.data;
        svg.selectAll(".css-node")
            .select("circle, rect")
            .transition()
            .duration(300)
            .attr("fill", (d, i) => data.nodes[i].fill);
    }};
    
    // Toggle edge visibility by type
    {module_name}.toggleEdges = function(svgSelector, edgeType, visible) {{
        const svg = d3.select(svgSelector);
        svg.selectAll(`.css-edge-${{edgeType}}`)
            .transition()
            .duration(200)
            .style("opacity", visible ? 1 : 0);
    }};
    
    // Export to global scope
    global.{module_name} = {module_name};
    
    // Also support CommonJS and AMD
    if (typeof module !== 'undefined' && module.exports) {{
        module.exports = {module_name};
    }}
    
}})(typeof window !== 'undefined' ? window : this);
"""
    
    with open(output_file, 'w') as f:
        f.write(js_content)
    
    print(f"✓ Generated embeddable JavaScript module: {output_file}")
    print(f"  Usage: {module_name}.render('#svg-id')")
    print(f"  Features: highlightErrors(), toggleEdges(), updateNodeColors()")


def generate_integration_example_html(module_file, output_file):
    """
    Generate an example HTML file showing how to integrate the JS module
    into a larger application with custom controls.
    """
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Integrated CSS Code Visualization Example</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="{Path(module_file).name}"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            margin-top: 0;
        }}
        .controls {{
            margin: 20px 0;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 4px;
            border: 1px solid #ddd;
        }}
        .control-group {{
            margin: 10px 0;
        }}
        button {{
            padding: 10px 20px;
            margin-right: 10px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        button:hover {{
            background: #45a049;
        }}
        button.secondary {{
            background: #2196F3;
        }}
        button.secondary:hover {{
            background: #0b7dda;
        }}
        label {{
            margin-right: 15px;
            cursor: pointer;
        }}
        input[type="checkbox"] {{
            margin-right: 5px;
        }}
        #svg-container {{
            margin-top: 20px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
            display: inline-block;
        }}
        .info {{
            margin-top: 20px;
            padding: 15px;
            background: #e3f2fd;
            border-left: 4px solid #2196F3;
            border-radius: 4px;
        }}
        .info h3 {{
            margin-top: 0;
        }}
        .code-example {{
            background: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
            margin-top: 10px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Modular CSS Code Visualization</h1>
        <p>This example shows how to embed the CSS code visualization in a larger application.</p>
        
        <div class="controls">
            <h3>Interactive Controls</h3>
            
            <div class="control-group">
                <strong>Error Simulation:</strong><br>
                <button onclick="simulateRandomError()">Add Random Error</button>
                <button onclick="simulateMultipleErrors()">Add Multiple Errors</button>
                <button class="secondary" onclick="clearAllErrors()">Clear All Errors</button>
            </div>
            
            <div class="control-group">
                <strong>Display Options:</strong><br>
                <label>
                    <input type="checkbox" id="show-x-edges" checked onchange="toggleXEdges()">
                    Show X-check Edges
                </label>
                <label>
                    <input type="checkbox" id="show-z-edges" checked onchange="toggleZEdges()">
                    Show Z-check Edges
                </label>
                <label>
                    <input type="checkbox" id="show-labels" checked onchange="toggleLabels()">
                    Show Labels
                </label>
            </div>
        </div>
        
        <div id="svg-container">
            <svg id="main-visualization"></svg>
        </div>
        
        <div class="info">
            <h3>📚 Integration Guide</h3>
            <p>This visualization is powered by a modular JavaScript file that can be embedded anywhere:</p>
            <div class="code-example">
&lt;!-- 1. Include D3.js --&gt;<br>
&lt;script src="https://d3js.org/d3.v7.min.js"&gt;&lt;/script&gt;<br><br>
&lt;!-- 2. Include the CSS Code Viz module --&gt;<br>
&lt;script src="{Path(module_file).name}"&gt;&lt;/script&gt;<br><br>
&lt;!-- 3. Render in your application --&gt;<br>
&lt;script&gt;<br>
&nbsp;&nbsp;// Initial render<br>
&nbsp;&nbsp;CSSCodeViz.render('#main-visualization');<br><br>
&nbsp;&nbsp;// Highlight errors on specific qubits<br>
&nbsp;&nbsp;CSSCodeViz.highlightErrors('#main-visualization', ['q0', 'q5']);<br><br>
&nbsp;&nbsp;// Toggle edge types<br>
&nbsp;&nbsp;CSSCodeViz.toggleEdges('#main-visualization', 'x_check', false);<br>
&lt;/script&gt;
            </div>
            <p><strong>Features:</strong> Error highlighting, edge toggling, custom styling, dynamic updates</p>
        </div>
    </div>
    
    <script>
        // Initialize the visualization
        let vizElements = CSSCodeViz.render('#main-visualization');
        let errorNodes = [];
        
        function simulateRandomError() {{
            const nodes = CSSCodeViz.data.nodes.filter(n => n.type === 'qubit');
            const randomNode = nodes[Math.floor(Math.random() * nodes.length)];
            errorNodes.push(randomNode.id);
            CSSCodeViz.highlightErrors('#main-visualization', errorNodes, '#ff0000');
            console.log('Error added to:', randomNode.id);
        }}
        
        function simulateMultipleErrors() {{
            const nodes = CSSCodeViz.data.nodes.filter(n => n.type === 'qubit');
            const numErrors = Math.min(3, nodes.length);
            for (let i = 0; i < numErrors; i++) {{
                const randomNode = nodes[Math.floor(Math.random() * nodes.length)];
                if (!errorNodes.includes(randomNode.id)) {{
                    errorNodes.push(randomNode.id);
                }}
            }}
            CSSCodeViz.highlightErrors('#main-visualization', errorNodes, '#ff0000');
            console.log('Errors on:', errorNodes);
        }}
        
        function clearAllErrors() {{
            errorNodes = [];
            CSSCodeViz.clearErrors('#main-visualization');
            console.log('All errors cleared');
        }}
        
        function toggleXEdges() {{
            const checked = document.getElementById('show-x-edges').checked;
            CSSCodeViz.toggleEdges('#main-visualization', 'x_check', checked);
        }}
        
        function toggleZEdges() {{
            const checked = document.getElementById('show-z-edges').checked;
            CSSCodeViz.toggleEdges('#main-visualization', 'z_check', checked);
        }}
        
        function toggleLabels() {{
            const checked = document.getElementById('show-labels').checked;
            const svg = d3.select('#main-visualization');
            svg.selectAll('.css-node-label')
                .transition()
                .duration(200)
                .style('opacity', checked ? 1 : 0);
        }}
    </script>
</body>
</html>"""
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"✓ Generated integration example: {output_file}")
    print(f"  Open in browser to see the modular visualization in action")


def generate_json_data_export(data, output_file):
    """Export pure JSON data for custom integrations."""
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Exported JSON data: {output_file}")
    print(f"  Use this data in any framework (React, Vue, Angular, etc.)")


def main():
    print("=" * 70)
    print("CSS Code D3.js Visualization - Modular Example")
    print("=" * 70)
    print()
    
    # Create a simple code instance
    print("Creating a rotated surface code (distance 5)...")
    code = RotatedSurfaceCode(5)
    
    # Prepare visualization data
    print("Preparing visualization data...")
    data = prepare_css_code_visualization_data(
        code,
        qubit_radius=12,
        check_radius=15,
        spacing=80,
        show_labels=True,
        label_fontsize=10,
        qubit_fill="#0091ff",
        x_check_fill="white",
        z_check_fill="white",
        x_edge_color="black",
        z_edge_color="black",
        edge_width=3
    )
    
    # Create output directory
    output_dir = Path("d3_modular_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating modular outputs in '{output_dir}/'...\n")
    
    # 1. Generate embeddable JavaScript module
    js_module = output_dir / "css_code_viz.js"
    generate_embeddable_js_module(data, js_module, module_name="CSSCodeViz")
    
    # 2. Generate integration example HTML
    example_html = output_dir / "integration_example.html"
    generate_integration_example_html(js_module, example_html)
    
    # 3. Export JSON data
    json_file = output_dir / "code_data.json"
    generate_json_data_export(data, json_file)
    
    print("\n" + "=" * 70)
    print("✅ All files generated successfully!")
    print("=" * 70)
    print("\nWhat you can do now:")
    print(f"  1. Open {example_html} in a browser to see the interactive demo")
    print(f"  2. Include {js_module} in your own HTML/JS application")
    print(f"  3. Use {json_file} data with React, Vue, or any framework")
    print("\nKey features of the modular approach:")
    print("  • Separate data from presentation")
    print("  • Embeddable JavaScript module with API")
    print("  • Error highlighting and dynamic updates")
    print("  • Easy integration into larger applications")
    print()


if __name__ == "__main__":
    main()
