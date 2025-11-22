"""
Generate an interactive surface code website with error injection and syndrome calculation.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
from qec.code_constructions.rotated_surface_code import RotatedSurfaceCode
from qec.code_constructions.surface_code import SurfaceCode
from qec.utils.draw_css_code_d3 import prepare_css_code_visualization_data


def prepare_code_data(code_class, distance):
    """Prepare visualization data and matrices for a specific code."""
    code = code_class(distance)
    
    data = prepare_css_code_visualization_data(
        code,
        qubit_radius=16,
        check_radius=16,
        spacing=80,
        show_labels=False,
        label_fontsize=12,
        qubit_fill="white",
        qubit_color="black",
        x_check_fill="white",
        x_check_color="black",
        z_check_fill="white",
        z_check_color="black",
        x_edge_color="#666666",
        z_edge_color="#666666",
        edge_width=2,
        margin=(50, 50, 50, 50)
    )
    
    # Get the parity check matrices
    Hx = code.x_stabilizer_matrix.toarray().tolist()
    Hz = code.z_stabilizer_matrix.toarray().tolist()
    
    # Create mapping of node IDs to matrix indices
    qubit_nodes = [node for node in data['nodes'] if node['type'] == 'qubit']
    x_check_nodes = [node for node in data['nodes'] if node['type'] == 'x_check']
    z_check_nodes = [node for node in data['nodes'] if node['type'] == 'z_check']
    
    qubit_id_to_index = {node['id']: i for i, node in enumerate(qubit_nodes)}
    x_check_id_to_index = {node['id']: i for i, node in enumerate(x_check_nodes)}
    z_check_id_to_index = {node['id']: i for i, node in enumerate(z_check_nodes)}
    
    return {
        'nodes': data['nodes'],
        'edges': data['edges'],
        'config': data['config'],
        'metadata': data['metadata'],
        'Hx': Hx,
        'Hz': Hz,
        'qubit_id_to_index': qubit_id_to_index,
        'x_check_id_to_index': x_check_id_to_index,
        'z_check_id_to_index': z_check_id_to_index
    }


def generate_interactive_surface_code_website(output_file="surface_code_interactive.html"):
    """
    Generate an interactive website for surface codes with error injection and syndrome calculation.
    
    Features:
    - Toggle between RotatedSurfaceCode and SurfaceCode
    - Select distance from dropdown (3-11)
    - Show/hide X-type and Z-type edges and nodes
    - Click to inject X, Y, and Z errors
    - Real-time syndrome calculation in JavaScript
    - Modern UI with Tailwind CSS
    """
    
    # Generate all code variants
    print("Generating code data for all variants...")
    all_codes_data = {}
    
    # Rotated Surface Code: odd distances 3, 5, 7, 9, 11
    print("  - Rotated Surface Codes (d=3,5,7,9,11)...")
    for d in [3, 5, 7, 9, 11]:
        print(f"    Distance {d}...")
        all_codes_data[f"rotated_{d}"] = prepare_code_data(RotatedSurfaceCode, d)
    
    # Surface Code: all distances 3-11
    print("  - Surface Codes (d=3-11)...")
    for d in range(3, 12):
        print(f"    Distance {d}...")
        all_codes_data[f"surface_{d}"] = prepare_code_data(SurfaceCode, d)
    
    # Serialize all code data to JSON
    all_codes_json = json.dumps(all_codes_data)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Surface Code Visualizer</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .css-node {{
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .css-node:hover {{
            opacity: 0.7;
        }}
        .css-node.error {{
            filter: drop-shadow(0 0 6px rgba(0, 0, 0, 0.4));
        }}
        .css-node.syndrome {{
            /* Syndrome styling applied via fill color change */
        }}
        .css-edge {{
            transition: opacity 0.3s ease;
        }}
        .toggle-btn {{
            transition: all 0.2s ease;
        }}
        .toggle-btn.active {{
            transform: scale(1.05);
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.5);
        }}
        .node-label {{
            font-family: 'Courier New', monospace;
            font-weight: bold;
            font-size: 12px;
            pointer-events: none;
            user-select: none;
            fill: black;
        }}
        .error-label {{
            font-family: 'Courier New', monospace;
            font-weight: bold;
            font-size: 14px;
            pointer-events: none;
            user-select: none;
            fill: white;
        }}
    </style>
</head>
<body class="bg-gray-900 text-gray-100">
    <div class="flex h-screen">
        <!-- Left Control Panel -->
        <div class="w-80 bg-gray-800 border-r border-gray-700 p-6 overflow-y-auto">
            <div class="mb-6">
                <h1 class="text-2xl font-bold text-blue-400 mb-2">Surface Code</h1>
                <p class="text-sm text-gray-400">Distance {distance}</p>
                <div class="mt-2 text-xs text-gray-500">
                    <div>Qubits: {data['metadata']['num_qubits']}</div>
                    <div>X-checks: {data['metadata']['num_x_checks']}</div>
                    <div>Z-checks: {data['metadata']['num_z_checks']}</div>
                </div>
            </div>

            <!-- Visibility Controls -->
            <div class="mb-8">
                <h2 class="text-lg font-semibold mb-4 text-gray-200">Display Options</h2>
                
                <div class="space-y-3">
                    <div class="bg-gray-700 rounded-lg p-4">
                        <label class="flex items-center justify-between cursor-pointer">
                            <div class="flex items-center space-x-3">
                                <div class="w-4 h-4 rounded bg-gray-400"></div>
                                <span class="text-sm font-medium">X-type</span>
                            </div>
                            <input type="checkbox" id="show-x-type" checked 
                                   class="w-5 h-5 text-blue-600 bg-gray-600 border-gray-500 rounded focus:ring-blue-500 focus:ring-2"
                                   onchange="toggleXType(this.checked)">
                        </label>
                        <p class="text-xs text-gray-400 mt-2 ml-7">X-checks and edges</p>
                    </div>

                    <div class="bg-gray-700 rounded-lg p-4">
                        <label class="flex items-center justify-between cursor-pointer">
                            <div class="flex items-center space-x-3">
                                <div class="w-4 h-4 rounded bg-gray-400"></div>
                                <span class="text-sm font-medium">Z-type</span>
                            </div>
                            <input type="checkbox" id="show-z-type" checked 
                                   class="w-5 h-5 text-blue-600 bg-gray-600 border-gray-500 rounded focus:ring-blue-500 focus:ring-2"
                                   onchange="toggleZType(this.checked)">
                        </label>
                        <p class="text-xs text-gray-400 mt-2 ml-7">Z-checks and edges</p>
                    </div>
                </div>
            </div>

            <!-- Error Injection Controls -->
            <div class="mb-8">
                <h2 class="text-lg font-semibold mb-4 text-gray-200">Error Injection</h2>
                
                <div class="space-y-3">
                    <button id="inject-x-btn" 
                            class="toggle-btn w-full bg-purple-600 hover:bg-purple-700 text-white font-semibold py-3 px-4 rounded-lg transition-all"
                            onclick="toggleErrorMode('X')">
                        <div class="flex items-center justify-center space-x-2">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path>
                            </svg>
                            <span>Inject X Errors</span>
                        </div>
                        <div class="text-xs mt-1 opacity-75">Click qubits to add</div>
                    </button>

                    <button id="inject-z-btn" 
                            class="toggle-btn w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-4 rounded-lg transition-all"
                            onclick="toggleErrorMode('Z')">
                        <div class="flex items-center justify-center space-x-2">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                            </svg>
                            <span>Inject Z Errors</span>
                        </div>
                        <div class="text-xs mt-1 opacity-75">Click qubits to add</div>
                    </button>

                    <button id="inject-y-btn" 
                            class="toggle-btn w-full bg-pink-600 hover:bg-pink-700 text-white font-semibold py-3 px-4 rounded-lg transition-all"
                            onclick="toggleErrorMode('Y')">
                        <div class="flex items-center justify-center space-x-2">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01"></path>
                            </svg>
                            <span>Inject Y Errors</span>
                        </div>
                        <div class="text-xs mt-1 opacity-75">Y = X + Z</div>
                    </button>
                </div>

                <div class="mt-4 p-3 bg-gray-700 rounded-lg text-xs text-gray-300">
                    <div class="font-semibold mb-2">Active Mode:</div>
                    <div id="mode-display" class="text-blue-400">None - Click a button above</div>
                </div>
                <div class="mt-3 p-2 bg-gray-700 rounded text-xs text-gray-400">
                    <div class="font-semibold mb-1">Legend:</div>
                    <div class="space-y-1">
                        <div class="flex items-center space-x-2">
                            <div class="w-3 h-3 rounded-full bg-red-500"></div>
                            <span>X Error</span>
                        </div>
                        <div class="flex items-center space-x-2">
                            <div class="w-3 h-3 rounded-full bg-blue-500"></div>
                            <span>Z Error</span>
                        </div>
                        <div class="flex items-center space-x-2">
                            <div class="w-3 h-3 rounded-full bg-pink-500"></div>
                            <span>Y Error</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Error Statistics -->
            <div class="mb-8">
                <h2 class="text-lg font-semibold mb-4 text-gray-200">Error Statistics</h2>
                <div class="bg-gray-700 rounded-lg p-4 space-y-2 text-sm">
                    <div class="flex justify-between">
                        <span class="text-gray-400">X Errors:</span>
                        <span id="x-error-count" class="font-semibold text-red-400">0</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Z Errors:</span>
                        <span id="z-error-count" class="font-semibold text-blue-400">0</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Y Errors:</span>
                        <span id="y-error-count" class="font-semibold text-pink-400">0</span>
                    </div>
                    <div class="border-t border-gray-600 pt-2 mt-2">
                        <div class="flex justify-between">
                            <span class="text-gray-400">X Syndrome:</span>
                            <span id="x-syndrome-count" class="font-semibold text-yellow-400">0</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-400">Z Syndrome:</span>
                            <span id="z-syndrome-count" class="font-semibold text-yellow-400">0</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Clear Button -->
            <div>
                <button onclick="clearAllErrors()" 
                        class="w-full bg-red-600 hover:bg-red-700 text-white font-semibold py-3 px-4 rounded-lg transition-all">
                    <div class="flex items-center justify-center space-x-2">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
                        </svg>
                        <span>Clear All Errors</span>
                    </div>
                </button>
            </div>

            <!-- Help Section -->
            <div class="mt-8 p-4 bg-blue-900 bg-opacity-30 border border-blue-700 rounded-lg text-xs">
                <h3 class="font-semibold text-blue-300 mb-2">How to use:</h3>
                <ol class="list-decimal list-inside space-y-1 text-gray-300">
                    <li>Toggle X or Z error mode</li>
                    <li>Click qubits to inject errors</li>
                    <li>Watch syndromes light up</li>
                    <li>Use checkboxes to hide/show elements</li>
                </ol>
            </div>
        </div>

        <!-- Main Visualization Area -->
        <div class="flex-1 bg-gray-900 flex items-center justify-center p-8">
            <div class="bg-white rounded-xl shadow-2xl p-8 border border-gray-300">
                <svg id="surface-code-viz"></svg>
            </div>
        </div>
    </div>

    <script>
        // Data
        const nodesData = {nodes_json};
        const edgesData = {edges_json};
        const config = {config_json};
        const Hx = {hx_json};  // Detects Z errors: Hx @ z_error % 2
        const Hz = {hz_json};  // Detects X errors: Hz @ x_error % 2
        const qubitIdToIndex = {qubit_mapping_json};
        const xCheckIdToIndex = {x_check_mapping_json};
        const zCheckIdToIndex = {z_check_mapping_json};

        // State
        let errorMode = null;  // 'X', 'Z', 'Y', or null
        let xErrors = new Set();  // Set of qubit IDs with X errors
        let zErrors = new Set();  // Set of qubit IDs with Z errors
        let yErrors = new Set();  // Set of qubit IDs with Y errors
        let xSyndrome = [];  // Indices of triggered X-checks
        let zSyndrome = [];  // Indices of triggered Z-checks

        // Initialize SVG
        const svg = d3.select("#surface-code-viz")
            .attr("width", config.width)
            .attr("height", config.height);

        // Draw edges
        const edgesGroup = svg.append("g").attr("class", "edges-group");
        
        const edges = edgesGroup.selectAll(".css-edge")
            .data(edgesData)
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

        // Draw nodes
        const nodesGroup = svg.append("g").attr("class", "nodes-group");
        
        const nodes = nodesGroup.selectAll(".css-node")
            .data(nodesData)
            .enter()
            .append("g")
            .attr("class", d => `css-node css-node-${{d.type}}`)
            .attr("transform", d => `translate(${{d.x}},${{d.y}})`)
            .attr("data-node-id", d => d.id)
            .attr("data-node-type", d => d.type)
            .on("click", handleNodeClick);

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

        // Add error labels to qubits (initially hidden)
        nodes.filter(d => d.type === "qubit").append("text")
            .attr("class", "error-label")
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "central")
            .style("visibility", "hidden");

        // Add tooltips (shows qubit index on hover)
        nodes.append("title")
            .text(d => `${{d.tooltip_label || d.id}} (${{d.type_label || d.type}})`);

        // Matrix multiplication: matrix @ vector % 2
        function matrixVectorMod2(matrix, vector) {{
            const result = [];
            for (let i = 0; i < matrix.length; i++) {{
                let sum = 0;
                for (let j = 0; j < matrix[i].length; j++) {{
                    sum += matrix[i][j] * vector[j];
                }}
                result.push(sum % 2);
            }}
            return result;
        }}

        // Calculate syndrome
        function calculateSyndrome() {{
            const numQubits = Object.keys(qubitIdToIndex).length;
            
            // Create error vectors
            const xErrorVector = new Array(numQubits).fill(0);
            const zErrorVector = new Array(numQubits).fill(0);
            
            // X errors (including Y errors, since Y = X + Z)
            xErrors.forEach(qubitId => {{
                const idx = qubitIdToIndex[qubitId];
                if (idx !== undefined) xErrorVector[idx] = 1;
            }});
            yErrors.forEach(qubitId => {{
                const idx = qubitIdToIndex[qubitId];
                if (idx !== undefined) xErrorVector[idx] = 1;
            }});
            
            // Z errors (including Y errors, since Y = X + Z)
            zErrors.forEach(qubitId => {{
                const idx = qubitIdToIndex[qubitId];
                if (idx !== undefined) zErrorVector[idx] = 1;
            }});
            yErrors.forEach(qubitId => {{
                const idx = qubitIdToIndex[qubitId];
                if (idx !== undefined) zErrorVector[idx] = 1;
            }});
            
            // Calculate syndromes
            // Hz detects X errors: Hz @ x_error % 2
            const zSyndromeVector = matrixVectorMod2(Hz, xErrorVector);
            
            // Hx detects Z errors: Hx @ z_error % 2
            const xSyndromeVector = matrixVectorMod2(Hx, zErrorVector);
            
            // Convert to triggered check indices
            zSyndrome = zSyndromeVector.map((val, idx) => val === 1 ? idx : -1).filter(idx => idx !== -1);
            xSyndrome = xSyndromeVector.map((val, idx) => val === 1 ? idx : -1).filter(idx => idx !== -1);
            
            updateSyndromeDisplay();
        }}

        // Update syndrome visualization
        function updateSyndromeDisplay() {{
            // Reset all check nodes to default colors
            svg.selectAll(".css-node-x_check rect").attr("fill", "white");
            svg.selectAll(".css-node-z_check rect").attr("fill", "white");
            svg.selectAll(".css-node").classed("syndrome", false);
            
            // Highlight X-checks (triggered by Z errors) in solid red
            xSyndrome.forEach(idx => {{
                const checkId = Object.keys(xCheckIdToIndex).find(id => xCheckIdToIndex[id] === idx);
                if (checkId) {{
                    const node = svg.select(`[data-node-id="${{checkId}}"]`);
                    node.classed("syndrome", true);
                    node.select("rect").attr("fill", "#ef4444");  // Solid red
                }}
            }});
            
            // Highlight Z-checks (triggered by X errors) in solid green
            zSyndrome.forEach(idx => {{
                const checkId = Object.keys(zCheckIdToIndex).find(id => zCheckIdToIndex[id] === idx);
                if (checkId) {{
                    const node = svg.select(`[data-node-id="${{checkId}}"]`);
                    node.classed("syndrome", true);
                    node.select("rect").attr("fill", "#10b981");  // Solid green
                }}
            }});
            
            // Update statistics
            document.getElementById('x-syndrome-count').textContent = xSyndrome.length;
            document.getElementById('z-syndrome-count').textContent = zSyndrome.length;
        }}

        // Handle node click
        function handleNodeClick(event, d) {{
            if (!errorMode || d.type !== 'qubit') return;
            
            const qubitId = d.id;
            
            if (errorMode === 'X') {{
                if (xErrors.has(qubitId)) {{
                    xErrors.delete(qubitId);
                }} else {{
                    xErrors.add(qubitId);
                }}
                document.getElementById('x-error-count').textContent = xErrors.size;
            }} else if (errorMode === 'Z') {{
                if (zErrors.has(qubitId)) {{
                    zErrors.delete(qubitId);
                }} else {{
                    zErrors.add(qubitId);
                }}
                document.getElementById('z-error-count').textContent = zErrors.size;
            }} else if (errorMode === 'Y') {{
                if (yErrors.has(qubitId)) {{
                    yErrors.delete(qubitId);
                }} else {{
                    yErrors.add(qubitId);
                }}
                document.getElementById('y-error-count').textContent = yErrors.size;
            }}
            
            updateErrorDisplay();
            calculateSyndrome();
        }}

        // Update error visualization
        function updateErrorDisplay() {{
            svg.selectAll(".css-node[data-node-type='qubit']").each(function(d) {{
                const node = d3.select(this);
                const hasXError = xErrors.has(d.id);
                const hasZError = zErrors.has(d.id);
                const hasYError = yErrors.has(d.id);
                
                node.classed("error", hasXError || hasZError || hasYError);
                
                // Update fill color based on error type
                let fill = "white";  // Default white
                let labelText = "";
                if (hasYError) {{
                    fill = "#ec4899";  // Pink for Y error
                    labelText = "Y";
                }} else if (hasXError && hasZError) {{
                    fill = "#9333ea";  // Purple for both X and Z
                    labelText = "XZ";
                }} else if (hasXError) {{
                    fill = "#ef4444";  // Red for X error
                    labelText = "X";
                }} else if (hasZError) {{
                    fill = "#3b82f6";  // Blue for Z error
                    labelText = "Z";
                }}
                
                node.select("circle").attr("fill", fill);
                
                // Update error label
                const errorLabel = node.select(".error-label");
                if (labelText) {{
                    errorLabel.text(labelText).style("visibility", "visible");
                }} else {{
                    errorLabel.style("visibility", "hidden");
                }}
            }});
        }}

        // Toggle error mode
        function toggleErrorMode(mode) {{
            const xBtn = document.getElementById('inject-x-btn');
            const zBtn = document.getElementById('inject-z-btn');
            const yBtn = document.getElementById('inject-y-btn');
            const modeDisplay = document.getElementById('mode-display');
            
            if (errorMode === mode) {{
                // Deactivate
                errorMode = null;
                xBtn.classList.remove('active');
                zBtn.classList.remove('active');
                yBtn.classList.remove('active');
                modeDisplay.textContent = 'None - Click a button above';
                modeDisplay.className = 'text-blue-400';
            }} else {{
                // Activate
                errorMode = mode;
                xBtn.classList.toggle('active', mode === 'X');
                zBtn.classList.toggle('active', mode === 'Z');
                yBtn.classList.toggle('active', mode === 'Y');
                
                if (mode === 'X') {{
                    modeDisplay.textContent = 'X Error Injection (Red)';
                    modeDisplay.className = 'text-red-400';
                }} else if (mode === 'Z') {{
                    modeDisplay.textContent = 'Z Error Injection (Blue)';
                    modeDisplay.className = 'text-blue-400';
                }} else if (mode === 'Y') {{
                    modeDisplay.textContent = 'Y Error Injection (Pink)';
                    modeDisplay.className = 'text-pink-400';
                }}
            }}
        }}

        // Toggle X-type visibility
        function toggleXType(visible) {{
            svg.selectAll(".css-node-x_check")
                .transition()
                .duration(300)
                .style("opacity", visible ? 1 : 0)
                .style("pointer-events", visible ? "all" : "none");
            
            svg.selectAll(".css-edge-x_check")
                .transition()
                .duration(300)
                .style("opacity", visible ? 1 : 0);
        }}

        // Toggle Z-type visibility
        function toggleZType(visible) {{
            svg.selectAll(".css-node-z_check")
                .transition()
                .duration(300)
                .style("opacity", visible ? 1 : 0)
                .style("pointer-events", visible ? "all" : "none");
            
            svg.selectAll(".css-edge-z_check")
                .transition()
                .duration(300)
                .style("opacity", visible ? 1 : 0);
        }}

        // Clear all errors
        function clearAllErrors() {{
            xErrors.clear();
            zErrors.clear();
            yErrors.clear();
            xSyndrome = [];
            zSyndrome = [];
            
            document.getElementById('x-error-count').textContent = '0';
            document.getElementById('z-error-count').textContent = '0';
            document.getElementById('y-error-count').textContent = '0';
            document.getElementById('x-syndrome-count').textContent = '0';
            document.getElementById('z-syndrome-count').textContent = '0';
            
            updateErrorDisplay();
            updateSyndromeDisplay();
        }}

        // Initialize display
        updateErrorDisplay();
        updateSyndromeDisplay();
    </script>
</body>
</html>"""
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"\n✅ Generated interactive surface code website: {output_file}")
    print(f"\nFeatures:")
    print(f"  ✓ Distance {distance} rotated surface code")
    print(f"  ✓ Show/hide X-type and Z-type elements")
    print(f"  ✓ Click-to-inject X and Z errors")
    print(f"  ✓ Real-time syndrome calculation (JavaScript)")
    print(f"  ✓ Modern UI with Tailwind CSS")
    print(f"\nOpen {output_file} in your browser to use the interactive visualization!")


if __name__ == "__main__":
    generate_interactive_surface_code_website(distance=7, output_file="surface_code_interactive.html")
