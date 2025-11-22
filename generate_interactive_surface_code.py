"""
Generate an interactive surface code website with multiple code types and distances.
Supports both RotatedSurfaceCode and SurfaceCode with selectable distances.
"""

import json
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from qec.code_constructions.rotated_surface_code import RotatedSurfaceCode
from qec.code_constructions.surface_code import SurfaceCode
from qec.utils.draw_css_code_d3 import prepare_css_code_visualization_data
from ldpc.mod2 import PluDecomposition


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
    
    # Compute PLU decomposition for Hx^T and Hz^T separately
    # For CSS codes:
    #   - Rows of Hx are X-stabilizers (X errors)
    #   - Rows of Hz are Z-stabilizers (Z errors)
    #   - X-checks (Hx) detect Z errors: syndrome_x = Hx @ e_z
    #   - Z-checks (Hz) detect X errors: syndrome_z = Hz @ e_x
    # 
    # To classify errors:
    #   1. Check zero syndrome: Hx @ e_z = 0 and Hz @ e_x = 0
    #   2. Check if e_x in row space of Hx: solve Hx^T @ sol_x = e_x
    #   3. Check if e_z in row space of Hz: solve Hz^T @ sol_z = e_z
    #   4. If both have solutions: STABILIZER, else: LOGICAL
    
    Hx_array = code.x_stabilizer_matrix.toarray()
    Hz_array = code.z_stabilizer_matrix.toarray()
    
    # PLU of Hx^T (for solving Hx^T @ sol = e_x)
    Hx_T = Hx_array.T
    plu_hx = PluDecomposition(Hx_T)
    
    # PLU of Hz^T (for solving Hz^T @ sol = e_z)
    Hz_T = Hz_array.T
    plu_hz = PluDecomposition(Hz_T)
    
    # Convert Hx PLU matrices to dense for JavaScript
    L_hx_dense = plu_hx.L.toarray().tolist()
    U_hx_dense = plu_hx.U.toarray().tolist()
    P_hx_dense = plu_hx.P.toarray().tolist()
    pivot_cols_hx = plu_hx.pivots.tolist()
    
    # Convert Hz PLU matrices to dense for JavaScript
    L_hz_dense = plu_hz.L.toarray().tolist()
    U_hz_dense = plu_hz.U.toarray().tolist()
    P_hz_dense = plu_hz.P.toarray().tolist()
    pivot_cols_hz = plu_hz.pivots.tolist()
    
    return {
        'nodes': data['nodes'],
        'edges': data['edges'],
        'config': data['config'],
        'metadata': data['metadata'],
        'Hx': Hx,
        'Hz': Hz,
        'qubit_id_to_index': qubit_id_to_index,
        'x_check_id_to_index': x_check_id_to_index,
        'z_check_id_to_index': z_check_id_to_index,
        'plu_hx': {
            'L': L_hx_dense,
            'U': U_hx_dense,
            'P': P_hx_dense,
            'rank': int(plu_hx.rank),
            'pivot_cols': pivot_cols_hx
        },
        'plu_hz': {
            'L': L_hz_dense,
            'U': U_hz_dense,
            'P': P_hz_dense,
            'rank': int(plu_hz.rank),
            'pivot_cols': pivot_cols_hz
        }
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
        code = RotatedSurfaceCode(d)
        data = prepare_css_code_visualization_data(
            code,
            qubit_radius=11,
            check_radius=11,
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
        
        all_codes_data[f"rotated_{d}"] = {
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
    
    # Surface Code: all distances 2-11
    print("  - Surface Codes (d=2-11)...")
    for d in range(2, 12):
        print(f"    Distance {d}...")
        all_codes_data[f"surface_{d}"] = prepare_code_data(SurfaceCode, d)
    
    # Serialize all code data to JSON
    print("Serializing code data to JSON...")
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
        .selected-stabilizer rect {{
            stroke: #fbbf24;
            stroke-width: 3;
            filter: drop-shadow(0 0 8px rgba(251, 191, 36, 0.6));
        }}
        .stabilizer-overlay {{
            filter: drop-shadow(0 0 4px currentColor);
        }}
        .logical-glow circle {{
            filter: drop-shadow(0 0 10px rgba(251, 146, 60, 0.8)) drop-shadow(0 0 20px rgba(251, 146, 60, 0.5));
        }}
        .stabilizer-component-glow rect {{
            filter: drop-shadow(0 0 10px rgba(34, 197, 94, 0.8)) drop-shadow(0 0 20px rgba(34, 197, 94, 0.5));
        }}
    </style>
</head>
<body class="bg-gray-900 text-gray-100">
    <div class="flex h-screen">
        <!-- Left Control Panel -->
        <div class="w-80 bg-gray-800 border-r border-gray-700 p-6 overflow-y-auto">
            <div class="mb-6">
                <h1 class="text-2xl font-bold text-blue-400 mb-2">Surface Code</h1>
                
                <!-- Code Type Selector -->
                <div class="mb-3">
                    <label class="block text-xs text-gray-400 mb-1">Code Type:</label>
                    <select id="code-type-select" 
                            class="w-full bg-gray-700 text-gray-100 border border-gray-600 rounded px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
                            onchange="onCodeTypeChange()">
                        <option value="rotated">Rotated Surface Code</option>
                        <option value="surface" selected>Surface Code</option>
                    </select>
                </div>
                
                <!-- Distance Selector -->
                <div class="mb-3">
                    <label class="block text-xs text-gray-400 mb-1">Distance:</label>
                    <select id="distance-select" 
                            class="w-full bg-gray-700 text-gray-100 border border-gray-600 rounded px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
                            onchange="onDistanceChange()">
                        <!-- Options populated by JavaScript -->
                    </select>
                </div>
                
                <div class="mt-2 p-3 bg-gray-700 rounded text-xs text-gray-300">
                    <div class="flex justify-between">
                        <span class="text-gray-400">Qubits:</span>
                        <span id="num-qubits" class="font-semibold">-</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">X-checks:</span>
                        <span id="num-x-checks" class="font-semibold">-</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Z-checks:</span>
                        <span id="num-z-checks" class="font-semibold">-</span>
                    </div>
                </div>
            </div>

            <!-- Legend -->
            <div class="mb-8">
                <h2 class="text-lg font-semibold mb-3 text-gray-200">Legend</h2>
                <div class="bg-white rounded-lg p-4">
                    <div class="grid grid-cols-2 gap-x-4 gap-y-3 text-xs">
                        <!-- Column 1 -->
                        <div class="space-y-3">
                            <!-- Nodes -->
                            <div>
                                <div class="font-semibold text-gray-800 mb-2">Nodes</div>
                                <div class="space-y-1.5">
                                    <div class="flex items-center space-x-2">
                                        <svg width="16" height="16">
                                            <circle cx="8" cy="8" r="6" fill="white" stroke="black" stroke-width="1.5"/>
                                        </svg>
                                        <span class="text-gray-800">Data qubit</span>
                                    </div>
                                    <div class="flex items-center space-x-2">
                                        <svg width="16" height="16">
                                            <rect x="2" y="2" width="12" height="12" fill="white" stroke="black" stroke-width="1.5"/>
                                        </svg>
                                        <span class="text-gray-800">Check qubit</span>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Edges -->
                            <div>
                                <div class="font-semibold text-gray-800 mb-2">Edges</div>
                                <div class="space-y-1.5">
                                    <div class="flex items-center space-x-2">
                                        <svg width="16" height="16">
                                            <line x1="0" y1="8" x2="16" y2="8" stroke="#666666" stroke-width="2"/>
                                        </svg>
                                        <span class="text-gray-800">X-type</span>
                                    </div>
                                    <div class="flex items-center space-x-2">
                                        <svg width="16" height="16">
                                            <line x1="0" y1="8" x2="16" y2="8" stroke="#666666" stroke-width="2" stroke-dasharray="3,3"/>
                                        </svg>
                                        <span class="text-gray-800">Z-type</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Column 2 -->
                        <div class="space-y-3">
                            <!-- Errors -->
                            <div>
                                <div class="font-semibold text-gray-800 mb-2">Errors</div>
                                <div class="space-y-1.5">
                                    <div class="flex items-center space-x-2">
                                        <div class="w-3 h-3 rounded-full bg-red-500"></div>
                                        <span class="text-gray-800">X Error</span>
                                    </div>
                                    <div class="flex items-center space-x-2">
                                        <div class="w-3 h-3 rounded-full bg-blue-500"></div>
                                        <span class="text-gray-800">Z Error</span>
                                    </div>
                                    <div class="flex items-center space-x-2">
                                        <div class="w-3 h-3 rounded-full bg-pink-500"></div>
                                        <span class="text-gray-800">Y Error</span>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Syndromes -->
                            <div>
                                <div class="font-semibold text-gray-800 mb-2">Syndromes</div>
                                <div class="space-y-1.5">
                                    <div class="flex items-center space-x-2">
                                        <svg width="16" height="16">
                                            <rect x="2" y="2" width="12" height="12" fill="#ef4444" stroke="black" stroke-width="1.5"/>
                                        </svg>
                                        <span class="text-gray-800">X-check</span>
                                    </div>
                                    <div class="flex items-center space-x-2">
                                        <svg width="16" height="16">
                                            <rect x="2" y="2" width="12" height="12" fill="#10b981" stroke="black" stroke-width="1.5"/>
                                        </svg>
                                        <span class="text-gray-800">Z-check</span>
                                    </div>
                                </div>
                            </div>
                        </div>
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
            </div>

            <!-- Clear Button -->
            <div class="mb-8">
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

            <!-- Visibility Controls (MOVED TO BOTTOM) -->
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

            <!-- Help Section -->
            <div class="mt-8 p-4 bg-blue-900 bg-opacity-30 border border-blue-700 rounded-lg text-xs">
                <h3 class="font-semibold text-blue-300 mb-2">How to use:</h3>
                <ol class="list-decimal list-inside space-y-1 text-gray-300">
                    <li>Select code type and distance</li>
                    <li>Toggle X, Y, or Z error mode</li>
                    <li>Click qubits to inject errors</li>
                    <li>Watch syndromes light up</li>
                    <li>Use checkboxes to hide/show elements</li>
                </ol>
            </div>
        </div>

        <!-- Main Visualization Area -->
        <div class="flex-1 bg-gray-900 flex flex-col items-center p-8 overflow-auto">
            <div class="bg-white rounded-xl shadow-2xl p-8 border border-gray-300 mb-8">
                <svg id="surface-code-viz"></svg>
            </div>
            
            <!-- Current State Display -->
            <div class="w-full max-w-4xl mb-6">
                <div class="bg-gray-800 rounded-xl shadow-2xl p-6 border border-gray-700">
                    <h2 class="text-xl font-semibold mb-4 text-gray-200">Current State</h2>
                    <div class="bg-gray-700 rounded-lg p-4">
                        <div class="flex items-center justify-between mb-2">
                            <span class="text-gray-400 text-sm">Classification:</span>
                            <span id="error-class-type-main" class="font-semibold text-lg text-gray-100">-</span>
                        </div>
                        <div id="error-class-details-main" class="text-sm text-gray-400 mt-3 hidden">
                            <div class="border-t border-gray-600 pt-3 mb-2"></div>
                            <div id="error-class-explanation-main"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Syndrome Information -->
            <div class="w-full max-w-4xl">
                <div class="bg-gray-800 rounded-xl shadow-2xl p-6 border border-gray-700">
                    <h2 class="text-xl font-semibold mb-4 text-gray-200">Syndrome Information</h2>
                    
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <!-- Error Vector / Stabilizer Selection -->
                        <div class="bg-gray-700 rounded-lg p-4">
                            <div id="error-vector-label" class="text-xs font-semibold text-gray-300 mb-2">Error Vector:</div>
                            <div id="error-vector-display" class="text-base font-mono text-gray-100 break-words min-h-[2rem] flex items-center">
                                <span class="text-gray-500">No errors</span>
                            </div>
                        </div>
                        
                        <!-- Syndrome Information -->
                        <div class="bg-gray-700 rounded-lg p-4">
                            <div class="text-xs font-semibold text-gray-300 mb-3">Syndrome:</div>
                            
                            <div class="space-y-2">
                                <!-- Z Syndrome -->
                                <div class="text-xs">
                                    <div class="text-gray-400 mb-1">S<sub>z</sub> = H<sub>z</sub> ⊗ e<sub>x</sub> mod 2</div>
                                    <div id="z-syndrome-display" class="font-mono text-sm text-green-400 break-words">
                                        <span class="text-gray-500">-</span>
                                    </div>
                                </div>
                                
                                <!-- X Syndrome -->
                                <div class="text-xs">
                                    <div class="text-gray-400 mb-1">S<sub>x</sub> = H<sub>x</sub> ⊗ e<sub>z</sub> mod 2</div>
                                    <div id="x-syndrome-display" class="font-mono text-sm text-red-400 break-words">
                                        <span class="text-gray-500">-</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // All code data embedded
        const ALL_CODES = {all_codes_json};
        
        // Current state
        let currentCodeType = 'surface';
        let currentDistance = 4;
        let currentCodeKey = 'surface_4';
        let currentData = null;
        
        // Error state
        let errorMode = null;  // 'X', 'Z', 'Y', 'X_STAB', 'Z_STAB', or null
        let xErrors = new Set();
        let zErrors = new Set();
        let yErrors = new Set();
        let xSyndrome = [];
        let zSyndrome = [];
        let selectedStabilizers = new Set();  // Track selected stabilizer check nodes
        
        // SVG reference
        let svg = null;
        
        // Initialize
        function init() {{
            svg = d3.select("#surface-code-viz");
            
            // Sync code type dropdown with initial state
            document.getElementById('code-type-select').value = currentCodeType;
            
            // Populate distance options (which will set the selected distance)
            populateDistanceOptions();
            loadCode();
        }}
        
        // Populate distance dropdown based on code type
        function populateDistanceOptions() {{
            const select = document.getElementById('distance-select');
            select.innerHTML = '';
            
            let distances;
            if (currentCodeType === 'rotated') {{
                distances = [3, 5, 7, 9, 11];  // Odd only
            }} else {{
                distances = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11];  // All
            }}
            
            distances.forEach(d => {{
                const option = document.createElement('option');
                option.value = d;
                option.textContent = `d = ${{d}}`;
                if (d === currentDistance) option.selected = true;
                select.appendChild(option);
            }});
        }}
        
        // Handle code type change
        function onCodeTypeChange() {{
            const select = document.getElementById('code-type-select');
            currentCodeType = select.value;
            
            // Adjust distance if needed
            if (currentCodeType === 'rotated' && currentDistance % 2 === 0) {{
                currentDistance = currentDistance + 1;
                if (currentDistance > 11) currentDistance = 11;
            }}
            
            populateDistanceOptions();
            loadCode();
        }}
        
        // Handle distance change
        function onDistanceChange() {{
            const select = document.getElementById('distance-select');
            currentDistance = parseInt(select.value);
            loadCode();
        }}
        
        // Load the selected code
        function loadCode() {{
            currentCodeKey = `${{currentCodeType}}_${{currentDistance}}`;
            currentData = ALL_CODES[currentCodeKey];
            
            if (!currentData) {{
                console.error('Code data not found:', currentCodeKey);
                return;
            }}
            
            // Update metadata display
            document.getElementById('num-qubits').textContent = currentData.metadata.num_qubits;
            document.getElementById('num-x-checks').textContent = currentData.metadata.num_x_checks;
            document.getElementById('num-z-checks').textContent = currentData.metadata.num_z_checks;
            
            // Clear errors
            clearAllErrors();
            
            // Redraw visualization
            drawVisualization();
        }}
        
        // Draw the visualization
        function drawVisualization() {{
            // Clear SVG
            svg.selectAll("*").remove();
            
            // Set SVG size
            svg.attr("width", currentData.config.width)
               .attr("height", currentData.config.height);
            
            // Draw edges
            const edgesGroup = svg.append("g").attr("class", "edges-group");
            
            edgesGroup.selectAll(".css-edge")
                .data(currentData.edges)
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
                .data(currentData.nodes)
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
            
            // Add tooltips
            nodes.append("title")
                .text(d => `${{d.tooltip_label || d.id}} (${{d.type_label || d.type}})`);
        }}
        
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
        
        // PLU Forward-Backward Solver for GF(2)
        // Solves: PAx = Py where A = LU
        // P is a permutation matrix, L is lower triangular, U is upper triangular
        // Returns solution x if y is in image of A, null otherwise
        // Solve A @ x = b over GF(2) using Gaussian elimination
        // Returns null if no solution exists, otherwise returns a solution vector
        function solveGF2(A, b) {{
            const m = A.length;      // number of rows
            const n = A[0].length;   // number of columns
            
            if (b.length !== m) {{
                console.log('solveGF2: dimension mismatch');
                return null;
            }}
            
            // Create augmented matrix [A | b]
            const aug = [];
            for (let i = 0; i < m; i++) {{
                aug[i] = [...A[i], b[i]];
            }}
            
            // Gaussian elimination
            let pivotRow = 0;
            const pivotCols = [];
            
            for (let col = 0; col < n && pivotRow < m; col++) {{
                // Find pivot
                let foundPivot = false;
                for (let row = pivotRow; row < m; row++) {{
                    if (aug[row][col] === 1) {{
                        // Swap rows
                        if (row !== pivotRow) {{
                            [aug[pivotRow], aug[row]] = [aug[row], aug[pivotRow]];
                        }}
                        foundPivot = true;
                        break;
                    }}
                }}
                
                if (!foundPivot) continue;
                
                pivotCols.push(col);
                
                // Eliminate column in other rows
                for (let row = 0; row < m; row++) {{
                    if (row !== pivotRow && aug[row][col] === 1) {{
                        for (let j = 0; j <= n; j++) {{
                            aug[row][j] ^= aug[pivotRow][j];
                        }}
                    }}
                }}
                
                pivotRow++;
            }}
            
            // Check for inconsistency: row with [0 0 ... 0 | 1]
            for (let row = 0; row < m; row++) {{
                let allZero = true;
                for (let col = 0; col < n; col++) {{
                    if (aug[row][col] === 1) {{
                        allZero = false;
                        break;
                    }}
                }}
                if (allZero && aug[row][n] === 1) {{
                    // Inconsistent system
                    return null;
                }}
            }}
            
            // Extract solution (one possible solution)
            const x = new Array(n).fill(0);
            for (let i = 0; i < pivotCols.length; i++) {{
                x[pivotCols[i]] = aug[i][n];
            }}
            
            return x;
        }}
        
        // Classify error as stabilizer, logical, or neither
        function classifyError() {{
            // First check if syndrome is zero
            if (xSyndrome.length > 0 || zSyndrome.length > 0) {{
                updateErrorClassDisplay('Non-trivial', 'Syndrome is non-zero. Not a stabilizer or logical.');
                return;
            }}
            
            // If no errors, it's trivial
            if (xErrors.size === 0 && zErrors.size === 0 && yErrors.size === 0) {{
                updateErrorClassDisplay('-', '');
                return;
            }}
            
            // Build error vectors e_x and e_z separately
            const numQubits = Object.keys(currentData.qubit_id_to_index).length;
            const e_x = new Array(numQubits).fill(0);
            const e_z = new Array(numQubits).fill(0);
            
            // X errors (including Y errors)
            xErrors.forEach(qubitId => {{
                const idx = currentData.qubit_id_to_index[qubitId];
                if (idx !== undefined) e_x[idx] = 1;
            }});
            yErrors.forEach(qubitId => {{
                const idx = currentData.qubit_id_to_index[qubitId];
                if (idx !== undefined) e_x[idx] = 1;
            }});
            
            // Z errors (including Y errors)
            zErrors.forEach(qubitId => {{
                const idx = currentData.qubit_id_to_index[qubitId];
                if (idx !== undefined) e_z[idx] = 1;
            }});
            yErrors.forEach(qubitId => {{
                const idx = currentData.qubit_id_to_index[qubitId];
                if (idx !== undefined) e_z[idx] = 1;
            }});
            
            // Check if e_x is in row space of Hx: solve Hx^T @ sol_x = e_x
            // Build Hx^T
            const numXChecks = currentData.Hx.length;
            const Hx_T = [];
            for (let i = 0; i < numQubits; i++) {{
                Hx_T[i] = [];
                for (let j = 0; j < numXChecks; j++) {{
                    Hx_T[i][j] = currentData.Hx[j][i];
                }}
            }}
            
            const sol_x = solveGF2(Hx_T, e_x);
            console.log('sol_x:', sol_x, 'e_x:', e_x);
            
            // Check if e_z is in row space of Hz: solve Hz^T @ sol_z = e_z
            const numZChecks = currentData.Hz.length;
            const Hz_T = [];
            for (let i = 0; i < numQubits; i++) {{
                Hz_T[i] = [];
                for (let j = 0; j < numZChecks; j++) {{
                    Hz_T[i][j] = currentData.Hz[j][i];
                }}
            }}
            
            const sol_z = solveGF2(Hz_T, e_z);
            console.log('sol_z:', sol_z, 'e_z:', e_z);
            
            // If both have solutions, it's a stabilizer; otherwise it's a logical
            if (sol_x === null || sol_z === null) {{
                console.log('Classification: LOGICAL (sol_x null?', sol_x === null, 'sol_z null?', sol_z === null, ')');
                
                // Apply glow to all error qubits (logical error)
                clearClassificationGlows();
                const allErrorQubits = new Set([...xErrors, ...zErrors, ...yErrors]);
                allErrorQubits.forEach(qubitId => {{
                    const node = svg.select(`[data-node-id="${{qubitId}}"]`);
                    node.classed("logical-glow", true);
                }});
                
                updateErrorClassDisplay('Logical Error', 'Error has zero syndrome but is NOT in the stabilizer group.');
            }} else {{
                // Both solutions exist - verify they reconstruct correctly
                // Verify Hx^T @ sol_x = e_x
                const Hx_T_times_sol_x = new Array(numQubits).fill(0);
                for (let i = 0; i < numQubits; i++) {{
                    Hx_T_times_sol_x[i] = Hx_T[i].reduce((sum, val, j) => sum ^ (val * sol_x[j]), 0);
                }}
                
                // Verify Hz^T @ sol_z = e_z
                const Hz_T_times_sol_z = new Array(numQubits).fill(0);
                for (let i = 0; i < numQubits; i++) {{
                    Hz_T_times_sol_z[i] = Hz_T[i].reduce((sum, val, j) => sum ^ (val * sol_z[j]), 0);
                }}
                
                // Check if reconstructions match
                let x_matches = true;
                for (let i = 0; i < numQubits; i++) {{
                    if (Hx_T_times_sol_x[i] !== e_x[i]) {{
                        x_matches = false;
                        break;
                    }}
                }}
                
                let z_matches = true;
                for (let i = 0; i < numQubits; i++) {{
                    if (Hz_T_times_sol_z[i] !== e_z[i]) {{
                        z_matches = false;
                        break;
                    }}
                }}
                
                console.log('x_matches:', x_matches, 'z_matches:', z_matches);
                console.log('Hx_T @ sol_x:', Hx_T_times_sol_x);
                console.log('Hz_T @ sol_z:', Hz_T_times_sol_z);
                
                if (!x_matches || !z_matches) {{
                    // This shouldn't happen with correct solver, but treat as logical if it does
                    clearClassificationGlows();
                    const allErrorQubits = new Set([...xErrors, ...zErrors, ...yErrors]);
                    allErrorQubits.forEach(qubitId => {{
                        const node = svg.select(`[data-node-id="${{qubitId}}"]`);
                        node.classed("logical-glow", true);
                    }});
                    updateErrorClassDisplay('Logical Error', 'Error has zero syndrome but stabilizer decomposition does NOT equal the error pattern.');
                }} else {{
                    // Count which stabilizers are used
                    // sol_x tells us which X-stabilizers (rows of Hx) are involved
                    // sol_z tells us which Z-stabilizers (rows of Hz) are involved
                    const usedXStabilizers = [];
                    const usedZStabilizers = [];
                    for (let i = 0; i < sol_x.length; i++) {{
                        if (sol_x[i] === 1) usedXStabilizers.push(i);
                    }}
                    for (let i = 0; i < sol_z.length; i++) {{
                        if (sol_z[i] === 1) usedZStabilizers.push(i);
                    }}
                    
                    if (usedXStabilizers.length === 0 && usedZStabilizers.length === 0) {{
                        clearClassificationGlows();
                        updateErrorClassDisplay('Identity', 'No errors present.');
                    }} else {{
                        // Apply glow to stabilizer check nodes
                        clearClassificationGlows();
                        
                        // Glow X-check nodes for used X-stabilizers
                        usedXStabilizers.forEach(stabIdx => {{
                            const checkId = Object.keys(currentData.x_check_id_to_index).find(id => currentData.x_check_id_to_index[id] === stabIdx);
                            if (checkId) {{
                                const node = svg.select(`[data-node-id="${{checkId}}"]`);
                                node.classed("stabilizer-component-glow", true);
                            }}
                        }});
                        
                        // Glow Z-check nodes for used Z-stabilizers
                        usedZStabilizers.forEach(stabIdx => {{
                            const checkId = Object.keys(currentData.z_check_id_to_index).find(id => currentData.z_check_id_to_index[id] === stabIdx);
                            if (checkId) {{
                                const node = svg.select(`[data-node-id="${{checkId}}"]`);
                                node.classed("stabilizer-component-glow", true);
                            }}
                        }});
                        
                        const parts = [];
                        if (usedXStabilizers.length > 0) {{
                            parts.push('X-stabs: ' + usedXStabilizers.join(','));
                        }}
                        if (usedZStabilizers.length > 0) {{
                            parts.push('Z-stabs: ' + usedZStabilizers.join(','));
                        }}
                        updateErrorClassDisplay('Stabilizer', parts.join(' | '));
                    }}
                }}
            }}
        }}
        
        // Update error classification display
        function updateErrorClassDisplay(type, explanation) {{
            // Update main display only (sidebar legend removed)
            const typeElemMain = document.getElementById('error-class-type-main');
            const detailsElemMain = document.getElementById('error-class-details-main');
            const explanationElemMain = document.getElementById('error-class-explanation-main');
            
            typeElemMain.textContent = type;
            
            let classNameMain = 'font-semibold text-lg text-gray-100';
            
            if (type === 'Stabilizer') {{
                classNameMain = 'font-semibold text-lg text-green-400';
            }} else if (type === 'Logical Error') {{
                classNameMain = 'font-semibold text-lg text-orange-400';
            }} else if (type === 'Non-trivial') {{
                classNameMain = 'font-semibold text-lg text-yellow-400';
            }}
            
            typeElemMain.className = classNameMain;
            
            if (explanation) {{
                explanationElemMain.textContent = explanation;
                detailsElemMain.classList.remove('hidden');
            }} else {{
                detailsElemMain.classList.add('hidden');
            }}
        }}
        
        // Calculate syndrome
        function calculateSyndrome() {{
            const numQubits = Object.keys(currentData.qubit_id_to_index).length;
            
            // Create error vectors
            const xErrorVector = new Array(numQubits).fill(0);
            const zErrorVector = new Array(numQubits).fill(0);
            
            // X errors (including Y errors)
            xErrors.forEach(qubitId => {{
                const idx = currentData.qubit_id_to_index[qubitId];
                if (idx !== undefined) xErrorVector[idx] = 1;
            }});
            yErrors.forEach(qubitId => {{
                const idx = currentData.qubit_id_to_index[qubitId];
                if (idx !== undefined) xErrorVector[idx] = 1;
            }});
            
            // Z errors (including Y errors)
            zErrors.forEach(qubitId => {{
                const idx = currentData.qubit_id_to_index[qubitId];
                if (idx !== undefined) zErrorVector[idx] = 1;
            }});
            yErrors.forEach(qubitId => {{
                const idx = currentData.qubit_id_to_index[qubitId];
                if (idx !== undefined) zErrorVector[idx] = 1;
            }});
            
            // Calculate syndromes
            const zSyndromeVector = matrixVectorMod2(currentData.Hz, xErrorVector);
            const xSyndromeVector = matrixVectorMod2(currentData.Hx, zErrorVector);
            
            // Convert to triggered check indices
            zSyndrome = zSyndromeVector.map((val, idx) => val === 1 ? idx : -1).filter(idx => idx !== -1);
            xSyndrome = xSyndromeVector.map((val, idx) => val === 1 ? idx : -1).filter(idx => idx !== -1);
            
            updateSyndromeDisplay();
            classifyError();
        }}
        
        // Clear all classification glows
        function clearClassificationGlows() {{
            svg.selectAll(".css-node").classed("logical-glow", false);
            svg.selectAll(".css-node").classed("stabilizer-component-glow", false);
        }}
        
        // Update syndrome visualization
        function updateSyndromeDisplay() {{
            // Reset all check nodes
            svg.selectAll(".css-node-x_check rect").attr("fill", "white");
            svg.selectAll(".css-node-z_check rect").attr("fill", "white");
            svg.selectAll(".css-node").classed("syndrome", false);
            
            // Clear classification glows when syndrome changes
            clearClassificationGlows();
            
            // Highlight X-checks (triggered by Z errors) in solid red
            xSyndrome.forEach(idx => {{
                const checkId = Object.keys(currentData.x_check_id_to_index).find(id => currentData.x_check_id_to_index[id] === idx);
                if (checkId) {{
                    const node = svg.select(`[data-node-id="${{checkId}}"]`);
                    node.classed("syndrome", true);
                    node.select("rect").attr("fill", "#ef4444");
                }}
            }});
            
            // Highlight Z-checks (triggered by X errors) in solid green
            zSyndrome.forEach(idx => {{
                const checkId = Object.keys(currentData.z_check_id_to_index).find(id => currentData.z_check_id_to_index[id] === idx);
                if (checkId) {{
                    const node = svg.select(`[data-node-id="${{checkId}}"]`);
                    node.classed("syndrome", true);
                    node.select("rect").attr("fill", "#10b981");
                }}
            }});
            
            // Update detailed error and syndrome displays
            updateErrorVectorDisplay();
            updateSyndromeTextDisplay();
        }}
        
        // Update error vector display with proper formatting
        function updateErrorVectorDisplay() {{
            const errorVectorDiv = document.getElementById('error-vector-display');
            
            console.log('updateErrorVectorDisplay called');
            console.log('errorMode:', errorMode);
            console.log('selectedStabilizers:', selectedStabilizers);
            
            // If in stabilizer selection mode, show selected stabilizers instead
            if (errorMode === 'X_STAB' || errorMode === 'Z_STAB') {{
                console.log('In stabilizer mode');
                if (selectedStabilizers.size === 0) {{
                    console.log('No stabilizers selected');
                    errorVectorDiv.innerHTML = '<span class="text-gray-500">No stabilizers selected</span>';
                    return;
                }}
                
                // Collect selected stabilizers with their indices
                const stabilizerList = [];
                
                selectedStabilizers.forEach(checkId => {{
                    console.log('Processing stabilizer:', checkId);
                    const checkNode = currentData.nodes.find(n => n.id === checkId);
                    console.log('Found node:', checkNode);
                    
                    if (!checkNode) return;
                    
                    const checkType = checkNode.type;
                    const idToIndex = checkType === 'x_check' ? currentData.x_check_id_to_index : currentData.z_check_id_to_index;
                    const idx = idToIndex[checkId];
                    
                    console.log('checkType:', checkType, 'idx:', idx);
                    
                    if (idx !== undefined) {{
                        const stabType = checkType === 'x_check' ? 'X' : 'Z';
                        const colorClass = checkType === 'x_check' ? 'text-orange-400' : 'text-teal-400';
                        stabilizerList.push({{ type: stabType, index: idx, colorClass }});
                    }}
                }});
                
                console.log('stabilizerList:', stabilizerList);
                
                // Sort by index
                stabilizerList.sort((a, b) => a.index - b.index);
                
                // Format as S_X(1) S_Z(2) etc.
                const formattedStabilizers = stabilizerList.map(stab => {{
                    return `<span class="${{stab.colorClass}}">S<sub>${{stab.type}}</sub>(${{stab.index}})</span>`;
                }});
                
                console.log('Formatted output:', formattedStabilizers.join(' '));
                errorVectorDiv.innerHTML = formattedStabilizers.join(' ');
                return;
            }}
            
            // Regular error display mode
            // Collect all errors with their types and qubit indices
            const errorList = [];
            
            xErrors.forEach(qubitId => {{
                const idx = currentData.qubit_id_to_index[qubitId];
                if (idx !== undefined) {{
                    errorList.push({{ type: 'X', index: idx }});
                }}
            }});
            
            zErrors.forEach(qubitId => {{
                const idx = currentData.qubit_id_to_index[qubitId];
                if (idx !== undefined) {{
                    errorList.push({{ type: 'Z', index: idx }});
                }}
            }});
            
            yErrors.forEach(qubitId => {{
                const idx = currentData.qubit_id_to_index[qubitId];
                if (idx !== undefined) {{
                    errorList.push({{ type: 'Y', index: idx }});
                }}
            }});
            
            if (errorList.length === 0) {{
                errorVectorDiv.innerHTML = '<span class="text-gray-500">No errors</span>';
                return;
            }}
            
            // Sort by index
            errorList.sort((a, b) => a.index - b.index);
            
            // Format as X_1 Z_2 Y_3 etc.
            const formattedErrors = errorList.map(err => {{
                const colorClass = err.type === 'X' ? 'text-red-400' : 
                                  err.type === 'Z' ? 'text-blue-400' : 'text-pink-400';
                return `<span class="${{colorClass}}">${{err.type}}<sub>${{err.index}}</sub></span>`;
            }});
            
            errorVectorDiv.innerHTML = formattedErrors.join(' ');
        }}
        
        // Update syndrome text display
        function updateSyndromeTextDisplay() {{
            const zSyndromeDiv = document.getElementById('z-syndrome-display');
            const xSyndromeDiv = document.getElementById('x-syndrome-display');
            
            // Format Z syndrome (triggered by X errors)
            if (zSyndrome.length === 0) {{
                zSyndromeDiv.innerHTML = '<span class="text-gray-500">-</span>';
            }} else {{
                const zChecks = zSyndrome.map(idx => `<span class="text-green-400">s<sub>z</sub>(${{idx}})</span>`);
                zSyndromeDiv.innerHTML = zChecks.join(', ');
            }}
            
            // Format X syndrome (triggered by Z errors)
            if (xSyndrome.length === 0) {{
                xSyndromeDiv.innerHTML = '<span class="text-gray-500">-</span>';
            }} else {{
                const xChecks = xSyndrome.map(idx => `<span class="text-red-400">s<sub>x</sub>(${{idx}})</span>`);
                xSyndromeDiv.innerHTML = xChecks.join(', ');
            }}
        }}
        
        // Handle node click
        function handleNodeClick(event, d) {{
            console.log('Node clicked:', d);
            console.log('Current errorMode:', errorMode);
            
            // Handle stabilizer selection mode
            if (errorMode === 'X_STAB' || errorMode === 'Z_STAB') {{
                console.log('In stabilizer selection mode');
                const expectedType = errorMode === 'X_STAB' ? 'x_check' : 'z_check';
                console.log('Expected type:', expectedType, 'Node type:', d.type);
                
                if (d.type !== expectedType) {{
                    console.log('Wrong type, returning');
                    return;
                }}
                
                const checkId = d.id;
                console.log('Check ID:', checkId);
                console.log('Before - selectedStabilizers:', Array.from(selectedStabilizers));
                
                if (selectedStabilizers.has(checkId)) {{
                    selectedStabilizers.delete(checkId);
                    console.log('Removed stabilizer');
                }} else {{
                    selectedStabilizers.add(checkId);
                    console.log('Added stabilizer');
                }}
                
                console.log('After - selectedStabilizers:', Array.from(selectedStabilizers));
                console.log('Calling applyStabilizerErrors()');
                applyStabilizerErrors();
                return;
            }}
            
            // Handle regular error injection mode
            if (!errorMode || d.type !== 'qubit') return;
            
            const qubitId = d.id;
            
            if (errorMode === 'X') {{
                if (xErrors.has(qubitId)) {{
                    xErrors.delete(qubitId);
                }} else {{
                    xErrors.add(qubitId);
                }}
            }} else if (errorMode === 'Z') {{
                if (zErrors.has(qubitId)) {{
                    zErrors.delete(qubitId);
                }} else {{
                    zErrors.add(qubitId);
                }}
            }} else if (errorMode === 'Y') {{
                if (yErrors.has(qubitId)) {{
                    yErrors.delete(qubitId);
                }} else {{
                    yErrors.add(qubitId);
                }}
            }}
            
            updateErrorDisplay();
            calculateSyndrome();
            updateErrorVectorDisplay();
        }}
        
        // Get qubits connected to a stabilizer check
        function getConnectedQubits(checkId, checkType) {{
            const matrix = checkType === 'x_check' ? currentData.Hx : currentData.Hz;
            const idToIndex = checkType === 'x_check' ? currentData.x_check_id_to_index : currentData.z_check_id_to_index;
            
            const checkIdx = idToIndex[checkId];
            if (checkIdx === undefined) return [];
            
            const connectedQubits = [];
            const row = matrix[checkIdx];
            row.forEach((val, qubitIdx) => {{
                if (val === 1) {{
                    // Find qubit ID from index
                    const qubitId = Object.keys(currentData.qubit_id_to_index).find(
                        id => currentData.qubit_id_to_index[id] === qubitIdx
                    );
                    if (qubitId) connectedQubits.push(qubitId);
                }}
            }});
            
            return connectedQubits;
        }}
        
        // Apply errors based on selected stabilizers
        function applyStabilizerErrors() {{
            // Clear existing errors first
            xErrors.clear();
            zErrors.clear();
            yErrors.clear();
            
            // Track how many times each qubit gets an error (for cancellation)
            const xErrorCounts = {{}};
            const zErrorCounts = {{}};
            
            selectedStabilizers.forEach(checkId => {{
                // Determine check type by looking at node
                const node = currentData.nodes.find(n => n.id === checkId);
                if (!node) return;
                
                const checkType = node.type;
                const connectedQubits = getConnectedQubits(checkId, checkType);
                
                // Apply appropriate errors
                connectedQubits.forEach(qubitId => {{
                    if (checkType === 'x_check') {{
                        // X-type stabilizers apply X errors
                        xErrorCounts[qubitId] = (xErrorCounts[qubitId] || 0) + 1;
                    }} else {{
                        // Z-type stabilizers apply Z errors
                        zErrorCounts[qubitId] = (zErrorCounts[qubitId] || 0) + 1;
                    }}
                }});
            }});
            
            // Only add errors that appear odd number of times (Pauli errors cancel)
            Object.entries(xErrorCounts).forEach(([qubitId, count]) => {{
                if (count % 2 === 1) xErrors.add(qubitId);
            }});
            
            Object.entries(zErrorCounts).forEach(([qubitId, count]) => {{
                if (count % 2 === 1) zErrors.add(qubitId);
            }});
            
            updateErrorDisplay();
            updateStabilizerOverlays();
            calculateSyndrome();
            updateErrorVectorDisplay();
        }}
        
        // Draw stabilizer overlays
        function updateStabilizerOverlays() {{
            // Remove existing overlays
            svg.selectAll(".stabilizer-overlay").remove();
            svg.selectAll(".stabilizer-fill").remove();
            svg.selectAll(".css-node").classed("selected-stabilizer", false);
            
            if (errorMode !== 'X_STAB' && errorMode !== 'Z_STAB') return;
            
            const color = errorMode === 'X_STAB' ? '#f97316' : '#14b8a6';
            
            // Track which qubits are shared between stabilizers
            const qubitToStabilizers = new Map();
            
            // First pass: map each qubit to the stabilizers that use it
            selectedStabilizers.forEach(checkId => {{
                const checkNode = currentData.nodes.find(n => n.id === checkId);
                if (!checkNode) return;
                
                const connectedQubits = getConnectedQubits(checkId, checkNode.type);
                connectedQubits.forEach(qubitId => {{
                    if (!qubitToStabilizers.has(qubitId)) {{
                        qubitToStabilizers.set(qubitId, new Set());
                    }}
                    qubitToStabilizers.get(qubitId).add(checkId);
                }});
            }});
            
            // Second pass: draw each stabilizer
            selectedStabilizers.forEach(checkId => {{
                const checkNode = currentData.nodes.find(n => n.id === checkId);
                if (!checkNode) return;
                
                // Highlight the check node itself
                svg.select(`[data-node-id="${{checkId}}"]`).classed("selected-stabilizer", true);
                
                // Get connected qubits and their positions
                const connectedQubits = getConnectedQubits(checkId, checkNode.type);
                const qubitPositions = connectedQubits.map(qubitId => {{
                    const qubitNode = currentData.nodes.find(n => n.id === qubitId);
                    return qubitNode ? {{ x: qubitNode.x, y: qubitNode.y, id: qubitId }} : null;
                }}).filter(p => p !== null);
                
                if (qubitPositions.length < 3) return;
                
                // Sort qubits by angle around check node
                const centerX = checkNode.x;
                const centerY = checkNode.y;
                
                qubitPositions.sort((a, b) => {{
                    const angleA = Math.atan2(a.y - centerY, a.x - centerX);
                    const angleB = Math.atan2(b.y - centerY, b.x - centerX);
                    return angleA - angleB;
                }});
                
                // For the plaquette: vertices should be at the neighboring check nodes
                // Each data qubit connects this check to another check of the same type
                // The plaquette is formed by these neighboring check positions
                const boundarySegments = [];
                const polygonCorners = [];
                
                // Find all neighboring check nodes of the same type
                const neighborChecks = [];
                qubitPositions.forEach(qubit => {{
                    // Find other checks connected to this qubit with the same type
                    currentData.edges.forEach(edge => {{
                        if ((edge.source === qubit.id || edge.target === qubit.id)) {{
                            const otherId = edge.source === qubit.id ? edge.target : edge.source;
                            const otherNode = currentData.nodes.find(n => n.id === otherId);
                            
                            if (otherNode && otherNode.id !== checkId && otherNode.type === checkNode.type) {{
                                // Check if not already added
                                if (!neighborChecks.some(n => n.id === otherId)) {{
                                    neighborChecks.push(otherNode);
                                }}
                            }}
                        }}
                    }});
                }});
                
                // For boundary lines: one per data qubit
                qubitPositions.forEach(qubit => {{
                    const dx = qubit.x - centerX;
                    const dy = qubit.y - centerY;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    
                    const perpX = -dy / dist;
                    const perpY = dx / dist;
                    
                    const halfLength = dist;
                    const p1 = {{
                        x: qubit.x + perpX * halfLength,
                        y: qubit.y + perpY * halfLength
                    }};
                    const p2 = {{
                        x: qubit.x - perpX * halfLength,
                        y: qubit.y - perpY * halfLength
                    }};
                    
                    boundarySegments.push({{
                        qubitId: qubit.id,
                        p1: p1,
                        p2: p2
                    }});
                }});
                
                // Build polygon from neighboring check positions
                if (neighborChecks.length >= 2) {{
                    // For bulk stabilizers: 4 neighboring checks
                    // For boundary stabilizers: 2 neighboring checks + 2 boundary qubits
                    
                    if (neighborChecks.length >= 3) {{
                        // Bulk stabilizer: use check positions
                        neighborChecks.forEach(check => {{
                            polygonCorners.push({{ x: check.x, y: check.y }});
                        }});
                        
                        // Sort by angle to form proper polygon
                        polygonCorners.sort((a, b) => {{
                            const angleA = Math.atan2(a.y - centerY, a.x - centerX);
                            const angleB = Math.atan2(b.y - centerY, b.x - centerX);
                            return angleA - angleB;
                        }});
                    }} else {{
                        // Boundary stabilizer: 2 checks + 2 boundary data qubits
                        // Add the check positions
                        neighborChecks.forEach(check => {{
                            polygonCorners.push({{ x: check.x, y: check.y }});
                        }});
                        
                        // Find the boundary qubits (the ones at the edges)
                        // Sort qubits by angle to identify which are the edge ones
                        const sortedQubits = [...qubitPositions].sort((a, b) => {{
                            const angleA = Math.atan2(a.y - centerY, a.x - centerX);
                            const angleB = Math.atan2(b.y - centerY, b.x - centerX);
                            return angleA - angleB;
                        }});
                        
                        // Add first and last qubits (the edge ones)
                        polygonCorners.push({{ x: sortedQubits[0].x, y: sortedQubits[0].y }});
                        polygonCorners.push({{ x: sortedQubits[sortedQubits.length - 1].x, y: sortedQubits[sortedQubits.length - 1].y }});
                        
                        // Sort corners by angle
                        polygonCorners.sort((a, b) => {{
                            const angleA = Math.atan2(a.y - centerY, a.x - centerX);
                            const angleB = Math.atan2(b.y - centerY, b.x - centerX);
                            return angleA - angleB;
                        }});
                    }}
                }}
                
                // Draw translucent fill
                if (polygonCorners.length >= 3) {{
                    const fillPolygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
                    fillPolygon.setAttribute("class", "stabilizer-fill");
                    fillPolygon.setAttribute("points", polygonCorners.map(p => `${{p.x}},${{p.y}}`).join(" "));
                    fillPolygon.setAttribute("fill", color);
                    fillPolygon.setAttribute("opacity", "0.15");
                    fillPolygon.setAttribute("stroke", "none");
                    
                    const firstChild = svg.node().firstChild;
                    if (firstChild) {{
                        svg.node().insertBefore(fillPolygon, firstChild);
                    }} else {{
                        svg.node().appendChild(fillPolygon);
                    }}
                }}
                
                // Draw boundary lines (skip if qubit is shared)
                boundarySegments.forEach(seg => {{
                    const stabilizersAtQubit = qubitToStabilizers.get(seg.qubitId);
                    const isShared = stabilizersAtQubit && stabilizersAtQubit.size > 1;
                    
                    if (!isShared) {{
                        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                        line.setAttribute("class", "stabilizer-overlay");
                        line.setAttribute("x1", seg.p1.x);
                        line.setAttribute("y1", seg.p1.y);
                        line.setAttribute("x2", seg.p2.x);
                        line.setAttribute("y2", seg.p2.y);
                        line.setAttribute("stroke", color);
                        line.setAttribute("stroke-width", "3");
                        line.setAttribute("opacity", "0.9");
                        line.setAttribute("filter", "drop-shadow(0 0 4px " + color + ")");
                        
                        const fillElement = svg.select(".stabilizer-fill").node();
                        if (fillElement && fillElement.nextSibling) {{
                            svg.node().insertBefore(line, fillElement.nextSibling);
                        }} else {{
                            svg.node().appendChild(line);
                        }}
                    }}
                }});
            }});
        }}
        
        // Compute convex hull using gift wrapping algorithm
        function computeConvexHull(points) {{
            if (points.length < 3) return points;
            
            // Find leftmost point
            let leftmost = points[0];
            points.forEach(p => {{
                if (p.x < leftmost.x || (p.x === leftmost.x && p.y < leftmost.y)) {{
                    leftmost = p;
                }}
            }});
            
            const hull = [];
            let current = leftmost;
            let next;
            
            do {{
                hull.push(current);
                next = points[0];
                
                for (let i = 1; i < points.length; i++) {{
                    const cross = (next.x - current.x) * (points[i].y - current.y) - 
                                  (next.y - current.y) * (points[i].x - current.x);
                    const dist1 = (current.x - next.x) ** 2 + (current.y - next.y) ** 2;
                    const dist2 = (current.x - points[i].x) ** 2 + (current.y - points[i].y) ** 2;
                    
                    if (next === current || cross < 0 || (cross === 0 && dist2 > dist1)) {{
                        next = points[i];
                    }}
                }}
                
                current = next;
            }} while (current !== leftmost && hull.length < points.length);
            
            return hull;
        }}
        
        // Update error visualization
        function updateErrorDisplay() {{
            svg.selectAll(".css-node[data-node-type='qubit']").each(function(d) {{
                const node = d3.select(this);
                const hasXError = xErrors.has(d.id);
                const hasZError = zErrors.has(d.id);
                const hasYError = yErrors.has(d.id);
                
                node.classed("error", hasXError || hasZError || hasYError);
                
                let fill = "white";
                let labelText = "";
                if (hasYError) {{
                    fill = "#ec4899";  // Pink
                    labelText = "Y";
                }} else if (hasXError && hasZError) {{
                    fill = "#9333ea";  // Purple
                    labelText = "XZ";
                }} else if (hasXError) {{
                    fill = "#ef4444";  // Red
                    labelText = "X";
                }} else if (hasZError) {{
                    fill = "#3b82f6";  // Blue
                    labelText = "Z";
                }}
                
                node.select("circle").attr("fill", fill);
                
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
            const xStabBtn = document.getElementById('select-x-stab-btn');
            const zStabBtn = document.getElementById('select-z-stab-btn');
            const modeDisplay = document.getElementById('mode-display');
            const errorVectorLabel = document.getElementById('error-vector-label');
            
            if (errorMode === mode) {{
                errorMode = null;
                xBtn.classList.remove('active');
                zBtn.classList.remove('active');
                yBtn.classList.remove('active');
                xStabBtn.classList.remove('active');
                zStabBtn.classList.remove('active');
                modeDisplay.textContent = 'None - Click a button above';
                modeDisplay.className = 'text-blue-400';
                errorVectorLabel.textContent = 'Error Vector:';
            }} else {{
                errorMode = mode;
                xBtn.classList.toggle('active', mode === 'X');
                zBtn.classList.toggle('active', mode === 'Z');
                yBtn.classList.toggle('active', mode === 'Y');
                xStabBtn.classList.toggle('active', mode === 'X_STAB');
                zStabBtn.classList.toggle('active', mode === 'Z_STAB');
                
                if (mode === 'X') {{
                    modeDisplay.textContent = 'X Error Injection (Red)';
                    modeDisplay.className = 'text-red-400';
                    errorVectorLabel.textContent = 'Error Vector:';
                }} else if (mode === 'Z') {{
                    modeDisplay.textContent = 'Z Error Injection (Blue)';
                    modeDisplay.className = 'text-blue-400';
                    errorVectorLabel.textContent = 'Error Vector:';
                }} else if (mode === 'Y') {{
                    modeDisplay.textContent = 'Y Error Injection (Pink)';
                    modeDisplay.className = 'text-pink-400';
                    errorVectorLabel.textContent = 'Error Vector:';
                }} else if (mode === 'X_STAB') {{
                    modeDisplay.textContent = 'X Stabilizer Selection (Orange)';
                    modeDisplay.className = 'text-orange-400';
                    errorVectorLabel.textContent = 'Selected Stabilizers:';
                }} else if (mode === 'Z_STAB') {{
                    modeDisplay.textContent = 'Z Stabilizer Selection (Teal)';
                    modeDisplay.className = 'text-teal-400';
                    errorVectorLabel.textContent = 'Selected Stabilizers:';
                }}
            }}
            
            // Update the display
            updateErrorVectorDisplay();
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
            selectedStabilizers.clear();
            
            updateErrorDisplay();
            updateSyndromeDisplay();
            updateErrorVectorDisplay();
            updateSyndromeTextDisplay();
            updateStabilizerOverlays();
        }}
        
        // Initialize on page load
        init();
    </script>
</body>
</html>"""
    
    # Write to file
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"\n✅ Generated interactive surface code website: {output_file}")
    print(f"\nFeatures:")
    print(f"  ✓ Rotated Surface Code (d=3,5,7,9,11)")
    print(f"  ✓ Surface Code (d=2-11)")
    print(f"  ✓ Dropdown selectors for code type and distance")
    print(f"  ✓ Show/hide X-type and Z-type elements")
    print(f"  ✓ Click-to-inject X, Y, and Z errors")
    print(f"  ✓ Real-time syndrome calculation (JavaScript)")
    print(f"  ✓ All data embedded (works offline)")
    print(f"  ✓ Modern UI with Tailwind CSS")
    print(f"\nOpen {output_file} in your browser to use the interactive visualization!")


if __name__ == "__main__":
    generate_interactive_surface_code_website(output_file="surface_code_interactive.html")
