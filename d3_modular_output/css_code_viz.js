
// CSSCodeViz - Embeddable CSS Code Visualization Module
// This module can be included in any D3.js application

(function(global) {
    'use strict';
    
    const CSSCodeViz = {};
    
    // Visualization data
    CSSCodeViz.data = {
        nodes: [
  {
    "id": "q0",
    "x": 56.5,
    "y": 69.24264068711929,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_0",
    "display_label": "Q_0",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "q1",
    "x": 136.5,
    "y": 69.24264068711929,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_1",
    "display_label": "Q_1",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "q2",
    "x": 216.5,
    "y": 69.24264068711929,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_2",
    "display_label": "Q_2",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "q3",
    "x": 296.5,
    "y": 69.24264068711929,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_3",
    "display_label": "Q_3",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "q4",
    "x": 376.5,
    "y": 69.24264068711929,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_4",
    "display_label": "Q_4",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "q5",
    "x": 56.5,
    "y": 149.2426406871193,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_5",
    "display_label": "Q_5",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "q6",
    "x": 136.5,
    "y": 149.2426406871193,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_6",
    "display_label": "Q_6",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "q7",
    "x": 216.5,
    "y": 149.2426406871193,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_7",
    "display_label": "Q_7",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "q8",
    "x": 296.5,
    "y": 149.2426406871193,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_8",
    "display_label": "Q_8",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "q9",
    "x": 376.5,
    "y": 149.2426406871193,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_9",
    "display_label": "Q_9",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "q10",
    "x": 56.5,
    "y": 229.2426406871193,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_10",
    "display_label": "Q_10",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "q11",
    "x": 136.5,
    "y": 229.2426406871193,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_11",
    "display_label": "Q_11",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "q12",
    "x": 216.5,
    "y": 229.2426406871193,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_12",
    "display_label": "Q_12",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "q13",
    "x": 296.5,
    "y": 229.2426406871193,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_13",
    "display_label": "Q_13",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "q14",
    "x": 376.5,
    "y": 229.2426406871193,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_14",
    "display_label": "Q_14",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "q15",
    "x": 56.5,
    "y": 309.24264068711926,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_15",
    "display_label": "Q_15",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "q16",
    "x": 136.5,
    "y": 309.24264068711926,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_16",
    "display_label": "Q_16",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "q17",
    "x": 216.5,
    "y": 309.24264068711926,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_17",
    "display_label": "Q_17",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "q18",
    "x": 296.5,
    "y": 309.24264068711926,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_18",
    "display_label": "Q_18",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "q19",
    "x": 376.5,
    "y": 309.24264068711926,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_19",
    "display_label": "Q_19",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "q20",
    "x": 56.5,
    "y": 389.24264068711926,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_20",
    "display_label": "Q_20",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "q21",
    "x": 136.5,
    "y": 389.24264068711926,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_21",
    "display_label": "Q_21",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "q22",
    "x": 216.5,
    "y": 389.24264068711926,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_22",
    "display_label": "Q_22",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "q23",
    "x": 296.5,
    "y": 389.24264068711926,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_23",
    "display_label": "Q_23",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "q24",
    "x": 376.5,
    "y": 389.24264068711926,
    "type": "qubit",
    "type_label": "Qubit",
    "tooltip_label": "Q_24",
    "display_label": "Q_24",
    "label_dx": 12.727922061357855,
    "label_dy": -12.727922061357855,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 12,
    "color": "#000000",
    "fill": "#0091ff"
  },
  {
    "id": "x0",
    "x": 96.5,
    "y": 109.24264068711929,
    "type": "x_check",
    "type_label": "X stabilizer",
    "tooltip_label": "X_0",
    "display_label": "X_0",
    "label_dx": 19.242640687119284,
    "label_dy": -19.242640687119284,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 15,
    "color": "#000000",
    "fill": "white"
  },
  {
    "id": "x1",
    "x": 256.5,
    "y": 109.24264068711929,
    "type": "x_check",
    "type_label": "X stabilizer",
    "tooltip_label": "X_1",
    "display_label": "X_1",
    "label_dx": 19.242640687119284,
    "label_dy": -19.242640687119284,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 15,
    "color": "#000000",
    "fill": "white"
  },
  {
    "id": "x2",
    "x": 176.5,
    "y": 189.2426406871193,
    "type": "x_check",
    "type_label": "X stabilizer",
    "tooltip_label": "X_2",
    "display_label": "X_2",
    "label_dx": 19.242640687119284,
    "label_dy": -19.242640687119284,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 15,
    "color": "#000000",
    "fill": "white"
  },
  {
    "id": "x3",
    "x": 336.5,
    "y": 189.2426406871193,
    "type": "x_check",
    "type_label": "X stabilizer",
    "tooltip_label": "X_3",
    "display_label": "X_3",
    "label_dx": 19.242640687119284,
    "label_dy": -19.242640687119284,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 15,
    "color": "#000000",
    "fill": "white"
  },
  {
    "id": "x4",
    "x": 96.5,
    "y": 269.24264068711926,
    "type": "x_check",
    "type_label": "X stabilizer",
    "tooltip_label": "X_4",
    "display_label": "X_4",
    "label_dx": 19.242640687119284,
    "label_dy": -19.242640687119284,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 15,
    "color": "#000000",
    "fill": "white"
  },
  {
    "id": "x5",
    "x": 256.5,
    "y": 269.24264068711926,
    "type": "x_check",
    "type_label": "X stabilizer",
    "tooltip_label": "X_5",
    "display_label": "X_5",
    "label_dx": 19.242640687119284,
    "label_dy": -19.242640687119284,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 15,
    "color": "#000000",
    "fill": "white"
  },
  {
    "id": "x6",
    "x": 176.5,
    "y": 349.24264068711926,
    "type": "x_check",
    "type_label": "X stabilizer",
    "tooltip_label": "X_6",
    "display_label": "X_6",
    "label_dx": 19.242640687119284,
    "label_dy": -19.242640687119284,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 15,
    "color": "#000000",
    "fill": "white"
  },
  {
    "id": "x7",
    "x": 336.5,
    "y": 349.24264068711926,
    "type": "x_check",
    "type_label": "X stabilizer",
    "tooltip_label": "X_7",
    "display_label": "X_7",
    "label_dx": 19.242640687119284,
    "label_dy": -19.242640687119284,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 15,
    "color": "#000000",
    "fill": "white"
  },
  {
    "id": "x8",
    "x": 16.5,
    "y": 189.2426406871193,
    "type": "x_check",
    "type_label": "X stabilizer",
    "tooltip_label": "X_8",
    "display_label": "X_8",
    "label_dx": 19.242640687119284,
    "label_dy": -19.242640687119284,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 15,
    "color": "#000000",
    "fill": "white"
  },
  {
    "id": "x9",
    "x": 16.5,
    "y": 349.24264068711926,
    "type": "x_check",
    "type_label": "X stabilizer",
    "tooltip_label": "X_9",
    "display_label": "X_9",
    "label_dx": 19.242640687119284,
    "label_dy": -19.242640687119284,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 15,
    "color": "#000000",
    "fill": "white"
  },
  {
    "id": "x10",
    "x": 416.5,
    "y": 109.24264068711929,
    "type": "x_check",
    "type_label": "X stabilizer",
    "tooltip_label": "X_10",
    "display_label": "X_10",
    "label_dx": 19.242640687119284,
    "label_dy": -19.242640687119284,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 15,
    "color": "#000000",
    "fill": "white"
  },
  {
    "id": "x11",
    "x": 416.5,
    "y": 269.24264068711926,
    "type": "x_check",
    "type_label": "X stabilizer",
    "tooltip_label": "X_11",
    "display_label": "X_11",
    "label_dx": 19.242640687119284,
    "label_dy": -19.242640687119284,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 15,
    "color": "#000000",
    "fill": "white"
  },
  {
    "id": "z0",
    "x": 176.5,
    "y": 109.24264068711929,
    "type": "z_check",
    "type_label": "Z stabilizer",
    "tooltip_label": "Z_0",
    "display_label": "Z_0",
    "label_dx": 19.242640687119284,
    "label_dy": -19.242640687119284,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 15,
    "color": "#000000",
    "fill": "white"
  },
  {
    "id": "z1",
    "x": 336.5,
    "y": 109.24264068711929,
    "type": "z_check",
    "type_label": "Z stabilizer",
    "tooltip_label": "Z_1",
    "display_label": "Z_1",
    "label_dx": 19.242640687119284,
    "label_dy": -19.242640687119284,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 15,
    "color": "#000000",
    "fill": "white"
  },
  {
    "id": "z2",
    "x": 96.5,
    "y": 189.2426406871193,
    "type": "z_check",
    "type_label": "Z stabilizer",
    "tooltip_label": "Z_2",
    "display_label": "Z_2",
    "label_dx": 19.242640687119284,
    "label_dy": -19.242640687119284,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 15,
    "color": "#000000",
    "fill": "white"
  },
  {
    "id": "z3",
    "x": 256.5,
    "y": 189.2426406871193,
    "type": "z_check",
    "type_label": "Z stabilizer",
    "tooltip_label": "Z_3",
    "display_label": "Z_3",
    "label_dx": 19.242640687119284,
    "label_dy": -19.242640687119284,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 15,
    "color": "#000000",
    "fill": "white"
  },
  {
    "id": "z4",
    "x": 176.5,
    "y": 269.24264068711926,
    "type": "z_check",
    "type_label": "Z stabilizer",
    "tooltip_label": "Z_4",
    "display_label": "Z_4",
    "label_dx": 19.242640687119284,
    "label_dy": -19.242640687119284,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 15,
    "color": "#000000",
    "fill": "white"
  },
  {
    "id": "z5",
    "x": 336.5,
    "y": 269.24264068711926,
    "type": "z_check",
    "type_label": "Z stabilizer",
    "tooltip_label": "Z_5",
    "display_label": "Z_5",
    "label_dx": 19.242640687119284,
    "label_dy": -19.242640687119284,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 15,
    "color": "#000000",
    "fill": "white"
  },
  {
    "id": "z6",
    "x": 96.5,
    "y": 349.24264068711926,
    "type": "z_check",
    "type_label": "Z stabilizer",
    "tooltip_label": "Z_6",
    "display_label": "Z_6",
    "label_dx": 19.242640687119284,
    "label_dy": -19.242640687119284,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 15,
    "color": "#000000",
    "fill": "white"
  },
  {
    "id": "z7",
    "x": 256.5,
    "y": 349.24264068711926,
    "type": "z_check",
    "type_label": "Z stabilizer",
    "tooltip_label": "Z_7",
    "display_label": "Z_7",
    "label_dx": 19.242640687119284,
    "label_dy": -19.242640687119284,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 15,
    "color": "#000000",
    "fill": "white"
  },
  {
    "id": "z8",
    "x": 96.5,
    "y": 29.242640687119284,
    "type": "z_check",
    "type_label": "Z stabilizer",
    "tooltip_label": "Z_8",
    "display_label": "Z_8",
    "label_dx": 19.242640687119284,
    "label_dy": -19.242640687119284,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 15,
    "color": "#000000",
    "fill": "white"
  },
  {
    "id": "z9",
    "x": 256.5,
    "y": 29.242640687119284,
    "type": "z_check",
    "type_label": "Z stabilizer",
    "tooltip_label": "Z_9",
    "display_label": "Z_9",
    "label_dx": 19.242640687119284,
    "label_dy": -19.242640687119284,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 15,
    "color": "#000000",
    "fill": "white"
  },
  {
    "id": "z10",
    "x": 176.5,
    "y": 429.24264068711926,
    "type": "z_check",
    "type_label": "Z stabilizer",
    "tooltip_label": "Z_10",
    "display_label": "Z_10",
    "label_dx": 19.242640687119284,
    "label_dy": -19.242640687119284,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 15,
    "color": "#000000",
    "fill": "white"
  },
  {
    "id": "z11",
    "x": 336.5,
    "y": 429.24264068711926,
    "type": "z_check",
    "type_label": "Z stabilizer",
    "tooltip_label": "Z_11",
    "display_label": "Z_11",
    "label_dx": 19.242640687119284,
    "label_dy": -19.242640687119284,
    "label_anchor": "start",
    "label_baseline": "alphabetic",
    "radius": 15,
    "color": "#000000",
    "fill": "white"
  }
],
        edges: [
  {
    "x1": 56.5,
    "y1": 69.24264068711929,
    "x2": 96.5,
    "y2": 109.24264068711929,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 136.5,
    "y1": 69.24264068711929,
    "x2": 96.5,
    "y2": 109.24264068711929,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 56.5,
    "y1": 149.2426406871193,
    "x2": 96.5,
    "y2": 109.24264068711929,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 136.5,
    "y1": 149.2426406871193,
    "x2": 96.5,
    "y2": 109.24264068711929,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 216.5,
    "y1": 69.24264068711929,
    "x2": 256.5,
    "y2": 109.24264068711929,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 296.5,
    "y1": 69.24264068711929,
    "x2": 256.5,
    "y2": 109.24264068711929,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 216.5,
    "y1": 149.2426406871193,
    "x2": 256.5,
    "y2": 109.24264068711929,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 296.5,
    "y1": 149.2426406871193,
    "x2": 256.5,
    "y2": 109.24264068711929,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 136.5,
    "y1": 149.2426406871193,
    "x2": 176.5,
    "y2": 189.2426406871193,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 216.5,
    "y1": 149.2426406871193,
    "x2": 176.5,
    "y2": 189.2426406871193,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 136.5,
    "y1": 229.2426406871193,
    "x2": 176.5,
    "y2": 189.2426406871193,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 216.5,
    "y1": 229.2426406871193,
    "x2": 176.5,
    "y2": 189.2426406871193,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 296.5,
    "y1": 149.2426406871193,
    "x2": 336.5,
    "y2": 189.2426406871193,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 376.5,
    "y1": 149.2426406871193,
    "x2": 336.5,
    "y2": 189.2426406871193,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 296.5,
    "y1": 229.2426406871193,
    "x2": 336.5,
    "y2": 189.2426406871193,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 376.5,
    "y1": 229.2426406871193,
    "x2": 336.5,
    "y2": 189.2426406871193,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 56.5,
    "y1": 229.2426406871193,
    "x2": 96.5,
    "y2": 269.24264068711926,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 136.5,
    "y1": 229.2426406871193,
    "x2": 96.5,
    "y2": 269.24264068711926,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 56.5,
    "y1": 309.24264068711926,
    "x2": 96.5,
    "y2": 269.24264068711926,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 136.5,
    "y1": 309.24264068711926,
    "x2": 96.5,
    "y2": 269.24264068711926,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 216.5,
    "y1": 229.2426406871193,
    "x2": 256.5,
    "y2": 269.24264068711926,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 296.5,
    "y1": 229.2426406871193,
    "x2": 256.5,
    "y2": 269.24264068711926,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 216.5,
    "y1": 309.24264068711926,
    "x2": 256.5,
    "y2": 269.24264068711926,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 296.5,
    "y1": 309.24264068711926,
    "x2": 256.5,
    "y2": 269.24264068711926,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 136.5,
    "y1": 309.24264068711926,
    "x2": 176.5,
    "y2": 349.24264068711926,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 216.5,
    "y1": 309.24264068711926,
    "x2": 176.5,
    "y2": 349.24264068711926,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 136.5,
    "y1": 389.24264068711926,
    "x2": 176.5,
    "y2": 349.24264068711926,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 216.5,
    "y1": 389.24264068711926,
    "x2": 176.5,
    "y2": 349.24264068711926,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 296.5,
    "y1": 309.24264068711926,
    "x2": 336.5,
    "y2": 349.24264068711926,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 376.5,
    "y1": 309.24264068711926,
    "x2": 336.5,
    "y2": 349.24264068711926,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 296.5,
    "y1": 389.24264068711926,
    "x2": 336.5,
    "y2": 349.24264068711926,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 376.5,
    "y1": 389.24264068711926,
    "x2": 336.5,
    "y2": 349.24264068711926,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 56.5,
    "y1": 149.2426406871193,
    "x2": 16.5,
    "y2": 189.2426406871193,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 56.5,
    "y1": 229.2426406871193,
    "x2": 16.5,
    "y2": 189.2426406871193,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 56.5,
    "y1": 309.24264068711926,
    "x2": 16.5,
    "y2": 349.24264068711926,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 56.5,
    "y1": 389.24264068711926,
    "x2": 16.5,
    "y2": 349.24264068711926,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 376.5,
    "y1": 69.24264068711929,
    "x2": 416.5,
    "y2": 109.24264068711929,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 376.5,
    "y1": 149.2426406871193,
    "x2": 416.5,
    "y2": 109.24264068711929,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 376.5,
    "y1": 229.2426406871193,
    "x2": 416.5,
    "y2": 269.24264068711926,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 376.5,
    "y1": 309.24264068711926,
    "x2": 416.5,
    "y2": 269.24264068711926,
    "color": "black",
    "style": "solid",
    "width": 3,
    "type": "x_check"
  },
  {
    "x1": 136.5,
    "y1": 69.24264068711929,
    "x2": 176.5,
    "y2": 109.24264068711929,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 216.5,
    "y1": 69.24264068711929,
    "x2": 176.5,
    "y2": 109.24264068711929,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 136.5,
    "y1": 149.2426406871193,
    "x2": 176.5,
    "y2": 109.24264068711929,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 216.5,
    "y1": 149.2426406871193,
    "x2": 176.5,
    "y2": 109.24264068711929,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 296.5,
    "y1": 69.24264068711929,
    "x2": 336.5,
    "y2": 109.24264068711929,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 376.5,
    "y1": 69.24264068711929,
    "x2": 336.5,
    "y2": 109.24264068711929,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 296.5,
    "y1": 149.2426406871193,
    "x2": 336.5,
    "y2": 109.24264068711929,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 376.5,
    "y1": 149.2426406871193,
    "x2": 336.5,
    "y2": 109.24264068711929,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 56.5,
    "y1": 149.2426406871193,
    "x2": 96.5,
    "y2": 189.2426406871193,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 136.5,
    "y1": 149.2426406871193,
    "x2": 96.5,
    "y2": 189.2426406871193,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 56.5,
    "y1": 229.2426406871193,
    "x2": 96.5,
    "y2": 189.2426406871193,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 136.5,
    "y1": 229.2426406871193,
    "x2": 96.5,
    "y2": 189.2426406871193,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 216.5,
    "y1": 149.2426406871193,
    "x2": 256.5,
    "y2": 189.2426406871193,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 296.5,
    "y1": 149.2426406871193,
    "x2": 256.5,
    "y2": 189.2426406871193,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 216.5,
    "y1": 229.2426406871193,
    "x2": 256.5,
    "y2": 189.2426406871193,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 296.5,
    "y1": 229.2426406871193,
    "x2": 256.5,
    "y2": 189.2426406871193,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 136.5,
    "y1": 229.2426406871193,
    "x2": 176.5,
    "y2": 269.24264068711926,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 216.5,
    "y1": 229.2426406871193,
    "x2": 176.5,
    "y2": 269.24264068711926,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 136.5,
    "y1": 309.24264068711926,
    "x2": 176.5,
    "y2": 269.24264068711926,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 216.5,
    "y1": 309.24264068711926,
    "x2": 176.5,
    "y2": 269.24264068711926,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 296.5,
    "y1": 229.2426406871193,
    "x2": 336.5,
    "y2": 269.24264068711926,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 376.5,
    "y1": 229.2426406871193,
    "x2": 336.5,
    "y2": 269.24264068711926,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 296.5,
    "y1": 309.24264068711926,
    "x2": 336.5,
    "y2": 269.24264068711926,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 376.5,
    "y1": 309.24264068711926,
    "x2": 336.5,
    "y2": 269.24264068711926,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 56.5,
    "y1": 309.24264068711926,
    "x2": 96.5,
    "y2": 349.24264068711926,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 136.5,
    "y1": 309.24264068711926,
    "x2": 96.5,
    "y2": 349.24264068711926,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 56.5,
    "y1": 389.24264068711926,
    "x2": 96.5,
    "y2": 349.24264068711926,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 136.5,
    "y1": 389.24264068711926,
    "x2": 96.5,
    "y2": 349.24264068711926,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 216.5,
    "y1": 309.24264068711926,
    "x2": 256.5,
    "y2": 349.24264068711926,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 296.5,
    "y1": 309.24264068711926,
    "x2": 256.5,
    "y2": 349.24264068711926,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 216.5,
    "y1": 389.24264068711926,
    "x2": 256.5,
    "y2": 349.24264068711926,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 296.5,
    "y1": 389.24264068711926,
    "x2": 256.5,
    "y2": 349.24264068711926,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 56.5,
    "y1": 69.24264068711929,
    "x2": 96.5,
    "y2": 29.242640687119284,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 136.5,
    "y1": 69.24264068711929,
    "x2": 96.5,
    "y2": 29.242640687119284,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 216.5,
    "y1": 69.24264068711929,
    "x2": 256.5,
    "y2": 29.242640687119284,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 296.5,
    "y1": 69.24264068711929,
    "x2": 256.5,
    "y2": 29.242640687119284,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 136.5,
    "y1": 389.24264068711926,
    "x2": 176.5,
    "y2": 429.24264068711926,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 216.5,
    "y1": 389.24264068711926,
    "x2": 176.5,
    "y2": 429.24264068711926,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 296.5,
    "y1": 389.24264068711926,
    "x2": 336.5,
    "y2": 429.24264068711926,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  },
  {
    "x1": 376.5,
    "y1": 389.24264068711926,
    "x2": 336.5,
    "y2": 429.24264068711926,
    "color": "black",
    "style": "dashed",
    "width": 3,
    "type": "z_check"
  }
],
        config: {
  "width": 460,
  "height": 446,
  "label_fontsize": 10,
  "background_color": "transparent"
}
    };
    
    // Core rendering function
    CSSCodeViz.render = function(svgSelector, options) {
        options = options || {};
        const data = this.data;
        const config = { ...data.config, ...options };
        
        // Get or create SVG element
        let svg = d3.select(svgSelector);
        if (svg.empty()) {
            console.error('SVG element not found:', svgSelector);
            return null;
        }
        
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
            .attr("class", d => `css-edge css-edge-${d.type}`)
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
            .attr("class", d => `css-node css-node-${d.type}`)
            .attr("transform", d => `translate(${d.x},${d.y})`)
            .attr("data-node-id", d => d.id);
        
        // Draw node shapes
        nodes.each(function(d) {
            const node = d3.select(this);
            if (d.type === "qubit") {
                node.append("circle")
                    .attr("r", d.radius)
                    .attr("fill", d.fill)
                    .attr("stroke", d.color)
                    .attr("stroke-width", 2);
            } else {
                node.append("rect")
                    .attr("x", -d.radius)
                    .attr("y", -d.radius)
                    .attr("width", d.radius * 2)
                    .attr("height", d.radius * 2)
                    .attr("fill", d.fill)
                    .attr("stroke", d.color)
                    .attr("stroke-width", 2);
            }
        });
        
        // Add labels if enabled
        if (config.showLabels !== false) {
            nodes.append("text")
                .attr("class", "css-node-label")
                .attr("x", d => d.label_dx || 10)
                .attr("y", d => d.label_dy || -10)
                .attr("font-size", config.label_fontsize || 12)
                .attr("text-anchor", d => d.label_anchor || "start")
                .text(d => d.display_label || d.label || d.id);
        }
        
        // Add tooltips
        nodes.append("title")
            .text(d => `${d.tooltip_label || d.label || d.id} (${d.type_label || d.type})`);
        
        return { svg, nodes, edges };
    };
    
    // Update node colors (useful for error visualization)
    CSSCodeViz.updateNodeColors = function(svgSelector, nodeColorMap) {
        const svg = d3.select(svgSelector);
        Object.keys(nodeColorMap).forEach(nodeId => {
            svg.select(`[data-node-id="${nodeId}"]`)
                .select("circle, rect")
                .transition()
                .duration(300)
                .attr("fill", nodeColorMap[nodeId]);
        });
    };
    
    // Highlight errors on specific nodes
    CSSCodeViz.highlightErrors = function(svgSelector, errorNodeIds, errorColor) {
        errorColor = errorColor || "#ff0000";
        const colorMap = {};
        errorNodeIds.forEach(id => {
            colorMap[id] = errorColor;
        });
        this.updateNodeColors(svgSelector, colorMap);
    };
    
    // Clear all error highlights
    CSSCodeViz.clearErrors = function(svgSelector) {
        const svg = d3.select(svgSelector);
        const data = this.data;
        svg.selectAll(".css-node")
            .select("circle, rect")
            .transition()
            .duration(300)
            .attr("fill", (d, i) => data.nodes[i].fill);
    };
    
    // Toggle edge visibility by type
    CSSCodeViz.toggleEdges = function(svgSelector, edgeType, visible) {
        const svg = d3.select(svgSelector);
        svg.selectAll(`.css-edge-${edgeType}`)
            .transition()
            .duration(200)
            .style("opacity", visible ? 1 : 0);
    };
    
    // Export to global scope
    global.CSSCodeViz = CSSCodeViz;
    
    // Also support CommonJS and AMD
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = CSSCodeViz;
    }
    
})(typeof window !== 'undefined' ? window : this);
