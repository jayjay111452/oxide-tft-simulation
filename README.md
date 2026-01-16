# OxideTFT-Studio: High-Fidelity IGZO TFT Simulation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**OxideTFT-Studio** is a high-precision, interactive physics simulation tool for Amorphous Oxide Semiconductor Thin-Film Transistors (AOS-TFTs), specifically focused on IGZO technology. 

It uses the **Finite Difference Method (FDM)** to solve the 2D Poisson equation self-consistently with carrier statistics, allowing researchers and engineers to visualize potential distribution, carrier concentration, and electric fields in real-time.

## 🌟 Key Features

* **Multi-Structure Support:** Simulate various device architectures:
    * Double Gate
    * Single Gate (Top/Bottom)
    * Source-Gated Transistor
* **Advanced Stacked Buffer Model:**
    * Support for composite dielectric layers: **SiN (Nitride) + SiO (Oxide)** buffer stacks.
    * Configurable thickness and permittivity for all layers (Buffer, Active IGZO, GI).
* **High-Precision Mesh:**
    * User-defined mesh density (up to 1000+ points in the active region).
    * Physics-aware sub-sampling for high-definition rendering of the thin active channel (50nm typical).
* **Interactive Visualization:**
    * Heatmaps for Carrier Concentration ($n$) and Electric Field ($E$).
    * Vertical 1D cut-lines for profile analysis.
    * Powered by **Plotly** for interactive zooming and inspection.

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/your-username/oxide-tft-simulation.git](https://github.com/your-username/oxide-tft-simulation.git)
    cd oxide-tft-simulation
    ```

2.  **Install dependencies**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Simulation**
    ```bash
    streamlit run app.py
    ```
    The tool will open automatically in your web browser at `http://localhost:8501`.

## 📐 Physics Engine

The core solver (`solver.py`) implements a numerical solution for the Poisson equation:

$$ \nabla \cdot (\epsilon \nabla \phi) = -q (N_{d}^{+} - n) $$

Where:
* $n$ is the electron density calculated using Fermi-Dirac statistics (approximated for oxide semiconductors).
* $\phi$ is the electrostatic potential.
* $\epsilon$ is the position-dependent permittivity (SiN, SiO, IGZO).

The non-linear system is solved using a Newton-Raphson iteration loop until convergence is achieved.

## 📂 Project Structure

* `app.py`: The Streamlit frontend interface. Handles user input, calls the solver, and renders Plotly charts.
* `solver.py`: The physics engine. Contains the `TFTPoissonSolver` class, mesh generation logic, and the finite difference matrix builder.
* `requirements.txt`: List of Python dependencies (numpy, scipy, streamlit, plotly).

## 📸 Screenshots

*(You can add a screenshot of your running app here)*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open-source and available under the MIT License.
