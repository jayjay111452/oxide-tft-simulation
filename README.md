# OxideTFT-Studio: High-Fidelity IGZO TFT Simulation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**OxideTFT-Studio** is a high-precision, interactive physics simulation tool for Amorphous Oxide Semiconductor Thin-Film Transistors (AOS-TFTs), specifically focused on IGZO technology. 

It uses the **Finite Difference Method (FDM)** to solve the 2D Poisson equation self-consistently with carrier statistics, allowing researchers and engineers to visualize potential distribution, carrier concentration, and electric fields in real-time.

🛠️ Usage

website：https://oxide-tft-simulation-z3wjys5v3czfmapanua62v.streamlit.app

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


# OxideTFT-Studio: 高精度 IGZO TFT 物理仿真工具

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**OxideTFT-Studio** 是一款专为非晶氧化物半导体薄膜晶体管（AOS-TFTs）开发的高精度交互式物理仿真工具，重点针对 IGZO 技术进行了优化。

该项目使用**有限差分法 (FDM)** 对二维泊松方程进行自洽求解，结合载流子统计模型，允许研究人员和工程师实时可视化器件内部的电势分布、载流子浓度以及电场强度。

🛠️ Usage 

传送门：https://oxide-tft-simulation-z3wjys5v3czfmapanua62v.streamlit.app

## 🌟 核心功能

* **多结构支持 (Multi-Structure Support):**
    * 支持模拟多种器件架构：双栅 (Double Gate)、单栅 (顶栅/底栅)、源控晶体管 (Source-Gated)。
* **高级叠层 Buffer 模型 (Stacked Buffer Model):**
    * 支持复合介质层模拟：**SiN (氮化硅) + SiO (氧化硅)** 叠层结构。
    * 支持自定义所有层（Buffer, 有源层 IGZO, GI）的厚度和介电常数。
* **高精度网格 (High-Precision Mesh):**
    * 用户可自定义网格密度（有源区支持 1000+ 网格点）。
    * 针对超薄沟道（典型值 50nm）内置了物理感知的子采样算法，实现高清渲染。
* **交互式可视化 (Interactive Visualization):**
    * 提供载流子浓度 ($n$) 和电场 ($E$) 的高分辨率热力图。
    * 支持垂直方向的 1D 切面分析 (Cut-line profile)。
    * 基于 **Plotly** 引擎，支持图表的实时交互、缩放和数据探查。

## 🛠️ 安装指南

1.  **克隆仓库**
    ```bash
    git clone [https://github.com/your-username/oxide-tft-simulation.git](https://github.com/your-username/oxide-tft-simulation.git)
    cd oxide-tft-simulation
    ```

2.  **安装依赖**
    建议在 Python 虚拟环境中运行。
    ```bash
    pip install -r requirements.txt
    ```

3.  **运行仿真**
    ```bash
    streamlit run app.py
    ```
    启动后，工具将自动在你的默认浏览器中打开，地址通常为 `http://localhost:8501`。

## 📐 物理引擎说明

核心求解器 (`solver.py`) 实现了泊松方程的数值解：

$$ \nabla \cdot (\epsilon \nabla \phi) = -q (N_{d}^{+} - n) $$

其中：
* $n$ 为电子浓度，由费米-狄拉克统计计算（针对氧化物半导体进行了近似）。
* $\phi$ 为静电势。
* $\epsilon$ 为位置相关的介电常数（分别对应 SiN, SiO, IGZO 等区域）。

该非线性系统通过牛顿-拉夫逊 (Newton-Raphson) 迭代法求解，直至达到收敛标准。

## 📂 项目结构

* `app.py`: Streamlit 前端界面。负责处理用户输入、调用求解器并渲染 Plotly 图表。
* `solver.py`: 物理引擎后端。包含 `TFTPoissonSolver` 类、自适应网格生成逻辑以及有限差分矩阵构建算法。
* `requirements.txt`: 项目所需的 Python 依赖库列表 (numpy, scipy, streamlit, plotly)。

