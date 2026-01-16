import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from solver import TFTPoissonSolver
import time

st.set_page_config(layout="wide", page_title="Oxide TFT HD")

st.title("Oxide TFT Simulation (High Precision)")

# --- Sidebar ---
st.sidebar.header("1. 器件结构 (Structure)")
struct_type = st.sidebar.selectbox(
    "偏置模式",
    ('Double Gate', 'Single Gate (Top)', 'Single Gate (Bottom)', 'Source-Gated Bottom')
)

st.sidebar.header("2. 几何尺寸 (Geometry)")
L_um = st.sidebar.number_input("沟道长度 L (um)", value=2.0)
t_buf_nm = st.sidebar.number_input("Buffer层厚度 (nm)", value=200.0)
t_igzo_nm = st.sidebar.number_input("IGZO层厚度 (nm)", value=50.0)
t_gi_nm = st.sidebar.number_input("GI层厚度 (nm)", value=100.0)

st.sidebar.header("3. 网格设置 (Mesh Setting)")
# 这里强制更新：Y轴最大1000，X轴最大200
ny_igzo = st.sidebar.slider("IGZO层 Y轴网格点数 (Max 1000)", 100, 1000, 400)
nx = st.sidebar.slider("X轴网格点数 (Max 200)", 20, 200, 50)

st.sidebar.header("4. 电压偏置 (Bias)")
v_tg = st.sidebar.slider("顶栅 Vtg (V)", -10.0, 20.0, 10.0)
v_bg = st.sidebar.slider("底栅 Vbg (V)", -10.0, 20.0, 10.0)
v_ds = st.sidebar.slider("漏极 Vds (V)", 0.0, 20.0, 5.0)

if st.sidebar.button("开始仿真 (RUN)", type="primary"):
    with st.spinner(f"正在计算... (IGZO层物理网格点: {ny_igzo})"):
        solver = TFTPoissonSolver(
            length=L_um,
            t_buffer=t_buf_nm/1000.0, eps_buffer=6.0,
            t_igzo=t_igzo_nm/1000.0, eps_igzo=10.0, nd_igzo=1e16,
            t_gi=t_gi_nm/1000.0, eps_gi=3.9,
            structure_type=struct_type,
            nx=nx, # 传入用户设定的 X
            ny=ny_igzo # 传入用户设定的 Y
        )
        
        start = time.time()
        phi, n_conc, E = solver.solve(v_top_gate_bias=v_tg, v_ds=v_ds, v_bot_gate_bias=v_bg)
        elapsed = time.time() - start
        
        st.success(f"计算完成，耗时 {elapsed:.3f}秒")
        
        # --- 物理级重采样渲染 ---
        y_phys_cm = solver.y
        x_phys_cm = solver.x
        
        # 1. 提取 IGZO 区域
        tol = 1e-10
        y_igzo_start = t_buf_nm / 1e7
        y_igzo_end = (t_buf_nm + t_igzo_nm) / 1e7
        
        idx_igzo = np.where((y_phys_cm >= y_igzo_start - tol) & 
                            (y_phys_cm <= y_igzo_end + tol))[0]
        
        y_igzo_subset = y_phys_cm[idx_igzo]
        phi_igzo_subset = phi[idx_igzo, :]
        
        # 2. 生成高清渲染网格 (800像素)
        target_render_ny = 800
        y_hd_cm = np.linspace(y_igzo_start, y_igzo_end, target_render_ny)
        
        # 3. 插值电势 (Potential) - 使用 Quadratic 保证平滑
        f_phi = interp1d(y_igzo_subset, phi_igzo_subset, axis=0, kind='quadratic')
        phi_hd = f_phi(y_hd_cm)
        
        # 4. 重算载流子 (Physics)
        v_ch_hd = np.zeros_like(phi_hd)
        for i in range(len(x_phys_cm)):
            v_ch_hd[:, i] = v_ds * (i / (len(x_phys_cm)-1))
            
        n_hd = solver.calculate_n_from_phi(phi_hd, v_ch_hd)
        n_log_hd = np.log10(n_hd + 1e-30)
        
        # 5. 绘图数据准备
        x_plot = x_phys_cm * 1e4
        y_plot = y_hd_cm * 1e7
        
        z_min, z_max = np.nanmin(n_log_hd), np.nanmax(n_log_hd)
        tick_vals = np.arange(np.floor(z_min), np.ceil(z_max)+0.1, 0.5)
        tick_text = [f"1e{v:.1f}" for v in tick_vals]

        tab1, tab2, tab3 = st.tabs(["载流子浓度 (Carrier)", "垂直切面 (Cut)", "电场分布 (E-Field)"])
        
        with tab1:
            st.subheader("电子浓度 (cm^-3)")
            fig = go.Figure(data=go.Heatmap(
                x=x_plot, y=y_plot, z=n_log_hd,
                colorscale='Jet',
                zsmooth='best', # 开启 GPU 平滑
                colorbar=dict(title='Concentration', tickvals=tick_vals, ticktext=tick_text),
                hovertemplate='Y: %{y:.2f} nm<br>n: 1e%{z:.2f}<extra></extra>'
            ))
            fig.update_layout(height=500, yaxis_title="IGZO Thickness (nm)")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.subheader("垂直切面 (Vertical Profile)")
            mid = n_hd.shape[1] // 2
            fig2 = go.Figure(go.Scatter(x=y_plot, y=n_hd[:, mid], mode='lines', name='n', line=dict(width=3)))
            fig2.update_layout(yaxis_type="log", yaxis_title="n (cm^-3)")
            st.plotly_chart(fig2, use_container_width=True)
            
        with tab3:
            st.subheader("电场分布 (V/cm)")
            E_subset = E[idx_igzo, :]
            f_E = interp1d(y_igzo_subset, E_subset, axis=0, kind='quadratic')
            E_hd = f_E(y_hd_cm)
            
            fig3 = go.Figure(go.Heatmap(
                x=x_plot, y=y_plot, z=E_hd, colorscale='Magma', zsmooth='best',
                colorbar=dict(title='Field (V/cm)', tickformat='.1e')
            ))
            st.plotly_chart(fig3, use_container_width=True)
