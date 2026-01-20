import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
from solver import TFTPoissonSolver
import time

st.set_page_config(layout="wide", page_title="Oxide TFT HD")

st.title("Oxide TFT Simulation (Stacked Buffer Model)")

# --- Sidebar ---
st.sidebar.header("1. 器件结构 (Structure)")
struct_type = st.sidebar.selectbox(
    "偏置模式",
    ('Double Gate', 'Single Gate (Top)')
)

st.sidebar.header("2. 几何尺寸 (Geometry)")
col_lw = st.sidebar.columns(2)
with col_lw[0]:
    L_um = st.number_input("沟道长度 L (um)", value=4.0)
with col_lw[1]:
    W_um = st.number_input("沟道宽度 W (um)", value=3.0)

st.sidebar.subheader("Buffer Layer (Bottom)")
col1, col2 = st.sidebar.columns(2)
with col1:
    t_sin_nm = st.number_input("SiN 厚度 (nm)", value=100.0)
    eps_sin = st.number_input("SiN 介电常数", value=7.0)
with col2:
    t_buf_sio_nm = st.number_input("Buf SiO 厚度 (nm)", value=300.0)
    eps_buf_sio = st.number_input("Buf SiO 介电常数", value=3.9)

st.sidebar.subheader("Active & GI Layer")
col3, col4 = st.sidebar.columns(2)
with col3:
    t_igzo_nm = st.number_input("IGZO层厚度 (nm)", value=25.0, help="典型值: 15-50 nm | 薄: 15-25 nm | 厚: 40-50 nm")
with col4:
    t_gi_nm = st.number_input("GI层 (Top SiO) 厚度 (nm)", value=140.0, help="典型值: 100-200 nm | 影响栅极电容和控制能力")
st.sidebar.caption("💡 IGZO厚度影响沟道电导；GI厚度影响栅控能力")
    
col5, col6 = st.sidebar.columns(2)
with col5:
    eps_igzo = st.number_input("IGZO 介电常数", value=10.0)
with col6:
    eps_gi = st.number_input("GI 介电常数", value=3.9)

nd_igzo = st.sidebar.number_input("初始载流子浓度 (cm^-3)", value=1e17, format="%.1e", min_value=1e10, max_value=1e20, 
                                   help="典型值: 1e15-1e18 | 低掺杂: 1e15-1e16 | 中等: 1e17 | 高掺杂: 1e18+")
st.sidebar.caption("💡 载流子浓度影响阈值电压和沟道电导")

st.sidebar.subheader("Interface Trap Density")
col_dit1, col_dit2 = st.sidebar.columns(2)
with col_dit1:
    dit_top = st.number_input("GI/IGZO 界面 (cm^-2)", value=3e10, format="%.1e", min_value=0.0, max_value=1e13,
                              help="典型值: 1e10-1e12 | 优质界面: <1e11 | 一般: 1e11-5e11 | 差: >1e12")
with col_dit2:
    dit_bottom = st.number_input("IGZO/Buffer 界面 (cm^-2)", value=5e10, format="%.1e", min_value=0.0, max_value=1e13,
                                  help="典型值: 1e10-1e12 | 优质界面: <1e11 | 一般: 1e11-5e11 | 差: >1e12")
st.sidebar.caption("💡 界面陷阱会降低等效栅压，影响电子浓度分布")

e_trap = st.sidebar.number_input("界面陷阱能级位置 (eV, 相对费米能级)", value=0.3, min_value=-1.5, max_value=1.5, step=0.1,
                                  help="典型值: 0.2-0.5 eV | 浅能级: <0.3 eV | 深能级: >0.5 eV")
st.sidebar.caption("💡 能级位置决定陷阱占据率，影响等效栅压降大小")

st.sidebar.subheader("Source/Drain Resistance")
col_sd1, col_sd2 = st.sidebar.columns(2)
with col_sd1:
    L_source_um = st.number_input("源极长度 (um)", value=3.0, min_value=0.0)
    Rs_sheet = st.number_input("源极方块电阻 (Ω/sq)", value=3700.0)
with col_sd2:
    L_drain_um = st.number_input("漏极长度 (um)", value=3.0, min_value=0.0)
    Rd_sheet = st.number_input("漏极方块电阻 (Ω/sq)", value=3700.0)

st.sidebar.header("3. 网格设置 (Mesh Setting)")
ny_igzo = st.sidebar.slider("IGZO层 Y轴网格点数 (Max 1000)", 100, 1000, 400)
nx = st.sidebar.slider("X轴网格点数 (Max 200)", 20, 200, 50)

st.sidebar.header("4. 电压偏置 (Bias)")
v_tg = st.sidebar.slider("顶栅 Vtg (V)", -10.0, 20.0, 10.0)
v_bg = st.sidebar.slider("底栅 Vbg (V)", -10.0, 20.0, 10.0)
v_ds = st.sidebar.slider("漏极 Vds (V)", 0.0, 20.0, 5.0)

st.sidebar.header("5. IdVg扫描参数 (IdVg Sweep)")
col_idvg1, col_idvg2 = st.sidebar.columns(2)
with col_idvg1:
    vd_lin = st.number_input("Vd线性区 (V)", value=0.1, min_value=0.0, max_value=10.0, step=0.1)
    vd_sat = st.number_input("Vd饱和区 (V)", value=5.1, min_value=0.0, max_value=20.0, step=0.1)
with col_idvg2:
    vg_start = st.number_input("Vg起始 (V)", value=-10.0, step=1.0)
    vg_stop = st.number_input("Vg终止 (V)", value=20.0, step=1.0)
vg_step = st.sidebar.number_input("Vg步长 (V)", value=0.2, min_value=0.01, max_value=1.0, step=0.05)

if st.sidebar.button("开始仿真 (RUN)", type="primary"):
    with st.spinner(f"正在计算... (IGZO层物理网格点: {ny_igzo})"):
        solver = TFTPoissonSolver(
            length=L_um,
            width=W_um,
            t_buf_sin=t_sin_nm/1000.0, eps_buf_sin=eps_sin,
            t_buf_sio=t_buf_sio_nm/1000.0, eps_buf_sio=eps_buf_sio,
            t_igzo=t_igzo_nm/1000.0, eps_igzo=eps_igzo, nd_igzo=nd_igzo,
            t_gi=t_gi_nm/1000.0, eps_gi=eps_gi,
            dit_top=dit_top, dit_bottom=dit_bottom, e_trap=e_trap,
            L_source=L_source_um, Rs_sheet=Rs_sheet,
            L_drain=L_drain_um, Rd_sheet=Rd_sheet,
            structure_type=struct_type,
            nx=nx, 
            ny=ny_igzo 
        )
        
        start = time.time()
        phi, n_conc, E, vd_eff, ids = solver.solve(v_top_gate_bias=v_tg, v_ds=v_ds, v_bot_gate_bias=v_bg)
        elapsed = time.time() - start
        
        st.success(f"计算完成，耗时 {elapsed:.3f}秒 | 有效Vd = {vd_eff:.4f}V | Ids = {ids:.2e}A")
        
        vg_ref, ids_ref = solver.load_reference_idvg()
        
        if vg_ref is not None and ids_ref is not None:
            params_ref = {
                'W': 3.0, 'L': 4.0,
                't_igzo': 25.0,
                'nd': 1e17,
                'dit_top': 3e10,
                'dit_bottom': 5e10
            }
            
            params_cur = {
                'W': W_um, 'L': L_um,
                't_igzo': t_igzo_nm,
                'nd': nd_igzo,
                'dit_top': dit_top,
                'dit_bottom': dit_bottom
            }
            
            vg_lin, ids_lin = solver.scale_idvg_curve(vg_ref, ids_ref, params_cur, params_ref)
            vg_sat, ids_sat = solver.scale_idvg_curve(vg_ref, ids_ref, params_cur, params_ref)
            
            ids_lin = ids_lin * (vd_lin / 5.1)
            ids_sat = ids_sat * (vd_sat / 5.1)
            
            C_gi = solver.eps0 * eps_gi / (t_gi_nm * 1e-7)
            
            vth_lin = solver.calculate_vth_simple(vg_lin, ids_lin, threshold=1e-9)
            vth_sat = solver.calculate_vth_simple(vg_sat, ids_sat, threshold=1e-9)
            ss_lin = solver.calculate_ss_simple(vg_lin, ids_lin)
            ss_sat = solver.calculate_ss_simple(vg_sat, ids_sat)
            mob_lin = solver.calculate_mobility_simple(vg_lin, ids_lin, vd_lin, W_um*1e-4, L_um*1e-4, C_gi)
            mob_sat = solver.calculate_mobility_simple(vg_sat, ids_sat, vd_sat, W_um*1e-4, L_um*1e-4, C_gi)
            ion_lin = solver.extract_ion_simple(vg_lin, ids_lin, vg_target=10.0)
            ion_sat = solver.extract_ion_simple(vg_sat, ids_sat, vg_target=10.0)
        else:
            st.warning("参考IdVg曲线未找到，跳过IdVg绘图")
            vg_lin, ids_lin, vg_sat, ids_sat = None, None, None, None
            vth_lin, vth_sat, ss_lin, ss_sat = np.nan, np.nan, np.nan, np.nan
            mob_lin, mob_sat, ion_lin, ion_sat = np.nan, np.nan, np.nan, np.nan
        
        # --- 物理级重采样渲染 ---
        y_phys_cm = solver.y
        x_phys_cm = solver.x
        
        # 1. 提取 IGZO 区域
        # solver输入单位是μm，内部转为cm (1μm = 1e-4 cm)
        total_buf_um = (t_sin_nm + t_buf_sio_nm) / 1000.0
        igzo_um = t_igzo_nm / 1000.0
        
        tol = 1e-10
        y_igzo_start = total_buf_um * 1e-4
        y_igzo_end = (total_buf_um + igzo_um) * 1e-4
        
        idx_igzo = np.where((y_phys_cm >= y_igzo_start - tol) & 
                            (y_phys_cm <= y_igzo_end + tol))[0]
        
        y_igzo_subset = y_phys_cm[idx_igzo]
        phi_igzo_subset = phi[idx_igzo, :]
        
        target_render_ny = 800
        y_hd_cm = np.linspace(y_igzo_subset.min(), y_igzo_subset.max(), target_render_ny)
        
        # 3. 插值电势 (Potential) - 使用 Quadratic 保证平滑
        f_phi = interp1d(y_igzo_subset, phi_igzo_subset, axis=0, kind='quadratic')
        phi_hd = f_phi(y_hd_cm)
        
        # 4. 重算载流子 (Physics)
        v_ch_hd = np.zeros_like(phi_hd)
        for i in range(len(x_phys_cm)):
            v_ch_hd[:, i] = vd_eff * (i / (len(x_phys_cm)-1))
            
        n_hd = solver.calculate_n_from_phi(phi_hd, v_ch_hd)
        n_hd = np.clip(n_hd, 1e10, 1e22)
        n_log_hd = np.log10(n_hd)
        
        x_plot = x_phys_cm * 1e4
        y_plot = y_hd_cm * 1e7
        
        z_min, z_max = np.nanmin(n_log_hd), np.nanmax(n_log_hd)
        
        if not np.isfinite(z_min):
            z_min = 10.0
        if not np.isfinite(z_max):
            z_max = 22.0
        
        z_min = max(z_min, 10.0)
        z_max = min(z_max, 22.0)
        
        tick_vals = np.arange(np.floor(z_min), np.ceil(z_max)+0.1, 0.5)
        tick_text = [f"1e{v:.1f}" for v in tick_vals]

        tab1, tab2, tab3, tab4 = st.tabs(["载流子浓度 (Carrier)", "垂直切面 (Cut)", "电场分布 (E-Field)", "IdVg特性曲线 (IdVg)"])
        
        with tab1:
            st.subheader("电子浓度 (cm^-3)")
            fig = go.Figure(data=go.Heatmap(
                x=x_plot, y=y_plot, z=n_log_hd,
                colorscale='Jet',
                zsmooth='best', 
                colorbar=dict(title='Concentration', tickvals=tick_vals, ticktext=tick_text),
                hovertemplate='Y: %{y:.2f} nm<br>n: 1e%{z:.2f}<extra></extra>'
            ))
            fig.update_layout(height=500, yaxis_title="IGZO Thickness (nm from Substrate)")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.subheader("垂直切面 (Vertical Profile)")
            mid = n_hd.shape[1] // 2
            fig2 = go.Figure(go.Scatter(x=y_plot, y=n_hd[:, mid], mode='lines', name='n', line=dict(width=3)))
            fig2.update_layout(yaxis_type="log", yaxis_title="n (cm^-3)", xaxis_title="Thickness (nm)")
            st.plotly_chart(fig2, use_container_width=True)
            
        with tab3:
            st.subheader("电场分布 (V/cm)")
            E_subset = E[idx_igzo, :]
            f_E = interp1d(y_igzo_subset, E_subset, axis=0, kind='quadratic')
            E_hd = f_E(y_hd_cm)
            
            fig3 = go.Figure(go.Heatmap(
                x=x_plot, y=y_plot, z=E_hd, colorscale='Hot', zsmooth='best',
                colorbar=dict(title='Field (V/cm)', tickformat='.1e')
            ))
            st.plotly_chart(fig3, use_container_width=True)
        
        with tab4:
            st.subheader("IdVg特性曲线 (Transfer Characteristics)")
            
            if vg_lin is not None and ids_lin is not None:
                col_param1, col_param2 = st.columns(2)
                
                with col_param1:
                    st.markdown("### 线性区参数 (Vd={:.2f}V)".format(vd_lin))
                    st.metric("阈值电压 Vth", f"{vth_lin:.3f} V" if not np.isnan(vth_lin) else "N/A")
                    st.metric("亚阈值摆幅 SS", f"{ss_lin:.3f} V/dec" if not np.isnan(ss_lin) else "N/A")
                    st.metric("场效应迁移率 μ", f"{mob_lin:.2f} cm²/V·s" if not np.isnan(mob_lin) else "N/A")
                    st.metric("Ion @ Vg=10V", f"{ion_lin:.2e} A" if not np.isnan(ion_lin) else "N/A")
                
                with col_param2:
                    st.markdown("### 饱和区参数 (Vd={:.2f}V)".format(vd_sat))
                    st.metric("阈值电压 Vth", f"{vth_sat:.3f} V" if not np.isnan(vth_sat) else "N/A")
                    st.metric("亚阈值摆幅 SS", f"{ss_sat:.3f} V/dec" if not np.isnan(ss_sat) else "N/A")
                    st.metric("场效应迁移率 μ", f"{mob_sat:.2f} cm²/V·s" if not np.isnan(mob_sat) else "N/A")
                    st.metric("Ion @ Vg=10V", f"{ion_sat:.2e} A" if not np.isnan(ion_sat) else "N/A")
                
                fig_idvg = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("线性坐标 (Linear Scale)", "对数坐标 (Log Scale)"),
                    horizontal_spacing=0.12
                )
                
                fig_idvg.add_trace(
                    go.Scatter(x=vg_lin, y=ids_lin, mode='lines+markers', 
                              name=f'Vd={vd_lin}V (Lin)', line=dict(color='blue', width=2)),
                    row=1, col=1
                )
                fig_idvg.add_trace(
                    go.Scatter(x=vg_sat, y=ids_sat, mode='lines+markers', 
                              name=f'Vd={vd_sat}V (Sat)', line=dict(color='red', width=2)),
                    row=1, col=1
                )
                
                ids_lin_safe = np.where(ids_lin > 0, ids_lin, 1e-15)
                ids_sat_safe = np.where(ids_sat > 0, ids_sat, 1e-15)
                
                fig_idvg.add_trace(
                    go.Scatter(x=vg_lin, y=ids_lin_safe, mode='lines+markers', 
                              name=f'Vd={vd_lin}V (Lin)', line=dict(color='blue', width=2),
                              showlegend=False),
                    row=1, col=2
                )
                fig_idvg.add_trace(
                    go.Scatter(x=vg_sat, y=ids_sat_safe, mode='lines+markers', 
                              name=f'Vd={vd_sat}V (Sat)', line=dict(color='red', width=2),
                              showlegend=False),
                    row=1, col=2
                )
                
                fig_idvg.update_xaxes(title_text="Vg (V)", row=1, col=1)
                fig_idvg.update_xaxes(title_text="Vg (V)", row=1, col=2)
                fig_idvg.update_yaxes(title_text="Ids (A)", row=1, col=1)
                fig_idvg.update_yaxes(title_text="Ids (A)", type="log", row=1, col=2)
                
                fig_idvg.update_layout(height=500, showlegend=True, legend=dict(x=0.02, y=0.98))
                
                st.plotly_chart(fig_idvg, use_container_width=True)
            else:
                st.warning("参考IdVg曲线数据未加载，无法显示IdVg曲线")