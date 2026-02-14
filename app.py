import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
from solver import TFTPoissonSolver
import time

st.set_page_config(layout="wide", page_title="Oxide TFT HD")

st.title("Oxide TFT Simulation (Stacked Buffer Model)")

if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = None

# --- Sidebar ---
st.sidebar.header("1. 几何尺寸 (Geometry)")
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

# 底栅设置：可以选择固定电压或 Floating
bg_col1, bg_col2 = st.sidebar.columns([3, 1])
with bg_col2:
    bg_floating = st.checkbox("Floating", value=False, key="bg_floating")

with bg_col1:
    if bg_floating:
        v_bg = st.slider("底栅 Vbg (V)", -10.0, 20.0, 10.0, disabled=True)
        st.caption("底栅 Floating 模式")
    else:
        v_bg = st.slider("底栅 Vbg (V)", -10.0, 20.0, 10.0)

v_ds = st.sidebar.slider("漏极 Vds (V)", 0.0, 20.0, 5.0)

def run_simulation():
    vd_sat = 5.1
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
            structure_type='Double Gate',
            nx=nx,
            ny=ny_igzo
        )

        start = time.time()
        phi, _, E, vd_eff, ids = solver.solve(v_top_gate_bias=v_tg, v_ds=v_ds, v_bot_gate_bias=v_bg)
        elapsed = time.time() - start

        vg_ref, ids_ref = solver.load_reference_idvg()

        if vg_ref is not None and ids_ref is not None:
            params_ref = {
                'W': 3.0, 'L': 4.0,
                't_igzo': 25.0,
                'nd': 1e17,
                'dit_top': 3e10,
                'dit_bottom': 5e10,
                'eps_gi': 3.9,
                't_gi': 140,
                'eps_buf_sio': 3.9,
                't_buf_sio': 300,
                'eps_sin': 7.0,
                't_sin': 100,
                'structure_type': 'Double Gate',
                'e_trap': 0.3,
                'L_source': 3.0,
                'Rs_sheet': 3700.0,
                'L_drain': 3.0,
                'Rd_sheet': 3700.0
            }

            params_cur = {
                'W': W_um, 'L': L_um,
                't_igzo': t_igzo_nm,
                'nd': nd_igzo,
                'dit_top': dit_top,
                'dit_bottom': dit_bottom,
                'eps_gi': eps_gi,
                't_gi': t_gi_nm,
                'eps_buf_sio': eps_buf_sio,
                't_buf_sio': t_buf_sio_nm,
                'eps_sin': eps_sin,
                't_sin': t_sin_nm,
                'structure_type': 'Double Gate',
                'e_trap': e_trap,
                'L_source': L_source_um,
                'Rs_sheet': Rs_sheet,
                'L_drain': L_drain_um,
                'Rd_sheet': Rd_sheet
            }

            vg_sat, ids_sat = solver.scale_idvg_curve(vg_ref, ids_ref, params_cur, params_ref, v_d=vd_sat)

            C_gi = solver.eps0 * eps_gi / (t_gi_nm * 1e-7)

            vth_sat = solver.calculate_vth_simple(vg_sat, ids_sat, threshold=1e-9) if vg_sat is not None and ids_sat is not None else np.nan
            ss_sat = solver.calculate_ss_simple(vg_sat, ids_sat) if vg_sat is not None and ids_sat is not None else np.nan
            mob_sat = solver.calculate_mobility_simple(vg_sat, ids_sat, vd_sat, W_um*1e-4, L_um*1e-4, C_gi) if vg_sat is not None and ids_sat is not None else np.nan
            ion_sat = solver.extract_ion_simple(vg_sat, ids_sat, vg_target=10.0) if vg_sat is not None and ids_sat is not None else np.nan

            params_tg_sweep = params_cur.copy()
            params_tg_sweep['structure_type'] = 'Single Gate (Top)'
            if bg_floating:
                params_tg_sweep['v_bg_floating'] = True
                params_tg_sweep['v_bg_fixed'] = None
            else:
                params_tg_sweep['v_bg_fixed'] = v_bg

            vg_tg_sweep, ids_tg_sweep = solver.scale_idvg_curve(vg_ref, ids_ref, params_tg_sweep, params_ref, v_d=vd_sat)

            vth_tg = solver.calculate_vth_simple(vg_tg_sweep, ids_tg_sweep, threshold=1e-9) if vg_tg_sweep is not None and ids_tg_sweep is not None else np.nan

            # 计算单栅扫描的理论SS（基于物理模型）
            ss_factor_tg = solver.calculate_ss_factor(params_tg_sweep, params_ref)
            ss_tg = ss_sat * ss_factor_tg if not np.isnan(ss_sat) else np.nan

            # 计算单栅扫描的理论迁移率（基于物理模型）
            mob_factor_tg = solver.calculate_mob_factor(params_tg_sweep, params_ref)
            mob_tg = mob_sat * mob_factor_tg if not np.isnan(mob_sat) else np.nan

            # 获取SS计算调试信息
            ss_debug = params_tg_sweep.get('_debug_ss', {})

            # 存储调试信息
            st.session_state.debug_info = {
                'ss_sat': ss_sat,
                'ss_factor_tg': ss_factor_tg,
                'ss_tg': ss_tg,
                'mob_sat': mob_sat,
                'mob_factor_tg': mob_factor_tg,
                'mob_tg': mob_tg,
                'params_tg_structure': params_tg_sweep.get('structure_type'),
                'v_bg_fixed': params_tg_sweep.get('v_bg_fixed'),
                'v_bg_floating': params_tg_sweep.get('v_bg_floating'),
                'W': params_tg_sweep.get('W'),
                'L': params_tg_sweep.get('L'),
                'ss_debug_structure': ss_debug.get('structure_cur'),
                'ss_debug_body_factor': ss_debug.get('ss_body_factor'),
                'ss_debug_Cit_ref': ss_debug.get('Cit_ref'),
                'ss_debug_Cit_cur': ss_debug.get('Cit_cur'),
                'ss_debug_ss_factor': ss_debug.get('ss_factor')
            }

            ion_tg = solver.extract_ion_simple(vg_tg_sweep, ids_tg_sweep, vg_target=10.0) if vg_tg_sweep is not None and ids_tg_sweep is not None else np.nan
        else:
            vg_sat, ids_sat = None, None
            vth_sat, ss_sat, mob_sat, ion_sat = np.nan, np.nan, np.nan, np.nan
            vg_tg_sweep, ids_tg_sweep = None, None
            vth_tg, ss_tg, mob_tg, ion_tg = np.nan, np.nan, np.nan, np.nan

        y_phys_cm = solver.y
        x_phys_cm = solver.x

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

        f_phi = interp1d(y_igzo_subset, phi_igzo_subset, axis=0, kind='quadratic')
        phi_hd = f_phi(y_hd_cm)

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

        E_subset = E[idx_igzo, :]
        f_E = interp1d(y_igzo_subset, E_subset, axis=0, kind='quadratic')
        E_hd = f_E(y_hd_cm)

        st.session_state.simulation_results = {
            'elapsed': elapsed,
            'vd_eff': vd_eff,
            'ids': ids,
            'vg_sat': vg_sat,
            'ids_sat': ids_sat,
            'vth_sat': vth_sat,
            'ss_sat': ss_sat,
            'mob_sat': mob_sat,
            'ion_sat': ion_sat,
            'vg_tg_sweep': vg_tg_sweep,
            'ids_tg_sweep': ids_tg_sweep,
            'vth_tg': vth_tg,
            'ss_tg': ss_tg,
            'mob_tg': mob_tg,
            'ion_tg': ion_tg,
            'v_bg': v_bg,
            'bg_floating': bg_floating,
            'x_plot': x_plot,
            'y_plot': y_plot,
            'n_log_hd': n_log_hd,
            'n_hd': n_hd,
            'E_hd': E_hd,
            'tick_vals': tick_vals,
            'tick_text': tick_text,
            'vd_sat': vd_sat
        }

if st.sidebar.button("开始仿真 (RUN)", type="primary"):
    run_simulation()

if st.session_state.simulation_results is not None:
    res = st.session_state.simulation_results

    st.success(f"计算完成，耗时 {res['elapsed']:.3f}秒 | 有效Vd = {res['vd_eff']:.4f}V | Ids = {res['ids']:.2e}A")

    # 显示调试信息
    if st.session_state.debug_info:
        dbg = st.session_state.debug_info
        with st.expander("调试信息 (点击展开)"):
            st.write(f"双栅 SS: {dbg['ss_sat']:.4f} V/dec")
            st.write(f"单栅 SS 因子: {dbg['ss_factor_tg']:.4f}")
            st.write(f"单栅 SS: {dbg['ss_tg']:.4f} V/dec")
            st.write(f"双栅 mob: {dbg['mob_sat']:.2f} cm²/V·s")
            st.write(f"单栅 mob 因子: {dbg['mob_factor_tg']:.4f}")
            st.write(f"单栅 mob: {dbg['mob_tg']:.2f} cm²/V·s")
            st.write(f"结构类型: {dbg['params_tg_structure']}")
            st.write(f"v_bg_fixed: {dbg['v_bg_fixed']}")
            st.write(f"v_bg_floating: {dbg['v_bg_floating']}")
            st.write(f"SS Debug - 结构: {dbg.get('ss_debug_structure')}")
            st.write(f"SS Debug - body_factor: {dbg.get('ss_debug_body_factor')}")
            st.write(f"SS Debug - Cit_ref: {dbg.get('ss_debug_Cit_ref')}")
            st.write(f"SS Debug - Cit_cur: {dbg.get('ss_debug_Cit_cur')}")
            st.write(f"SS Debug - ss_factor: {dbg.get('ss_debug_ss_factor')}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["载流子浓度 (Carrier)", "垂直切面 (Cut)", "电场分布 (E-Field)", "IdVg双栅扫描 (DG)", "IdVg顶栅扫描 (TG Sweep)"])

    with tab1:
        st.subheader("电子浓度 (cm^-3)")
        fig = go.Figure(data=go.Heatmap(
            x=res['x_plot'], y=res['y_plot'], z=res['n_log_hd'],
            colorscale='Jet',
            zsmooth='best',
            colorbar=dict(title='Concentration', tickvals=res['tick_vals'], ticktext=res['tick_text']),
            hovertemplate='Y: %{y:.2f} nm<br>n: 1e%{z:.2f}<extra></extra>'
        ))
        fig.update_layout(height=500, yaxis_title="IGZO Thickness (nm from Substrate)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("垂直切面 (Vertical Profile)")
        mid = res['n_hd'].shape[1] // 2
        fig2 = go.Figure(go.Scatter(x=res['y_plot'], y=res['n_hd'][:, mid], mode='lines', name='n', line=dict(width=3)))
        fig2.update_layout(yaxis_type="log", yaxis_title="n (cm^-3)", xaxis_title="Thickness (nm)")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("电场分布 (V/cm)")
        fig3 = go.Figure(go.Heatmap(
            x=res['x_plot'], y=res['y_plot'], z=res['E_hd'], colorscale='Hot', zsmooth='best',
            colorbar=dict(title='Field (V/cm)', tickformat='.1e')
        ))
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.subheader("双栅IdVg扫描 (Double Gate Sweep)")
        st.caption("顶栅和底栅相连同时扫描")

        if res['vg_sat'] is not None and res['ids_sat'] is not None:
            st.markdown("### 器件参数 (Vd={:.2f}V, Vg从-15V到+15V)".format(res['vd_sat']))
            col_param1, col_param2, col_param3, col_param4 = st.columns(4)

            with col_param1:
                st.metric("阈值电压 Vth", f"{res['vth_sat']:.3f} V" if not np.isnan(res['vth_sat']) else "N/A")
            with col_param2:
                st.metric("亚阈值摆幅 SS", f"{res['ss_sat']:.3f} V/dec" if not np.isnan(res['ss_sat']) else "N/A")
            with col_param3:
                st.metric("场效应迁移率 μ", f"{res['mob_sat']:.2f} cm²/V·s" if not np.isnan(res['mob_sat']) else "N/A")
            with col_param4:
                st.metric("Ion @ Vg=10V", f"{res['ion_sat']:.2e} A" if not np.isnan(res['ion_sat']) else "N/A")

            fig_idvg = make_subplots(
                rows=1, cols=2,
                subplot_titles=("线性坐标 (Linear Scale)", "对数坐标 (Log Scale)"),
                horizontal_spacing=0.12
            )

            fig_idvg.add_trace(
                go.Scatter(x=res['vg_sat'], y=res['ids_sat'], mode='lines+markers',
                          name=f'Vd={res["vd_sat"]}V', line=dict(color='red', width=2)),
                row=1, col=1
            )

            ids_sat_safe = np.where(res['ids_sat'] > 0, res['ids_sat'], 1e-15) if res['ids_sat'] is not None else None

            if ids_sat_safe is not None:
                fig_idvg.add_trace(
                    go.Scatter(x=res['vg_sat'], y=ids_sat_safe, mode='lines+markers',
                              name=f'Vd={res["vd_sat"]}V', line=dict(color='red', width=2),
                              showlegend=False),
                    row=1, col=2
                )

            fig_idvg.update_xaxes(title_text="Vg (V)", range=[-15, 15], row=1, col=1)
            fig_idvg.update_xaxes(title_text="Vg (V)", range=[-15, 15], row=1, col=2)
            fig_idvg.update_yaxes(title_text="Ids (A)", row=1, col=1)
            fig_idvg.update_yaxes(title_text="Ids (A)", type="log", row=1, col=2)

            fig_idvg.update_layout(height=500, showlegend=True, legend=dict(x=0.02, y=0.98))

            st.plotly_chart(fig_idvg, use_container_width=True)

            import pandas as pd
            import io

            vg_formatted = np.round(res['vg_sat'], 1)
            ids_formatted = np.where(np.abs(res['ids_sat']) < 1e-14, 0.0, res['ids_sat'])

            df_idvg = pd.DataFrame({
                'Vg (V)': vg_formatted,
                'Ids (A)': ids_formatted
            })

            csv_buffer = io.StringIO()
            df_idvg.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            st.download_button(
                label="📥 下载双栅IdVg数据 (CSV)",
                data=csv_data,
                file_name=f"idvg_DG_Vd{res['vd_sat']:.1f}V.csv",
                mime="text/csv",
                help="下载双栅扫描IdVg曲线数据"
            )
        else:
            st.warning("参考IdVg曲线数据未加载，无法显示IdVg曲线")

    with tab5:
        st.subheader("顶栅IdVg扫描 (Top Gate Sweep)")

        if res['bg_floating']:
            st.caption("底栅 Floating，仅顶栅扫描")
            vbg_display = "Floating"
            vbg_filename = "Floating"
        else:
            st.caption(f"底栅固定电压 Vbg = {res['v_bg']:.1f}V，仅顶栅扫描")
            vbg_display = f"{res['v_bg']:.1f}V"
            vbg_filename = f"{res['v_bg']:.1f}V"

        if res['vg_tg_sweep'] is not None and res['ids_tg_sweep'] is not None:
            st.markdown(f"### 器件参数 (Vd={res['vd_sat']:.2f}V, Vtg从-15V到+15V, Vbg={vbg_display})")
            col_param1, col_param2, col_param3, col_param4 = st.columns(4)

            with col_param1:
                st.metric("阈值电压 Vth", f"{res['vth_tg']:.3f} V" if not np.isnan(res['vth_tg']) else "N/A")
            with col_param2:
                st.metric("亚阈值摆幅 SS", f"{res['ss_tg']:.3f} V/dec" if not np.isnan(res['ss_tg']) else "N/A")
            with col_param3:
                st.metric("场效应迁移率 μ", f"{res['mob_tg']:.2f} cm²/V·s" if not np.isnan(res['mob_tg']) else "N/A")
            with col_param4:
                st.metric("Ion @ Vtg=10V", f"{res['ion_tg']:.2e} A" if not np.isnan(res['ion_tg']) else "N/A")

            fig_idvg_tg = make_subplots(
                rows=1, cols=2,
                subplot_titles=("线性坐标 (Linear Scale)", "对数坐标 (Log Scale)"),
                horizontal_spacing=0.12
            )

            fig_idvg_tg.add_trace(
                go.Scatter(x=res['vg_tg_sweep'], y=res['ids_tg_sweep'], mode='lines+markers',
                          name=f'Vd={res["vd_sat"]}V, Vbg={vbg_display}', line=dict(color='blue', width=2)),
                row=1, col=1
            )

            ids_tg_safe = np.where(res['ids_tg_sweep'] > 0, res['ids_tg_sweep'], 1e-15) if res['ids_tg_sweep'] is not None else None

            if ids_tg_safe is not None:
                fig_idvg_tg.add_trace(
                    go.Scatter(x=res['vg_tg_sweep'], y=ids_tg_safe, mode='lines+markers',
                              name=f'Vd={res["vd_sat"]}V, Vbg={vbg_display}', line=dict(color='blue', width=2),
                              showlegend=False),
                    row=1, col=2
                )

            fig_idvg_tg.update_xaxes(title_text="Vtg (V)", range=[-15, 15], row=1, col=1)
            fig_idvg_tg.update_xaxes(title_text="Vtg (V)", range=[-15, 15], row=1, col=2)
            fig_idvg_tg.update_yaxes(title_text="Ids (A)", row=1, col=1)
            fig_idvg_tg.update_yaxes(title_text="Ids (A)", type="log", row=1, col=2)

            fig_idvg_tg.update_layout(height=500, showlegend=True, legend=dict(x=0.02, y=0.98))

            st.plotly_chart(fig_idvg_tg, use_container_width=True)

            import pandas as pd
            import io

            vg_tg_formatted = np.round(res['vg_tg_sweep'], 1)
            ids_tg_formatted = np.where(np.abs(res['ids_tg_sweep']) < 1e-14, 0.0, res['ids_tg_sweep'])

            df_idvg_tg = pd.DataFrame({
                'Vtg (V)': vg_tg_formatted,
                'Ids (A)': ids_tg_formatted
            })

            csv_buffer_tg = io.StringIO()
            df_idvg_tg.to_csv(csv_buffer_tg, index=False)
            csv_data_tg = csv_buffer_tg.getvalue()

            st.download_button(
                label="📥 下载顶栅IdVg数据 (CSV)",
                data=csv_data_tg,
                file_name=f"idvg_TG_Vd{res['vd_sat']:.1f}V_Vbg{vbg_filename}.csv",
                mime="text/csv",
                help="下载顶栅扫描IdVg曲线数据"
            )
        else:
            st.warning("参考IdVg曲线数据未加载，无法显示顶栅扫描IdVg曲线")