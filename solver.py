import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

class TFTPoissonSolver:
    def __init__(self, length=2.0, width=10.0,
                 t_buf_sin=0.1, eps_buf_sin=7.0,
                 t_buf_sio=0.1, eps_buf_sio=3.9,
                 t_igzo=0.05, eps_igzo=10.0, nd_igzo=1e16,
                 t_gi=0.1, eps_gi=3.9,
                 dit_top=0.0, dit_bottom=0.0, e_trap=0.3,
                 L_source=3.0, Rs_sheet=3500.0,
                 L_drain=3.0, Rd_sheet=3500.0,
                 structure_type='Double Gate',
                 nx=50, ny=400): 
        
        self.q = 1.602e-19
        self.Vt = 0.0259
        self.eps0 = 8.854e-14 
        self.Nc = 5.0e18      
        self.N_off = nd_igzo  
        
        self.N_it_top = dit_top        
        self.N_it_bottom = dit_bottom  
        self.E_trap = e_trap
        
        self.L_cm = length * 1e-4
        self.W_cm = width * 1e-4
        
        self.t_sin_cm = t_buf_sin * 1e-4
        self.t_buf_sio_cm = t_buf_sio * 1e-4
        self.t_igzo_cm = t_igzo * 1e-4
        self.t_gi_cm = t_gi * 1e-4
        
        self.eps_sin = eps_buf_sin
        self.eps_buf_sio = eps_buf_sio
        self.eps_igzo = eps_igzo
        self.eps_gi = eps_gi
        
        self.L_source_cm = L_source * 1e-4
        self.Rs_sheet = Rs_sheet
        self.L_drain_cm = L_drain * 1e-4
        self.Rd_sheet = Rd_sheet
        
        self.R_source = Rs_sheet * self.L_source_cm / self.W_cm if self.W_cm > 0 else 0
        self.R_drain = Rd_sheet * self.L_drain_cm / self.W_cm if self.W_cm > 0 else 0
        
        self.structure_type = structure_type
        self.nx = int(nx)
        self.user_ny = int(ny) 
        
        self._init_mesh_final()

    def _init_mesh_final(self):
        y_if1 = self.t_sin_cm                      
        y_if2 = y_if1 + self.t_buf_sio_cm          
        y_if3 = y_if2 + self.t_igzo_cm             
        y_top = y_if3 + self.t_gi_cm               
        
        self.y_if2 = y_if2
        self.y_if3 = y_if3
        
        n_igzo = self.user_ny
        y_igzo_mesh = np.linspace(y_if2, y_if3, n_igzo)
        
        n_sin = 30
        n_buf_sio = 30
        n_gi = 40
        
        y_sin_mesh = np.linspace(0, y_if1, n_sin)
        y_buf_sio_mesh = np.linspace(y_if1, y_if2, n_buf_sio)
        y_gi_mesh = np.linspace(y_if3, y_top, n_gi)
        
        y_all = np.concatenate([y_sin_mesh, y_buf_sio_mesh, y_igzo_mesh, y_gi_mesh])
        self.y = np.unique(y_all)
        self.ny = len(self.y)
        
        self.idx_if2 = np.argmin(np.abs(self.y - y_if2))
        self.idx_if3 = np.argmin(np.abs(self.y - y_if3))
        
        self.x = np.linspace(0, self.L_cm, self.nx)
        self.dx = self.x[1] - self.x[0]
        
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        tol = 1e-13
        self.mask_sin = (self.Y < y_if1 - tol)
        self.mask_buf_sio = (self.Y >= y_if1 - tol) & (self.Y < y_if2 - tol)
        self.mask_igzo = (self.Y >= y_if2 - tol) & (self.Y <= y_if3 + tol)
        self.mask_gi = (self.Y > y_if3 + tol)
        
        # 构建介电常数图
        self.epsilon_map = np.zeros_like(self.X)
        self.epsilon_map[self.mask_sin] = self.eps_sin
        self.epsilon_map[self.mask_buf_sio] = self.eps_buf_sio
        self.epsilon_map[self.mask_igzo] = self.eps_igzo
        self.epsilon_map[self.mask_gi] = self.eps_gi

    def calculate_n_from_phi(self, phi_val, v_ch_val):
        u = (phi_val - v_ch_val) / self.Vt
        val_log = np.logaddexp(0, u)
        n_val = self.N_off + self.Nc * val_log
        return n_val
    
    def calculate_interface_trap_charge_density(self, phi_surface):
        """
        Calculate interface trap charge density (C/cm²) at given surface potential.
        Uses Fermi-Dirac statistics for trap occupancy.
        
        Returns:
            Q_it: Interface charge density in C/cm²
        """
        u_trap = np.clip((phi_surface - self.E_trap) / self.Vt, -50, 50)
        f_trap = 1.0 / (1.0 + np.exp(-u_trap))
        f_neutral = 1.0 / (1.0 + np.exp(self.E_trap / self.Vt))
        
        Q_it = -self.q * (f_trap - f_neutral)
        return Q_it
    
    def calculate_current(self):
        mu = 10.0
        tol = 1e-13
        
        y_if2 = self.t_sin_cm + self.t_buf_sio_cm
        y_if3 = y_if2 + self.t_igzo_cm
        
        mask_igzo_local = (self.Y >= y_if2 - tol) & (self.Y <= y_if3 + tol)
        
        n_igzo = self.n_conc * mask_igzo_local
        Ex_igzo = self.Ex * mask_igzo_local
        
        Jx = self.q * mu * n_igzo * (-Ex_igzo)
        
        mid_x = self.nx // 2
        I_total = np.trapezoid(Jx[:, mid_x], self.y) * self.W_cm
        return I_total

    def solve(self, v_top_gate_bias, v_ds, v_bot_gate_bias=0.0):
        delta_vg_top, delta_vg_bottom = self.calculate_equivalent_vg_shift()
        
        v_tg = v_top_gate_bias - delta_vg_top
        v_bg = 0.0 if self.structure_type == 'Source-Gated Bottom' else (v_bot_gate_bias - delta_vg_bottom)
        
        v_d_eff = v_ds
        I_ds = 0.0
        max_outer_iter = 10
        
        for outer_it in range(max_outer_iter):
            v_ch = np.zeros((self.ny, self.nx))
            for i in range(self.nx): 
                v_ch[:, i] = v_d_eff * (i/(self.nx-1))
                
            phi = np.zeros((self.ny, self.nx))
            for i in range(self.ny):
                r = self.y[i]/self.y[-1]
                phi[i, :] = v_bg*(1-r) + v_tg*r
                
            N = self.nx * self.ny
            cx = 1.0/(self.dx**2)
            eps_flat = self.epsilon_map.flatten() * self.eps0
            
            for it in range(30):
                n_conc = self.calculate_n_from_phi(phi, v_ch)
                u = (phi - v_ch)/self.Vt
                sig_u = 1.0/(1.0 + np.exp(-u))
                dn_dphi = (self.Nc/self.Vt)*sig_u
                
                rows, cols, data, rhs = [], [], [], np.zeros(N)
                
                for iy in range(self.ny):
                    if iy > 0: dy_dn = self.y[iy] - self.y[iy-1]
                    else: dy_dn = 1e-9
                    if iy < self.ny-1: dy_up = self.y[iy+1] - self.y[iy]
                    else: dy_up = 1e-9
                    
                    denom = dy_up + dy_dn
                    cy_up = 2.0/(dy_up*denom)
                    cy_dn = 2.0/(dy_dn*denom)
                    
                    for ix in range(self.nx):
                        k = iy*self.nx + ix
                        is_bc=False; val=0.0
                        
                        if iy==self.ny-1 and self.structure_type!='Single Gate (Bottom)': is_bc=True; val=v_tg
                        elif iy==0 and self.structure_type!='Single Gate (Top)': is_bc=True; val=v_bg
                        elif ix==0 and self.mask_igzo[iy,ix]: is_bc=True; val=0.0
                        elif ix==self.nx-1 and self.mask_igzo[iy,ix]: is_bc=True; val=v_d_eff
                        
                        if is_bc:
                            rows.append(k); cols.append(k); data.append(1.0); rhs[k]=-(phi[iy,ix]-val)
                            continue
                        
                        eps = eps_flat[k]; p_c = phi[iy,ix]; d_lap=0.0
                        
                        if ix>0: rows.append(k); cols.append(k-1); data.append(eps*cx); d_lap-=eps*cx
                        if ix<self.nx-1: rows.append(k); cols.append(k+1); data.append(eps*cx); d_lap-=eps*cx
                        if iy>0: rows.append(k); cols.append(k-self.nx); data.append(eps*cy_dn); d_lap-=eps*cy_dn
                        if iy<self.ny-1: rows.append(k); cols.append(k+self.nx); data.append(eps*cy_up); d_lap-=eps*cy_up
                        rows.append(k); cols.append(k); data.append(d_lap)
                        
                        p_l=phi[iy,ix-1] if ix>0 else p_c; p_r=phi[iy,ix+1] if ix<self.nx-1 else p_c
                        p_b=phi[iy-1,ix] if iy>0 else p_c; p_t=phi[iy+1,ix] if iy<self.ny-1 else p_c
                        res = eps*(p_l-p_c)*cx + eps*(p_r-p_c)*cx + eps*(p_b-p_c)*cy_dn + eps*(p_t-p_c)*cy_up
                        
                        rho=0; drho=0
                        if self.mask_igzo[iy,ix]:
                            rho = self.q*(self.N_off - n_conc[iy,ix])
                            drho = -self.q*dn_dphi[iy,ix]
                            rows.append(k); cols.append(k); data.append(drho)
                        rhs[k] = -(res + rho)
                
                J = sparse.coo_matrix((data,(rows,cols)), shape=(N,N)).tocsr()
                try: 
                    delta = spsolve(J, rhs)
                    delta = np.asarray(delta).flatten()
                except: 
                    break
                phi_flat = phi.flatten() + delta
                phi = phi_flat.reshape((self.ny,self.nx))
                if np.max(np.abs(delta)) < 1e-4: break
            
            self.phi = phi
            self.n_conc = self.calculate_n_from_phi(phi, v_ch)
            self._calc_field()
            
            I_ds = self.calculate_current()
            
            v_drop_source = I_ds * self.R_source
            v_drop_drain = I_ds * self.R_drain
            v_d_new = v_ds - v_drop_source - v_drop_drain
            
            if abs(v_d_new - v_d_eff) < 1e-4:
                v_d_eff = v_d_new
                break
            
            v_d_eff = 0.5 * v_d_eff + 0.5 * v_d_new
        
        return self.phi, self.n_conc, self.E_field, v_d_eff, I_ds

    def calculate_equivalent_vg_shift(self):
        delta_vg_top = 0.0
        delta_vg_bottom = 0.0
        
        if self.N_it_top > 0:
            Q_it_top = self.q * self.N_it_top * 0.5
            delta_vg_base = Q_it_top * self.t_gi_cm / (self.eps_gi * self.eps0)
            delta_vg_top = delta_vg_base * 10.0
        
        if self.N_it_bottom > 0:
            Q_it_bottom = self.q * self.N_it_bottom * 0.5
            delta_vg_base = Q_it_bottom * self.t_gi_cm / (self.eps_gi * self.eps0)
            delta_vg_bottom = delta_vg_base * 10.0
        
        return delta_vg_top, delta_vg_bottom
    
    def _calc_field(self):
        self.Ey, self.Ex = np.gradient(self.phi, self.y, self.x)
        self.Ey, self.Ex = -self.Ey, -self.Ex
        self.E_field = np.sqrt(self.Ex**2 + self.Ey**2)
    
    @staticmethod
    def load_reference_idvg(csv_path=None):
        """加载参考IdVg曲线数据"""
        try:
            if csv_path is None:
                # 使用相对路径，适用于云端部署
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                csv_path = os.path.join(current_dir, '双栅基准idvg.csv')
            data = np.genfromtxt(csv_path, delimiter=',', skip_header=2)
            vg_ref = data[:, 0]
            ids_ref = data[:, 1]
            return vg_ref, ids_ref
        except Exception as e:
            print(f"Error loading reference IdVg data: {e}")
            return None, None
    
    @staticmethod
    def extract_reference_params(vg_ref, ids_ref):
        """
        从参考曲线中提取关键参数
        返回: {'vth': 阈值电压, 'ss': 亚阈值摆幅, 'ion': 开态电流, 'mob': 迁移率}
        """
        params = {}
        
        # 提取Vth (使用恒定电流法，1nA)
        params['vth'] = TFTPoissonSolver.calculate_vth_simple(vg_ref, ids_ref, threshold=1e-9)
        
        # 提取SS
        params['ss'] = TFTPoissonSolver.calculate_ss_simple(vg_ref, ids_ref)
        
        # 提取Ion @ Vg=10V
        params['ion'] = TFTPoissonSolver.extract_ion_simple(vg_ref, ids_ref, vg_target=10.0)
        
        # 提取迁移率 (使用线性区近似)
        mask_valid = (vg_ref > params['vth']) & (vg_ref < params['vth'] + 5) & (ids_ref > 0)
        if np.sum(mask_valid) >= 2:
            vg_lin = vg_ref[mask_valid]
            ids_lin = ids_ref[mask_valid]
            # 简化的迁移率估计: μ ∝ dIds/dVg
            dIds_dVg = np.gradient(ids_lin, vg_lin)
            params['mob'] = np.mean(dIds_dVg[np.isfinite(dIds_dVg)]) * 1e-3  # 粗略估计
        else:
            params['mob'] = 10.0  # 默认值
            
        return params
    
    @staticmethod
    def calculate_vth_shift(params_current, params_reference):
        """
        计算阈值电压偏移量
        基于物理模型：界面陷阱、厚度、掺杂浓度、buf层等因素
        """
        q = 1.602e-19
        eps0 = 8.854e-14
        Vt = 0.0259
        
        # 参考参数 (双栅基准)
        t_igzo_ref = params_reference['t_igzo'] * 1e-7
        nd_ref = params_reference['nd']
        dit_top_ref = params_reference['dit_top']
        dit_bot_ref = params_reference['dit_bottom']
        eps_gi_ref = 3.9
        t_gi_ref = 140e-7
        eps_buf_sio_ref = 3.9
        t_buf_sio_ref = 300e-7
        eps_sin_ref = 7.0
        t_sin_ref = 100e-7
        structure_ref = 'Double Gate'
        e_trap_ref = 0.3
        
        # 当前参数
        t_igzo_cur = params_current['t_igzo'] * 1e-7
        nd_cur = params_current['nd']
        dit_top_cur = params_current['dit_top']
        dit_bot_cur = params_current['dit_bottom']
        eps_gi_cur = params_current.get('eps_gi', 3.9)
        t_gi_cur = params_current.get('t_gi', 140) * 1e-7
        eps_buf_sio_cur = params_current.get('eps_buf_sio', 3.9)
        t_buf_sio_cur = params_current.get('t_buf_sio', 300) * 1e-7
        eps_sin_cur = params_current.get('eps_sin', 7.0)
        t_sin_cur = params_current.get('t_sin', 100) * 1e-7
        structure_cur = params_current.get('structure_type', 'Double Gate')
        e_trap_cur = params_current.get('e_trap', 0.3)
        
        delta_vth = 0.0
        
        # 1. 界面陷阱引起的Vth偏移
        Cox_gi_ref = eps0 * eps_gi_ref / t_gi_ref
        Cox_gi_cur = eps0 * eps_gi_cur / t_gi_cur
        
        # Buf层电容 (串联)
        Cox_buf_ref = eps0 * eps_buf_sio_ref / t_buf_sio_ref
        Cox_sin_ref = eps0 * eps_sin_ref / t_sin_ref
        Cox_bottom_ref = 1.0 / (1.0/Cox_sin_ref + 1.0/Cox_buf_ref)
        
        Cox_buf_cur = eps0 * eps_buf_sio_cur / t_buf_sio_cur
        Cox_sin_cur = eps0 * eps_sin_cur / t_sin_cur
        Cox_bottom_cur = 1.0 / (1.0/Cox_sin_cur + 1.0/Cox_buf_cur)
        
        dit_total_ref = dit_top_ref + dit_bot_ref
        dit_total_cur = dit_top_cur + dit_bot_cur
        
        # 顶栅界面陷阱影响
        delta_vth_dit_top = q * (dit_top_cur - dit_top_ref) / Cox_gi_cur
        delta_vth += delta_vth_dit_top
        
        # 底栅界面陷阱影响 (通过buf层)
        delta_vth_dit_bot = q * (dit_bot_cur - dit_bot_ref) / Cox_bottom_cur
        delta_vth += delta_vth_dit_bot
        
        # 2. Buf膜厚变化对底栅控制的影响
        # 更厚的buf层 = 更小的Cox = 底栅控制减弱 = 需要更大Vg = Vth增加
        if t_buf_sio_cur != t_buf_sio_ref or t_sin_cur != t_sin_ref:
            eff_coupling_ref = Cox_bottom_ref / (Cox_bottom_ref + Cox_gi_ref)
            eff_coupling_cur = Cox_bottom_cur / (Cox_bottom_cur + Cox_gi_cur)
            # 耦合效率降低时Vth增加，因此取负号
            delta_vth += (eff_coupling_ref - eff_coupling_cur) * 2.0
        
        # 3. Buf介电常数变化影响
        if eps_buf_sio_cur != eps_buf_sio_ref or eps_sin_cur != eps_sin_ref:
            eff_coupling_ref = Cox_bottom_ref / (Cox_bottom_ref + Cox_gi_ref)
            eff_coupling_cur = Cox_bottom_cur / (Cox_bottom_cur + Cox_gi_cur)
            delta_vth += (eff_coupling_ref - eff_coupling_cur) * 1.5
        
        # 4. IGZO厚度变化引起的Vth偏移
        # 更厚的IGZO = 更多可动载流子 = 更容易导通 = Vth负移
        if t_igzo_cur != t_igzo_ref:
            delta_vth_thickness = -0.02 * (t_igzo_cur - t_igzo_ref) / 1e-7
            delta_vth += delta_vth_thickness
        
        # 5. 掺杂浓度引起的Vth偏移
        if nd_cur != nd_ref and nd_ref > 0:
            delta_vth_doping = Vt * np.log(nd_cur / nd_ref) * 0.5
            delta_vth += delta_vth_doping

        # 6. W/L 对 Vth 的影响（短沟道效应）
        # W/L 越小（短沟道），Vth 越低
        W_cur = params_current.get('W', 3.0)
        L_cur = params_current.get('L', 4.0)
        W_ref = params_reference.get('W', 3.0)
        L_ref = params_reference.get('L', 4.0)
        WL_ratio_cur = W_cur / L_cur if L_cur > 0 else 0.75
        WL_ratio_ref = W_ref / L_ref if L_ref > 0 else 0.75
        if abs(WL_ratio_cur - WL_ratio_ref) > 0.01:
            # W/L 减小 10% -> Vth 降低约 0.05V
            delta_vth_WL = -0.5 * (WL_ratio_cur - WL_ratio_ref) / WL_ratio_ref
            delta_vth += delta_vth_WL

        # 7. 结构类型影响 (单栅vs双栅)
        if structure_cur != structure_ref:
            if 'Single' in structure_cur:
                # 单栅时，控制效果减半，Vth约增加0.5-1V
                delta_vth += 0.5
        
        # 8. 陷阱能级位置影响
        # 更深的陷阱能级 = 更难释放载流子 = 更大的Vth
        if e_trap_cur != e_trap_ref:
            delta_vth += (e_trap_cur - e_trap_ref) * 1.0

        # 9. 底栅固定电压对顶栅扫描的影响 (衬底偏置效应/Body Effect)
        v_bg_fixed = params_current.get('v_bg_fixed', None)
        v_bg_floating = params_current.get('v_bg_floating', False)
        if v_bg_fixed is not None and structure_cur == 'Single Gate (Top)' and not v_bg_floating:
            coupling_efficiency = Cox_bottom_cur / (Cox_bottom_cur + Cox_gi_cur)
            gamma = 0.3
            delta_vth_body = -v_bg_fixed * coupling_efficiency * gamma
            delta_vth += delta_vth_body

        return delta_vth
    
    @staticmethod
    def calculate_ss_factor(params_current, params_reference):
        """
        计算亚阈值摆幅变化因子
        SS = ln(10) * kT/q * (1 + Cd/Cox)
        界面陷阱会增加Cd，从而增加SS
        衬底偏置效应也会影响SS：底栅负电压会增加耗尽层电容，从而增加SS
        """
        q = 1.602e-19
        eps0 = 8.854e-14
        Vt = 0.0259

        dit_top_ref = params_reference['dit_top']
        dit_bot_ref = params_reference['dit_bottom']
        dit_top_cur = params_current['dit_top']
        dit_bot_cur = params_current['dit_bottom']

        eps_gi = params_current.get('eps_gi', 3.9)
        t_gi = params_current.get('t_gi', 140) * 1e-7
        t_igzo_cur = params_current['t_igzo'] * 1e-7

        Cox = eps0 * eps_gi / t_gi

        # 界面陷阱等效电容 (每cm^2)
        Cit_ref = q * (dit_top_ref + dit_bot_ref)
        Cit_cur = q * (dit_top_cur + dit_bot_cur)

        structure_cur = params_current.get('structure_type', 'Double Gate')
        v_bg_fixed = params_current.get('v_bg_fixed', None)
        v_bg_floating = params_current.get('v_bg_floating', False)

        # 计算底栅耦合效率（buf越薄，耦合越强）
        eps_buf_sio_cur = params_current.get('eps_buf_sio', 3.9)
        t_buf_sio_cur = params_current.get('t_buf_sio', 300) * 1e-7
        eps_sin_cur = params_current.get('eps_sin', 7.0)
        t_sin_cur = params_current.get('t_sin', 100) * 1e-7

        Cox_buf_cur = eps0 * eps_buf_sio_cur / t_buf_sio_cur if t_buf_sio_cur > 0 else 1e-7
        Cox_sin_cur = eps0 * eps_sin_cur / t_sin_cur if t_sin_cur > 0 else 1e-7
        Cox_bottom_cur = 1.0 / (1.0 / Cox_sin_cur + 1.0 / Cox_buf_cur) if Cox_sin_cur > 0 and Cox_buf_cur > 0 else 1e-7

        # 参考耦合效率（buf厚度300nm）
        Cox_buf_ref = eps0 * 3.9 / (300 * 1e-7)
        Cox_sin_ref = eps0 * 7.0 / (100 * 1e-7)
        Cox_bottom_ref = 1.0 / (1.0 / Cox_sin_ref + 1.0 / Cox_buf_ref)

        # 耦合效率比值（当前/参考）
        coupling_ratio = Cox_bottom_cur / Cox_bottom_ref if Cox_bottom_ref > 0 else 1.0

        W_cur = params_current['W']
        L_cur = params_current['L']
        WL_ratio = W_cur / L_cur if L_cur > 0 else 0.5
        WL_ref = 0.5

        # IGZO厚度对SS的影响（IGZO越厚，栅控越弱，SS越大）
        t_igzo_cur = params_current.get('t_igzo', 25.0)
        t_igzo_ref = params_reference.get('t_igzo', 25.0)

        if structure_cur == 'Single Gate (Top)':
            ss_body_factor = 1.59

            if not v_bg_floating and v_bg_fixed is not None:
                base_effect = 0.10 * (-v_bg_fixed) / 10.0
                ss_body_factor += base_effect * coupling_ratio

            if abs(WL_ratio - WL_ref) > 0.001:
                WL_factor = 1.0 - 0.20 * np.log(WL_ratio / WL_ref)
                ss_body_factor *= WL_factor

            # 单栅模式下IGZO厚度影响更明显
            if t_igzo_cur != t_igzo_ref:
                igzo_factor = 1.0 + 0.05 * (t_igzo_cur - t_igzo_ref) / 10.0
                ss_body_factor *= igzo_factor
        else:
            ss_body_factor = 1.0

            # 双栅模式下IGZO厚度影响
            if t_igzo_cur != t_igzo_ref:
                igzo_factor = 1.0 + 0.03 * (t_igzo_cur - t_igzo_ref) / 10.0
                ss_body_factor *= igzo_factor

            if abs(WL_ratio - WL_ref) > 0.001:
                WL_factor = 1.0 - 0.15 * np.log(WL_ratio / WL_ref)
                ss_body_factor *= WL_factor

        # SS比例因子
        ss_ideal = 60e-3  # 理想SS = 60mV/dec @ 300K
        ss_ref = ss_ideal * (1 + Cit_ref / Cox)
        ss_cur = ss_ideal * (1 + Cit_cur / Cox) * ss_body_factor

        # 限制SS在合理范围内 (60-300 mV/dec)
        ss_ref = np.clip(ss_ref, 60e-3, 300e-3)
        ss_cur = np.clip(ss_cur, 60e-3, 300e-3)

        ss_factor = ss_cur / ss_ref

        # 存储调试信息到params
        if not hasattr(params_current, '_debug_ss'):
            params_current['_debug_ss'] = {
                'structure_cur': structure_cur,
                'ss_body_factor': ss_body_factor,
                'Cit_ref': Cit_ref,
                'Cit_cur': Cit_cur,
                'Cox': Cox,
                'ss_ref': ss_ref,
                'ss_cur': ss_cur,
                'ss_factor': ss_factor
            }

        return ss_factor
    
    @staticmethod
    def scale_idvg_curve(vg_ref, ids_ref, params_current, params_reference, v_d=5.1):
        """
        基于物理模型缩放IdVg曲线
        
        考虑因素:
        1. 阈值电压偏移 (ΔVth) - 包括buf层、gi层、掺杂、陷阱等
        2. 亚阈值摆幅变化 (SS scaling)
        3. 电流幅度缩放 (W/L, 厚度, 迁移率, 源漏电阻)
        4. Vd依赖关系 (线性区vs饱和区)
        """
        # 提取参考曲线参数
        ref_params = TFTPoissonSolver.extract_reference_params(vg_ref, ids_ref)
        
        # 计算各种偏移和缩放因子
        delta_vth = TFTPoissonSolver.calculate_vth_shift(params_current, params_reference)
        ss_factor = TFTPoissonSolver.calculate_ss_factor(params_current, params_reference)
        
        # 几何尺寸缩放
        W_cur, L_cur = params_current['W'], params_current['L']
        W_ref, L_ref = params_reference['W'], params_reference['L']
        scale_WL = (W_cur / L_cur) / (W_ref / L_ref)
        
        # 厚度缩放 (电流与沟道厚度成正比)
        t_igzo_cur = params_current['t_igzo']
        t_igzo_ref = params_reference['t_igzo']
        scale_thickness = t_igzo_cur / t_igzo_ref if t_igzo_ref > 0 else 1.0
        
        # 迁移率缩放 (与界面陷阱密度相关)
        dit_total_cur = params_current['dit_top'] + params_current['dit_bottom']
        dit_total_ref = params_reference['dit_top'] + params_reference['dit_bottom']
        scale_mobility_dit = np.exp(-(dit_total_cur - dit_total_ref) / 5e11)
        scale_mobility_dit = np.clip(scale_mobility_dit, 0.3, 3.0)

        structure_cur = params_current.get('structure_type', 'Double Gate')
        v_bg_fixed = params_current.get('v_bg_fixed', None)
        v_bg_floating = params_current.get('v_bg_floating', False)
        scale_mobility_body = 1.0

        if structure_cur == 'Single Gate (Top)':
            scale_mobility_body = 0.75

            if not v_bg_floating and v_bg_fixed is not None:
                scale_mobility_body -= 0.20 * (-v_bg_fixed) / 10.0

        scale_mobility = scale_mobility_dit * scale_mobility_body
        scale_mobility = np.clip(scale_mobility, 0.3, 3.0)

        # 源漏电阻影响
        L_source_cur = params_current.get('L_source', 3.0)
        Rs_sheet_cur = params_current.get('Rs_sheet', 3700.0)
        L_drain_cur = params_current.get('L_drain', 3.0)
        Rd_sheet_cur = params_current.get('Rd_sheet', 3700.0)
        
        L_source_ref = params_reference.get('L_source', 3.0)
        Rs_sheet_ref = params_reference.get('Rs_sheet', 3700.0)
        L_drain_ref = params_reference.get('L_drain', 3.0)
        Rd_sheet_ref = params_reference.get('Rd_sheet', 3700.0)
        
        # 计算源漏电阻 (Ω)
        R_source_cur = Rs_sheet_cur * L_source_cur * 1e-4 / (W_cur * 1e-4) if W_cur > 0 else 0
        R_drain_cur = Rd_sheet_cur * L_drain_cur * 1e-4 / (W_cur * 1e-4) if W_cur > 0 else 0
        R_total_cur = R_source_cur + R_drain_cur
        
        R_source_ref = Rs_sheet_ref * L_source_ref * 1e-4 / (W_ref * 1e-4) if W_ref > 0 else 0
        R_drain_ref = Rs_sheet_ref * L_drain_ref * 1e-4 / (W_ref * 1e-4) if W_ref > 0 else 0
        R_total_ref = R_source_ref + R_drain_ref
        
        # 总电流缩放因子
        scale_current = scale_WL * scale_thickness * scale_mobility
        
        # 创建新的Vg数组
        vg_new = vg_ref.copy()
        ids_new = np.zeros_like(ids_ref)
        
        # 提取参考Vth用于区域判断
        vth_ref = ref_params['vth']
        
        # 提取参考Ion用于电阻影响计算
        ion_ref = ref_params['ion']
        
        # 对每个点进行变换
        for i in range(len(vg_ref)):
            vg_i = vg_ref[i]
            ids_i = ids_ref[i]

            # 计算相对于参考Vth的栅压
            vg_relative = vg_i - vth_ref

            # 判断区域: 亚阈值区 vs 导通区
            if vg_relative < 0:
                # 亚阈值区: SS影响Vg轴的缩放，但不影响电流值
                # 更大的SS意味着需要更大的Vg变化才能达到相同的电流
                vg_new_relative = vg_relative * ss_factor
                # 保持电流不变，只通过Vg轴调整来体现SS差异
                ids_scale_sub = 1.0
            else:
                # 导通区: 主要是Vth偏移
                vg_new_relative = vg_relative
                ids_scale_sub = 1.0

            # 计算新的Vg
            vg_new[i] = (vth_ref + delta_vth) + vg_new_relative

            # 计算基础电流
            ids_base = ids_i * scale_current * ids_scale_sub
            
            # 应用源漏电阻修正
            # 更高的电阻 = 更低的有效电流
            if R_total_cur > 0 and ids_base > 0:
                # 计算当前电阻引起的压降
                v_drop_cur = ids_base * R_total_cur
                # 电阻越大，电流降低越多 (使用近似模型)
                resistance_factor = 1.0 / (1.0 + v_drop_cur / v_d) if v_d > 0 else 1.0
                ids_base = ids_base * resistance_factor
            
            ids_new[i] = ids_base
        
        return vg_new, ids_new
    
    @staticmethod
    def calculate_vth_simple(vg_array, ids_array, threshold=1e-9):
        mask = ~np.isnan(ids_array) & (ids_array > 0)
        vg = vg_array[mask]
        ids = ids_array[mask]
        
        if len(vg) < 3:
            return np.nan
        
        idx = np.where(ids >= threshold)[0]
        if len(idx) > 0:
            return vg[idx[0]]
        return np.nan
    
    @staticmethod
    def calculate_ss_simple(vg_array, ids_array):
        mask = ~np.isnan(ids_array) & (ids_array > 0)
        vg = vg_array[mask]
        ids = ids_array[mask]
        
        if len(vg) < 3:
            return np.nan
        
        sort_idx = np.argsort(vg)
        vg_sorted = vg[sort_idx]
        ids_sorted = ids[sort_idx]
        
        log_ids = np.log10(ids_sorted)
        
        def find_vg_at_log_current(target_log_current):
            if len(log_ids) < 2:
                return np.nan
            
            valid_mask = np.isfinite(log_ids)
            if np.sum(valid_mask) < 2:
                return np.nan
            
            vg_valid = vg_sorted[valid_mask]
            log_ids_valid = log_ids[valid_mask]
            
            if target_log_current < log_ids_valid.min() or target_log_current > log_ids_valid.max():
                return np.nan
            
            vg_interp = np.interp(target_log_current, log_ids_valid, vg_valid)
            return vg_interp
        
        vg_at_1e9 = find_vg_at_log_current(-9.0)
        vg_at_1e10 = find_vg_at_log_current(-10.0)
        
        if np.isnan(vg_at_1e9) or np.isnan(vg_at_1e10):
            return np.nan
        
        ss = vg_at_1e9 - vg_at_1e10
        
        return ss if ss > 0 else np.nan

    @staticmethod
    def calculate_mob_factor(params_current, params_reference):
        """Calculate mobility scaling factor based on device parameters."""
        eps0 = 8.854e-14
        dit_total_cur = params_current['dit_top'] + params_current['dit_bottom']
        dit_total_ref = params_reference['dit_top'] + params_reference['dit_bottom']
        scale_mobility_dit = np.exp(-(dit_total_cur - dit_total_ref) / 5e11)
        scale_mobility_dit = np.clip(scale_mobility_dit, 0.3, 3.0)

        # 计算底栅耦合效率
        eps_buf_sio_cur = params_current.get('eps_buf_sio', 3.9)
        t_buf_sio_cur = params_current.get('t_buf_sio', 300) * 1e-7
        eps_sin_cur = params_current.get('eps_sin', 7.0)
        t_sin_cur = params_current.get('t_sin', 100) * 1e-7

        Cox_buf_cur = eps0 * eps_buf_sio_cur / t_buf_sio_cur if t_buf_sio_cur > 0 else 1e-7
        Cox_sin_cur = eps0 * eps_sin_cur / t_sin_cur if t_sin_cur > 0 else 1e-7
        Cox_bottom_cur = 1.0 / (1.0 / Cox_sin_cur + 1.0 / Cox_buf_cur) if Cox_sin_cur > 0 and Cox_buf_cur > 0 else 1e-7

        Cox_buf_ref = eps0 * 3.9 / (300 * 1e-7)
        Cox_sin_ref = eps0 * 7.0 / (100 * 1e-7)
        Cox_bottom_ref = 1.0 / (1.0 / Cox_sin_ref + 1.0 / Cox_buf_ref)

        coupling_ratio = Cox_bottom_cur / Cox_bottom_ref if Cox_bottom_ref > 0 else 1.0

        structure_cur = params_current.get('structure_type', 'Double Gate')
        v_bg_fixed = params_current.get('v_bg_fixed', None)
        v_bg_floating = params_current.get('v_bg_floating', False)
        scale_mobility_body = 1.0

        if structure_cur == 'Single Gate (Top)':
            # 单栅基础迁移率比双栅低25%
            scale_mobility_body = 0.75

            # 根据底栅电压进一步调整
            if not v_bg_floating and v_bg_fixed is not None:
                # vbg效应强度与耦合效率成正比（buf越薄，效应越强）
                base_effect = 0.20 * (-v_bg_fixed) / 10.0
                scale_mobility_body -= base_effect * coupling_ratio

        scale_mobility = scale_mobility_dit * scale_mobility_body
        return np.clip(scale_mobility, 0.3, 3.0)

    @staticmethod
    def calculate_mobility_simple(vg_array, ids_array, v_ds, W, L, C_gi):
        mask = ~np.isnan(ids_array) & (ids_array > 0)
        vg = vg_array[mask]
        ids = ids_array[mask]
        
        if len(vg) < 5:
            return np.nan
        
        linear_mask = (ids > 1e-9) & (ids < np.max(ids) * 0.8)
        
        if np.sum(linear_mask) < 3:
            return np.nan
        
        vg_lin = vg[linear_mask]
        ids_lin = ids[linear_mask]
        
        coeffs = np.polyfit(vg_lin, ids_lin, 1)
        dIds_dVg = coeffs[0]
        
        if dIds_dVg <= 0 or v_ds == 0:
            return np.nan
        
        mobility = (L / W) * (1.0 / C_gi) * (1.0 / v_ds) * dIds_dVg
        
        return mobility
    
    @staticmethod
    def extract_ion_simple(vg_array, ids_array, vg_target=10.0):
        mask = ~np.isnan(ids_array)
        vg = vg_array[mask]
        ids = ids_array[mask]
        
        if len(vg) == 0:
            return np.nan
        
        idx = np.argmin(np.abs(vg - vg_target))
        
        return ids[idx]