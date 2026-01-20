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
                    if sparse.issparse(delta):
                        delta = delta.toarray().flatten()
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
    def load_reference_idvg(csv_path='/Users/rinn_kennroku/Desktop/tft_sim/双栅基准idvg.csv'):
        try:
            data = np.genfromtxt(csv_path, delimiter=',', skip_header=2)
            vg_ref = data[:, 0]
            ids_ref = data[:, 1]
            return vg_ref, ids_ref
        except:
            return None, None
    
    @staticmethod
    def scale_idvg_curve(vg_ref, ids_ref, params_current, params_reference):
        W_cur, L_cur = params_current['W'], params_current['L']
        t_igzo_cur = params_current['t_igzo']
        nd_cur = params_current['nd']
        dit_top_cur = params_current['dit_top']
        dit_bot_cur = params_current['dit_bottom']
        
        W_ref, L_ref = params_reference['W'], params_reference['L']
        t_igzo_ref = params_reference['t_igzo']
        nd_ref = params_reference['nd']
        dit_top_ref = params_reference['dit_top']
        dit_bot_ref = params_reference['dit_bottom']
        
        scale_WL = (W_cur / L_cur) / (W_ref / L_ref)
        
        scale_nd = nd_cur / nd_ref
        
        scale_dit = 1.0
        if dit_top_ref > 0 or dit_bot_ref > 0:
            dit_total_ref = dit_top_ref + dit_bot_ref
            dit_total_cur = dit_top_cur + dit_bot_cur
            scale_dit = np.exp(-(dit_total_cur - dit_total_ref) / 1e11)
        
        ids_scaled = ids_ref * scale_WL * scale_nd * scale_dit
        
        return vg_ref.copy(), ids_scaled
    
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