import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

class TFTPoissonSolver:
    def __init__(self, length=2.0, 
                 t_buffer=0.2, eps_buffer=6.0,
                 t_igzo=0.05, eps_igzo=10.0, nd_igzo=1e16,
                 t_gi=0.1, eps_gi=3.9,
                 structure_type='Double Gate',
                 nx=50, ny=400): 
        
        self.q = 1.602e-19
        self.Vt = 0.0259
        self.eps0 = 8.854e-14 
        self.Nc = 5.0e18      
        self.N_off = nd_igzo  
        
        self.L_cm = length * 1e-4
        self.t_buffer_cm = t_buffer * 1e-4
        self.t_igzo_cm = t_igzo * 1e-4
        self.t_gi_cm = t_gi * 1e-4
        
        self.eps_buf = eps_buffer
        self.eps_igzo = eps_igzo
        self.eps_gi = eps_gi
        
        self.structure_type = structure_type
        self.nx = int(nx)
        self.user_ny = int(ny) # 保存用户想要的高精度
        
        self._init_mesh_final()

    def _init_mesh_final(self):
        y_if1 = self.t_buffer_cm
        y_if2 = self.t_buffer_cm + self.t_igzo_cm
        y_top = y_if2 + self.t_gi_cm
        
        # 1. IGZO Mesh: 严格使用用户定义的高精度
        n_igzo = self.user_ny
        y_igzo = np.linspace(y_if1, y_if2, n_igzo)
        
        # 2. Dielectrics: 适当的点数即可
        n_dielec = 60
        y_buf = np.linspace(0, y_if1, n_dielec)
        y_gi = np.linspace(y_if2, y_top, n_dielec)
        
        y_all = np.concatenate([y_buf, y_igzo, y_gi])
        self.y = np.unique(y_all)
        self.ny = len(self.y)
        
        self.x = np.linspace(0, self.L_cm, self.nx)
        self.dx = self.x[1] - self.x[0]
        
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Masks
        tol = 1e-12
        self.mask_buffer = (self.Y < y_if1 - tol)
        self.mask_igzo   = (self.Y >= y_if1 - tol) & (self.Y <= y_if2 + tol)
        self.mask_gi     = (self.Y > y_if2 + tol)
        
        self.epsilon_map = np.zeros_like(self.X)
        self.epsilon_map[self.mask_buffer] = self.eps_buf
        self.epsilon_map[self.mask_igzo]   = self.eps_igzo
        self.epsilon_map[self.mask_gi]     = self.eps_gi

    def calculate_n_from_phi(self, phi_val, v_ch_val):
        u = (phi_val - v_ch_val) / self.Vt
        val_log = np.logaddexp(0, u)
        n_val = self.N_off + self.Nc * val_log
        return n_val

    def solve(self, v_top_gate_bias, v_ds, v_bot_gate_bias=0.0):
        v_tg = v_top_gate_bias
        v_bg = 0.0 if self.structure_type == 'Source-Gated Bottom' else v_bot_gate_bias
        
        v_ch = np.zeros((self.ny, self.nx))
        for i in range(self.nx): v_ch[:, i] = v_ds * (i/(self.nx-1))
            
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
                    elif ix==self.nx-1 and self.mask_igzo[iy,ix]: is_bc=True; val=v_ds
                    
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
            try: delta = spsolve(J, rhs)
            except: break
            phi_flat = phi.flatten() + delta
            phi = phi_flat.reshape((self.ny,self.nx))
            if np.max(np.abs(delta)) < 1e-4: break
            
        self.phi = phi
        self.n_conc = self.calculate_n_from_phi(phi, v_ch)
        self._calc_field()
        return self.phi, self.n_conc, self.E_field

    def _calc_field(self):
        self.Ey, self.Ex = np.gradient(self.phi, self.y, self.x)
        self.Ey, self.Ex = -self.Ey, -self.Ex
        self.E_field = np.sqrt(self.Ex**2 + self.Ey**2)
