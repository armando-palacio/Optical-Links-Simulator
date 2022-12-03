import numpy as np
import scipy.special as sp
from scipy.constants import pi
import scipy.integrate as integ

Q = lambda x: 0.5*sp.erfc(x/2**0.5)

## S0 y S1 son las varianzas no las desviaciones estándar
def mu1_S1(R,nf,g,fo,Ps,Dv,Df,T):
    from scipy.constants import e,k,h
    N = Dv/Df; Rl = 50 #ohms
    Pase = Dv*nf*h*fo*(g-1)
    S_sig_sp = 2 * R**2 * g * Ps * Pase / N 
    S_sp_sp = R**2 * Pase**2 / N * (1 - 1/2/N)
    S_T = 4 * k * T * Df / Rl 
    S_S = 2 * e * R * (g * Ps + Pase) * Df
    return R*(g*Ps+Pase), S_sig_sp + S_sp_sp + S_T + S_S

def mu0_S0(R,nf,g,fo,Dv,Df,T):
    from scipy.constants import e,k,h
    N = Dv/Df; Rl = 50 #ohms
    Pase = Dv*nf*h*fo*(g-1)
    S_sp_sp = R**2 * Pase**2 / N * (1 - 1/2/N)
    S_T = 4 * k * T * Df / Rl 
    S_S = 2 * e * R * Pase * Df
    return R*Pase, S_sp_sp + S_T + S_S

def Peb_ook(R,nf,g,fo,Ps,Dv,Df,T):
    mu0,S0 = mu0_S0(R,nf,g,fo,Dv,Df,T); s0 = S0**0.5
    mu1,S1 = mu1_S1(R,nf,g,fo,Ps,Dv,Df,T); s1 = S1**0.5
    x = 1/(S1-S0)*(mu0*S1-mu1*S0+s1*s0*((mu1-mu0)**2+2*(S1-S0)*np.log(s1/s0))**0.5)
    return 1/2*np.min(Q((mu1-x)/s1) + Q((x-mu0)/s0))

def Peb_H(M,R,nf,g,fo,Ps,Dv,Df,T):
    mu0,S0 = mu0_S0(R,nf,g,fo,Dv,Df,T); s0 = S0**0.5
    mu1,S1 = mu1_S1(R,nf,g,fo,Ps,Dv,Df,T); s1 = S1**0.5
    x = 1/(S1-S0)*(mu0*S1-mu1*S0+s1*s0*((mu1-mu0)**2+2*(S1-S0)*np.log(s1/s0*(M-1)))**0.5)
    pe_slot_min = np.min(1/M*Q((mu1-x)/s1) + (M-1)/M*Q((x-mu0)/s0)) 
    pe_symb = 1-(1-pe_slot_min)**M
    return M/2/(M-1)*pe_symb

def Peb_S(M,R,nf,g,fo,Ps,Dv,Df,T):
    mu0,S0 = mu0_S0(R,nf,g,fo,Dv,Df,T); s0 = S0**0.5
    mu1,S1 = mu1_S1(R,nf,g,fo,Ps,Dv,Df,T); s1 = S1**0.5
    pe_symb = 1-1/(2*pi)**0.5*integ.quad(lambda x: (1-Q((mu1-mu0+s1*x)/s0))**(M-1)*np.exp(-x**2/2),-np.inf,np.inf)[0]
    return M/2/(M-1)*pe_symb

"""
    # Ejemplo de uso
    >>>R = 0.9; nf = 2; g = 100; fo = 3e8/1550e-9; Ps = 1e-3; Dv = 25e9; Df = 500e6; T = 300; M = 4 
    >>>print(Peb_ook(R,nf,g,fo,Ps,Dv,Df,T))
    >>>print(Peb_H(M,R,nf,g,fo,Ps,Dv,Df,T))
    >>>print(Peb_S(M,R,nf,g,fo,Ps,Dv,Df,T))

    En el caso de que se quiera generar una curva de BER para direrentes valore por ejemplo de Ps
    se puede hacer de la siguiente manera:
    
    >>>Ps = np.linspace(1e-3,1e-2,100)
    >>>BER_ook = np.vectorize(Peb_ook)(R,nf,g,fo,Ps,Dv,Df,T)
    >>>BER_H = np.vectorize(Peb_H)(M,R,nf,g,fo,Ps,Dv,Df,T)
    >>>BER_S = np.vectorize(Peb_S)(M,R,nf,g,fo,Ps,Dv,Df,T) 
"""
