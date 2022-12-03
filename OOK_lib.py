#### Importamos librerías necesarias ####
from numpy.fft import fftfreq, fftshift
from scipy.constants import e,c,h,pi
from scipy.constants import k as kB
from matplotlib.pyplot import *
import sklearn.cluster as sk
import scipy.special as sp
import scipy.signal as sg
import numpy as np
import time as tm
import sys


#### Funciones de control ####
def tic(): global __; __ = tm.time()

def toc(): global __; ___ = tm.time()-__; print('Execution time: %.2f'%___); return ___ 

def error(msg): print(msg); sys.exit()


#### Definimos funciones útiles ####
gaus = lambda x,mu,std: 1/std/(2*pi)**0.5*np.exp(-0.5*(x-mu)**2/std**2) # función gaussiana

Qfunc = lambda x: 1/2*sp.erfc(x/np.sqrt(2)) # Función Q(x)

db = lambda x: 10*np.log10(x) # función de conversión a dB

dbm = lambda x: 10*np.log10(x*1e3) # función de conversión a dBm

idb = lambda x: 10**(x/10) # función de conversión a veces

idbm = lambda x: 10**(x/10-3) # función de conversión a Watts

diff_lag = lambda data,lag: data[lag:]-data[:-lag]  # Diferencia entre dos elemento de un array separados una distancia 'lag'

def power(sig, unit='W'): 
    """
    Determina la potencia media del vector 'sig'. Devuelve el valor en 'W' o 'dBm'. Por defecto, en dBm. 
    """
    p = np.mean(np.abs(sig)**2, axis=-1)
    if unit=='w' or unit=='W':
        return p
    elif unit=='dbm' or unit=='dBm':
        return dbm(p)
    else:
        error('Unidad no reconocida!!')

def find_nearest(levels, data): 
    """
    Encuentra el elemento de 'levels' más cercano a cada valor de 'data'.

    Entradas:
    levels: niveles de referencia.
    data: valores a comparar.

    Retorna:
    out: vector o entero con los valores de 'levels' correspondientes a cada valor de 'data'
    """
    if type(data) != np.ndarray and type(data) != list:
        return levels[np.argmin(np.abs(levels - data))]
    else:
        return levels[np.argmin( np.abs( np.repeat([levels],len(data),axis=0) - np.reshape(data,(-1,1)) ),axis=1 )]

def shorth_int(data):
    """
    Estimación del intervalo más corto que contiene el 50% de las muestras de 'data'.

    Entradas:
    data: array de datos
    
    Retorna:
    int: intervalo más corto que contiene el 50% de las muestras de 'data'
    """
    data = np.sort(data)
    lag = len(data)//2
    diff = diff_lag(data,lag)
    i = np.where(np.abs(diff - np.min(diff))<1e-10)[0]
    if len(i)>1:
        i = int(np.mean(i))
    return (data[i], data[i+lag])

# Función que calcula los parámetros del diagrama de ojo (Paper: A Robust Algorithm for Eye-Diagram Analysis)
def eye_params(ydata, spb):
    """
    Estima todos los parámetros fundamentales y métricas del diagrama de ojo de la señal ydata.

    Entradas:
    ydata: señal a analizar
    sps: muestras por slot
    M: orden de modulación PPM 

    Retorna:
    eye: objeto de la clase 'Eye' con todos los parámetros y métricas del diagrama de ojo
    """
    n = len(ydata)%(2*spb)
    if n: ydata = ydata[:-n]

    ydata = np.roll(ydata, -(spb//2-1)) # Para centrar el ojo en el gráfico
    y_set = np.array(list(set(ydata)))

    # Obtenemos el centroide de las muestras en el eje Y
    vm = np.mean(sk.KMeans(n_clusters=2).fit(ydata.reshape(-1,1)).cluster_centers_)

    # obtenemos el intervalo más corto de la mitad superior que contiene al 50% de las muestras
    top_int = shorth_int(ydata[ydata>vm]) 
    # Obtenemos el LMS del nivel 1
    state_1 = np.mean(top_int)

    # obtenemos el intervalo más corto de la mitad inferior que contiene al 50% de las muestras
    bot_int = shorth_int(ydata[ydata<vm])
    # Obtenemos el LMS del nivel 0
    state_0 = np.mean(bot_int)

    # Obtenemos la amplitud entre los dos niveles 0 y 1
    d01 = state_1 - state_0

    # Tomamos el 75% de nivel de umbral
    v75 = state_1 - 0.25*d01

    # Tomamos el 25% de nivel de umbral
    v25 = state_0 + 0.25*d01

    # Creamos vector de tiempo normalizado a un tiempo de bit o de slot
    t = np.kron(np.ones((len(ydata)//spb)//2), np.linspace(-1,1,2*spb, endpoint=False))
    t_set = np.array(list(set(t)))

    # El siguiente vector se utilizará solo para determinar los tiempos de cruce
    tt = t[(ydata>v25)&(ydata<v75)]

    # Obtenemos el centroide de los datos de tiempo
    tm = np.mean(sk.KMeans(n_clusters=2).fit(tt.reshape(-1,1)).cluster_centers_)

    # Obtenemos el tiempo de cruce por la izquierda
    t_left = find_nearest(t_set, np.mean(tt[tt<tm]))

    # Obtenemos el tiempo de cruce por la derecha
    t_right = find_nearest(t_set, np.mean(tt[tt>tm]))

    # Determinamos el centro del ojo
    t_center = find_nearest(t_set, (t_left + t_right)/2)

    # Para el 20% del centro del diagrama de ojo
    t_dist = t_right - t_left
    t_span0 = t_center - 0.1*t_dist
    t_span1 = t_center + 0.1*t_dist

    # Dentro del 20% de los datos del centro del diagrama de ojo, separamos en dos clusters superior e inferior
    y_center = find_nearest(y_set, (state_0 + state_1)/2)

    # Obtenemos el instante óptimo para realizar el down sampling
    instant = np.abs(t-t_center).argmin() - (spb//2 + 1)

    # Obtenemos el cluster superior
    y_top = ydata[(ydata > y_center) & ((t_span0 < t) & (t < t_span1))]

    # Obtenemos el cluster inferior
    y_bot = ydata[(ydata < y_center) & ((t_span0 < t) & (t < t_span1))]

    # Para cada cluster calculamos las medias y desviaciones estándar
    mu1 = np.mean(y_top)
    s1 = np.std(y_top)
    mu0 = np.mean(y_bot)
    s0 = np.std(y_bot)

    # Obtenemos la relación de extinción
    er = 10*np.log10(mu1/mu0)

    # Obtenemos la apertura del ojo
    eye_h = mu1 - 3 * s1 - mu0 - 3 * s0

    # obtenemos el umbral de decisión
    umbral = 1/(s1**2-s0**2)*(mu0*s1**2-mu1*s0**2+s1*s0*((mu1-mu0)**2+2*(s1**2-s0**2)*np.log(s1/s0))**0.5)

    ### EXTRA ###
    # Calculamos los valores de amplitud correspondientes a los cruces de tiempo
    tll = t_left - 0.01 * t_dist
    thh = t_left + 0.01 * t_dist
    amp_dat = ydata[((ydata < v75) & (ydata > v25)) & ((tll < t) & (t < thh))]
    # Obtenemos la amplitud de los cruce por la izquierda
    ampx_1 = find_nearest(y_set, np.mean(amp_dat))

    tll = t_right - 0.01 * t_dist
    thh = t_right + 0.01 * t_dist
    amp_dat = ydata[((ydata < v75) & (ydata > v25)) & ((tll < t) & (t < thh))]
    # Obtenemos la amplitud de los cruce por la derecha
    ampx_2 = find_nearest(y_set, np.mean(amp_dat))

    # Claculamos el promedio
    ampx = find_nearest(y_set, (ampx_1 + ampx_2)/2)

    # Para calcular el valor RMS del jitter
    tll = t_left - 0.25 * t_dist
    thh = t_left + 0.25 * t_dist

    tj = t[((ampx_1 - 0.01 * d01 < ydata) & (ydata < ampx_1 + 0.01 * d01)) & ((tll < t) & (t < thh))]

    # Obtenemos la amplitud del jitter por la izquierda
    jitter_1 = np.std(tj)

    tll = t_right - 0.25 * t_dist
    thh = t_right + 0.25 * t_dist

    tj = t[((ampx_2 - 0.01 * d01 < ydata) & (ydata < ampx_2 + 0.01 * d01)) & ((tll < t) & (t < thh))]

    # Obtenemos la amplitud del jitter por la derecha
    jitter_2 = np.std(tj)

    # Obtenemos el valor RMS del jitter
    jitter = (jitter_1 + jitter_2)/2

    # Obtenemos el ancho del ojo
    eye_w = t_right - 3 * jitter_2 - t_left - 3 * jitter_1

    return EYE(t,ydata,t_left,t_right,t_dist,t_center,mu0,s0,mu1,s1,umbral,instant,y_bot,y_top,er,eye_h,eye_w,jitter,ampx,spb)

## Funciones para calcular la BER teórica. S0 y S1 son las varianzas no las desviaciones estándar
def mu1_S1(R,nf,g,fo,Ps,Dv,Df,T=300):
    """
    Esta función calcula la media y varianza del bit 1 a partir de los parámetros del receptor.

    Retorna:
    mu1: Media del bit 1
    S1: Varianza del bit 1
    """
    N = Dv/Df; Rl = 50 #ohms
    Pase = Dv*nf*h*fo*(g-1)
    S_sig_sp = 2 * R**2 * g * Ps * Pase / N 
    S_sp_sp = R**2 * Pase**2 / N * (1 - 1/2/N)
    S_T = 4 * kB * T * Df / Rl 
    S_S = 2 * e * R * (g * Ps + Pase) * Df
    return R*(g*Ps+Pase), S_sig_sp + S_sp_sp + S_T + S_S

def mu0_S0(R,nf,g,fo,Dv,Df,T=300):
    """
    Esta función calcula la media y varianza del bit 0 a partir de los parámetros del receptor.

    Retorna:
    mu0: Media del bit 0
    S0: Varianza del bit 0
    """
    N = Dv/Df; Rl = 50 #ohms
    Pase = Dv*nf*h*fo*(g-1)
    S_sp_sp = R**2 * Pase**2 / N * (1 - 1/2/N)
    S_T = 4 * kB * T * Df / Rl 
    S_S = 2 * e * R * Pase * Df
    return R*Pase, S_sp_sp + S_T + S_S

def BER_TEO(R,nf,g,fo,Ps,Dv,Df,T):
    """
    Esta función calcula la BER teórica a partir de los parámetros del receptor.

    Retorna:
    BER: BER teórica
    """
    mu0,S0 = mu0_S0(R,nf,g,fo,Dv,Df,T); s0 = S0**0.5
    mu1,S1 = mu1_S1(R,nf,g,fo,Ps,Dv,Df,T); s1 = S1**0.5
    x = 1/(S1-S0)*(mu0*S1-mu1*S0+s1*s0*((mu1-mu0)**2+2*(S1-S0)*np.log(s1/s0))**0.5)
    return 1/2*(Qfunc((mu1-x)/s1) + Qfunc((x-mu0)/s0))

#### Definimos las clases necesarias ####
class EYE:
    """
    Esta clase contiene los datos de un diagrama de ojo.
    """
    def __init__(self, t, y, t_left, t_right, t_bit, t_opt, mu0, std0, mu1, std1, umbral, instant, cluster0, cluster1, er, eye_h, eye_w, jitter, ampx, spb):
        self.x = t
        self.y = y
        self.t_left = t_left
        self.t_right = t_right
        self.t_bit = t_bit
        self.t_opt = t_opt
        self.mu0 = mu0
        self.s0 = std0
        self.mu1 = mu1
        self.s1 = std1
        self.umbral = umbral
        self.i = instant
        self.cluster0 = cluster0
        self.cluster1 = cluster1
        self.er = er
        self.eye_h = eye_h
        self.eye_w = eye_w
        self.jitter = jitter
        self.ampx = ampx
        self.spb = spb
    
    def __str__(self):
        mu0 = 'mu0 = %.3f' % self.mu0
        std0 = 'std0 = %.3f' % self.s0
        mu1 = 'mu1 = %.3f' % self.mu1
        std1 = 'std1 = %.3f' % self.s1
        er = 'ER = %.2f dB' % self.er
        umbral = 'umbral = %.3f' % self.umbral
        t_opt = 't_opt = %.3f' % self.t_opt
        i = 'i = %d' % self.i
        eye_h = 'eye_h = %.3f' % self.eye_h
        eye_w = 'eye_w = %.3f' % self.eye_w
        jitter = 'jitter = %.3f' % self.jitter
        ampx = 'ampx = %.3f' % self.ampx
        return '\n'.join([mu0, std0, mu1, std1, er, umbral, t_opt, i, eye_h, eye_w, jitter, ampx])

class Parametros:
    """
    Esta clase contiene los parámetros de simulación del enlace.
    """
    def __init__(self, P_laser, bit_rate, nbits, spb, BW_opt, BW_elec, G_rx, NF_rx, G_tx, landa, Responsivity, orden_elec, orden_opt, CHANNEL_LOSS):
        self.Ptx = P_laser
        self.ptx = idbm(P_laser)
        self.bit_rate = bit_rate
        self.nbits = nbits
        self.spb = spb
        self.BW_opt = BW_opt
        self.orden_opt = orden_opt
        self.BW_elec = BW_elec
        self.orden_elec = orden_elec
        self.h = h
        self.c = c
        self.G_rx = G_rx
        self.NF_rx = NF_rx
        self.G_tx = G_tx
        self.g_rx = idb(G_rx)
        self.nf_rx = idb(NF_rx)
        self.g_tx = idb(G_tx)
        self.landa = landa
        self.fo = c/(landa)
        self.R = Responsivity
        self.i_dark = 10e-9
        self.nt = nbits*spb
        self.dt = 1/bit_rate/spb
        self.T_windows = nbits / bit_rate
        self.t = np.arange(-self.T_windows/2, self.T_windows/2, self.dt)[:self.nt]
        self.f = fftshift(fftfreq(len(self.t), self.dt))
        self.CHANNEL_LOSS = CHANNEL_LOSS
        self.channel_loss = idb(-CHANNEL_LOSS)

class Transmisor:
    """
    Esta clase realiza la función de un transmisor y almacena todos los datos de la señal en cada etapa del mismo.
    """
    def __init__(self, parameters, prbs_type = 'random', user_prbs = None):
        self.p = parameters

        # secuencias de bits
        if prbs_type == 'random':
            self.ook_s = self.PRBS()
        elif prbs_type == 'user':
            if user_prbs is None:
                error('Debe introducir una secuencia de bits')
            self.ook_s = user_prbs

        # dominio eléctrico
        self.nrz_pulses = self.NRZ_PULSE_SHAPE(self.ook_s)

        # dominio óptico
        self.opt_signal = self.MODULADOR(self.nrz_pulses)
        self.OUT        = self.EDFA(self.opt_signal)
    
    def PRBS(self):
        return np.random.randint(0,2, self.p.nbits)

    def NRZ_PULSE_SHAPE(self, seq, type = 'rect'):  
        x = np.kron(seq, np.ones(self.p.spb))
        if type == 'rect':
            return x

    def MODULADOR(self, elec_signal):
        return np.sqrt(self.p.ptx)*elec_signal

    def EDFA(self, signal_in):
        return np.sqrt(self.p.g_tx)*signal_in

class Canal:
    """
    Esta clase realiza la función de un canal y almacena todos los datos de la señal en cada etapa del mismo.
    """
    def __init__(self, parameters, signal_in):
        self.p = parameters
        self.IN = signal_in

        self.OUT = self.channel(self.IN)

    def channel(self, signal_in):
        return np.sqrt(self.p.channel_loss)*signal_in

class Receptor:
    """
    Esta clase realiza la función de un receptor y almacena todos los datos de la señal en cada etapa del mismo.
    """
    def __init__(self, parameters, signal_in, ook_tx):
        self.p = parameters
        self.IN = signal_in
        self.ook_tx = ook_tx

        tic()
        self.BER_teo, self.I0_teo, self.I1_teo, self.sigma0_teo, self.sigma1_teo, self.umbral_teo = self.TEO_PARAMETERS()
        
        # Amplificamos y filtramos la señal recibida
        self.x_rx_amp_polx, self.x_rx_amp_poly, self.p_sig_mean, self.p_ase = self.EDFA(self.IN)
        self.OSNR = db(2*self.p_sig_mean/self.p_ase)

        # Detectamos la señal con el Fotodiodo
        self.y_PD = self.PD(self.x_rx_amp_polx, self.x_rx_amp_poly)

        # Filtramos la señal proveniente del fotodiodo
        self.y_PD_filt = self.LPF(self.y_PD)

        # estimamos parámetros del ojo
        max_samples = min(6000000, self.p.nt)
        self.eye = self.EYE_PROCESS(self.y_PD_filt[:max_samples])

        # realizamos un submuestreo de la señal a una muestra por bit, en el instante del bit (i) 
        self.y_PD_filt_down = self.DOWN_SAMPLING(self.y_PD_filt, self.eye.i)

        # ADC (umbral de decisión)
        self.ook_rx = self.ADC(self.y_PD_filt_down, self.eye.umbral)

        # BER por conteo de errores
        self.BER_count = self.BER_COUNTER(self.ook_tx, self.ook_rx)

        self.BER_est, self.I0_est, self.I1_est, self.sigma0_est, self.sigma1_est = self.SIM_PARAMETERS()
        toc()


    def EDFA(self, signal_in):
        sig_amp = self.BPF( np.sqrt(self.p.g_rx)*signal_in )
        p_sig_mean = power(sig_amp)

        S_ase = self.p.nf_rx/2 * self.p.h * self.p.fo * (self.p.g_rx-1)
        P_ase = 2 * S_ase * self.p.BW_opt

        Eop = self.BPF( np.exp(-1j*np.random.uniform(0,2*np.pi, self.p.nt)) )
        Ecp = self.BPF( np.exp(-1j*np.random.uniform(0,2*np.pi, self.p.nt)) )

        norm = power(Eop)

        Eop /= norm**0.5 / (P_ase/2)**0.5
        Ecp /= norm**0.5 / (P_ase/2)**0.5

        p_noise = power(Eop) + power(Ecp)

        sig_pol_x = sig_amp + Ecp
        sig_pol_y = Eop
        return (sig_pol_x, sig_pol_y, p_sig_mean, p_noise)

    def BPF(self, signal_in):
        global sos_band
        sos_band = sg.bessel(N = self.p.orden_opt, Wn = self.p.BW_opt/2, btype = 'low', fs=1/self.p.dt, output='sos', norm='mag')
        return sg.sosfiltfilt(sos_band, signal_in)

    def PD(self, signal_x, signal_y, T=300, Rl = 50):
        S_T = 4 * kB * T * self.p.BW_elec / Rl
        thermal_noise = np.random.normal(0, S_T**0.5, signal_x.size)
        S_S = 2 * e * self.p.R * (power(signal_x) + power(signal_y)) * self.p.BW_elec
        shot_noise = np.random.normal(0, S_S**0.5, signal_x.size)

        return self.p.R * ( np.abs(signal_x)**2 + np.abs(signal_y)**2 ) + thermal_noise + shot_noise + self.p.i_dark

    def LPF(self, signal_in):
        global sos_low
        sos_low = sg.bessel(N = self.p.orden_elec, Wn = self.p.BW_elec, btype = 'low', fs=1/self.p.dt, output='sos', norm='mag')
        return sg.sosfiltfilt(sos_low, signal_in)

    def EYE_PROCESS(self, ydata):
        return eye_params(ydata, self.p.spb)

    def DOWN_SAMPLING(self, signal_over_sampled, instant):
        return signal_over_sampled[instant::self.p.spb]

    def ADC(self, signal_down, umbral):
        return signal_down > umbral

    def BER_COUNTER(self, prbs_in, prbs_out):
        errors = np.sum(prbs_in != prbs_out)
        if errors < 10:
            return 0
        return errors/len(prbs_in)

    def TEO_PARAMETERS(self):
        P1 = self.p.g_rx * 2 * power(self.IN)  # Potencia pico del símbolo '1'
        P_ASE = self.p.nf_rx * self.p.h * self.p.fo * (self.p.g_rx-1) * self.p.BW_opt  # Potencia de ASE

        I1 = self.p.R * (P1 + P_ASE)
        I0 = self.p.R * P_ASE

        sigma2_sigsp = 2*self.p.R**2 * P1 * P_ASE * self.p.BW_elec/self.p.BW_opt
        sigma2_spsp = self.p.R**2 * P_ASE**2 * ( self.p.BW_elec/self.p.BW_opt - self.p.BW_elec**2/2/self.p.BW_opt**2 )

        s0 = np.sqrt( sigma2_spsp )
        s1 = np.sqrt( sigma2_spsp + sigma2_sigsp )

        umbral = 1/(s1**2-s0**2)*(I0*s1**2-I1*s0**2+s1*s0*((I1-I0)**2+2*(s1**2-s0**2)*np.log(s1/s0))**0.5)
        BER = 1/2*Qfunc((I1-umbral)/s1) + 1/2*Qfunc((umbral-I0)/s0)

        return BER, I0, I1, s0, s1, umbral

    def PRINT_TEO_PARAMS(self):
        print('## Parámetros Teóricos ##')
        print('    I0 = %.2e' % self.I0_teo)
        print('    I1 = %.2e' % self.I1_teo)
        print('    s0 = %.2e' % self.sigma0_teo)
        print('    s1 = %.2e' % self.sigma1_teo)
        print('umbral = %.2e' % self.umbral_teo)
        print('   BER = %.2e' % self.BER_teo)
        print()

    def SIM_PARAMETERS(self):
        I0 = self.eye.mu0
        I1 = self.eye.mu1
        s0 = self.eye.s0
        s1 = self.eye.s1
        umbral = self.eye.umbral
        BER = 1/2*Qfunc((I1-umbral)/s1) + 1/2*Qfunc((umbral-I0)/s0)
        return BER, I0, I1, s0, s1

    def PRINT_SIM_PARAMS(self):
        print('## Parámetros Estimados ##')
        print('    I0 = %.2e' % self.I0_est)
        print('    I1 = %.2e' % self.I1_est)
        print('    s0 = %.2e' % self.sigma0_est)
        print('    s1 = %.2e' % self.sigma1_est)
        print('umbral = %.2e' % self.eye.umbral)
        print('   BER = %.2e' % self.BER_est)
        print()
  
    def PSD(self, signal, title_ ='', new_fig=True):
        f, pxx = sg.welch(signal, fs=1/self.p.dt, nfft=1024, return_onesided=False)
        if new_fig:
            figure()
            text(0, dbm(pxx).min(), 'power : %.2f dBm' % (dbm(np.mean(pxx)) + db(1/self.p.dt)), ha="center", va="center", size=10,
            bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
        
        plot(fftshift(f)*1e-9, dbm(fftshift(pxx)))
        title(title_)
        xlabel('Frecuencia [GHz]')
        ylabel('PSD [dBm/Hz]')
        grid()
    
    def EYE_DIAGRAM(self, nbits=1000, title='', show_=True, save=False):
        _,ax = subplots(1,2, sharey=True, gridspec_kw={'width_ratios': [4,1], 
                                                        'height_ratios': [4], 
                                                        'wspace': 0.1})

        eye = self.eye
        ax[0].plot(eye.x[self.p.spb: nbits*self.p.spb], eye.y[self.p.spb: nbits*self.p.spb], 'k.', lw=0.2, alpha=0.15)
   
        suptitle('Diagrama de ojo, BER_est = %.1e' % (self.BER_est))

        ax[0].set_title(title)
        ax[0].set_xlim(-1,1)
        ax[0].set_xlabel('Tiempo [t/T_b]', fontsize=12)
        ax[0].set_ylabel('Amplitud [V]', fontsize=12)
        ax[0].grid()
        ax[0].set_xticks([-1,-0.5,0,0.5,1])
        ax[0].text(eye.t_opt, eye.mu0-4*eye.s0, 't_opt = %.2f' % eye.t_opt, ha="center", va="center", size=10,
        bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))

        ax[0].axhline(eye.umbral, color = 'b', ls = '--', alpha = 0.5)
        ax[1].axhline(eye.umbral, color = 'b', ls = '--', alpha = 0.5)
        ax[0].axhline(eye.mu0, color = 'r', ls = '--', alpha = 0.5)
        ax[1].axhline(eye.mu0, color = 'r', ls = '--', alpha = 0.5)
        ax[0].axhline(eye.mu1, color = 'r', ls = '--', alpha = 0.5)
        ax[1].axhline(eye.mu1, color = 'r', ls = '--', alpha = 0.5)

        _,bins,_ = ax[1].hist(eye.y[(eye.x==eye.t_opt) | (eye.x==eye.t_opt-1) | (eye.x==eye.t_opt + 1)], bins=100, density=True, orientation = 'horizontal', color = 'b', alpha = 0.5)
        x = bins[:-1] + np.diff(bins)/2; x = x.astype(np.float)
        
        plot(1/2*gaus(x, eye.mu1, eye.s1), x, color = '#E18F00')
        plot(1/2*gaus(x, eye.mu0, eye.s0), x, color = '#E18F00')
        
        if save: 
            if title:
                savefig(title+'.png', dpi=300)
            else:
                savefig('eye_diagram.png', dpi=300)
        if show_: show()

    def FILTERS_RESPONSE(self):
        _,H_elec = sg.sosfreqz(sos_low, worN = self.p.f, fs = 1/self.p.dt)
        _,H_opt = sg.sosfreqz(sos_band, worN = self.p.f, fs = 1/self.p.dt)

        title('Respuesta en frecuencia de los filtros')

        plot(self.p.f*1e-9, db(abs(H_elec)**2), alpha = 0.5, label='Filtro eléctrico')
        axhline(-3,color = 'k', ls='--')
        axvline(self.p.BW_elec*1e-9, color = 'k', ls='--')
        axvline(-self.p.BW_elec*1e-9, color = 'k', ls='--')
        
        limx = max(self.p.BW_opt, self.p.BW_elec)

        plot(self.p.f*1e-9, db(abs(H_opt)**2), alpha = 0.5, label='Filtro óptico')
        xlim(-limx*1e-9, limx*1e-9)
        ylim(-15,1)
        axhline(-3,color = 'k', ls='--')
        axvline(-self.p.BW_opt/2*1e-9, color = 'k', ls='--')
        axvline(self.p.BW_opt/2*1e-9, color = 'k', ls='--')
        xlabel('Frecuencia [GHz]', fontsize=15)
        ylabel('Magnitud [dB]', fontsize=15)
        grid()
        legend()
        show()

    def SIG_BEFORE_THEN_FILT(self, nbits = 15):
        sec_plot = np.arange(self.p.nt/2, self.p.nt/2 + nbits*self.p.spb, dtype=int)
        plot(self.p.t[sec_plot]*self.p.bit_rate, self.y_PD[sec_plot]*1e3, 'b', alpha=0.5, label='señal detectada')
        plot(self.p.t[sec_plot]*self.p.bit_rate, self.y_PD_filt[sec_plot]*1e3, 'red', label = 'señal filtrada')
        plot(self.DOWN_SAMPLING(self.p.t, self.eye.i)*self.p.bit_rate, self.DOWN_SAMPLING(self.y_PD_filt, self.eye.i)*1e3, 'ko', label = 'decisión')
        axhline(self.eye.umbral*1e3, color = 'k', ls='--', label = 'umbral estimado')
        axhline(self.umbral_teo*1e3, color = 'y', ls='--', label = 'umbral teórico')
        xticks(np.arange(0,nbits+1))
        grid()
        xlabel('Time [t/Tb]')
        ylabel('Amplitud [u.a.]')
        xlim(0,nbits)
        legend()
        show()

""" 
    A la hora de simular conviene hacerlo en un archivo '.ipynb' de jupyter notebook, para tener acceso a todas las variables 
    de cada clase y no tener que ejecutar todo cada vez que se quiera acceder a una variable en específico.

    Example:
    --------
    from OOK_sim_lib import *

    p = Parametros(...)

    Tx = Transmisor(p)
    Ch = Canal(p, Tx.OUT)
    Rx = Receptor(p, Ch.OUT, Tx.ook_s)
"""