from PPM_lib import eye_params, decode, error, gaus
from matplotlib.pyplot import *
import scipy.signal as sg
import pandas as pd
import numpy as np

# FS_GEN = 500e6              # Frecuencia de muestreo del generador
# FS_OSC = 2.5e9              # Frecuencia de muestreo del osciloscopio
# SPB_GEN = 2                 # Samples per bit del generador

# TB = 1/FS_GEN*SPB_GEN  # Tiempo de bit 
# SPB_OSC = int(SPB_GEN/FS_GEN*FS_OSC)    # Samples per bit del osciloscopio

def read_csv(filename, skiprows = 20):
    """ Parámetros
        ----------
        filename : nombre del archivo .csv.
        skiprows : número de líneas a saltar.

        Retorna
        -------
        t : tiempo en [s].
        signal : señal analógica en [mV].
    """
    t,x = pd.read_csv(filename, skiprows=skiprows).to_numpy().T
    return t, x*1000


def load_seq_tx(filename, sps_gen):
    """ Parámetros
        ----------
        filename : nombre del archivo con la secuencia de bits transmitidos.
        sps_gen : muestras por slot en el generador.

        Retorna
        -------
        slots_tx : secuencia de slots transmitida.
    """
    return np.loadtxt(filename)[::sps_gen] == 0


def delay_correction(signal_rx, slots_tx, sps_osc):
    """ Corrige el error temporal entre la señal transmitida y recibida.

        Parámetros
        ----------
        signal : señal analógica.
        slots_tx : secuencia de slots que se transmitió.
        sps_osc : muestras por slot en el osciloscopio.

        Retorna
        -------
        signal_corr : señal analógica corregida.
        i : posición a partir de la cual se realiza la corrección.
    """
    signal_tx = np.kron(slots_tx, np.ones(sps_osc))

    if len(signal_rx)<len(signal_tx): error('La longitud de los datos debe ser mayor al de la prbs!!')
    l = len(signal_tx)

    corr = sg.fftconvolve(signal_rx[:2*l], signal_tx[::-1], mode='valid')
    i = sg.find_peaks(corr, height= np.mean((max(corr),min(corr))))[0][0]
    signal_corr = signal_rx[i:]
    return signal_corr, i


def estimate_eye_params(signal, sps_osc, M):
    """
    Estima los parámetros del ojo de la señal recibida.

    Entradas:
    signal: señal analógica recibida
    sps_osc: muestras por slot en el osciloscopio
    M: orden de modulación PPM
    
    Retorna:
    eye: parámetros del ojo
    """
    l = np.min( len(signal), 2**21 )
    return eye_params(signal[:l], sps_osc, M)


def down_sampled(signal, sps_osc, instant = None):
    """ Parámetros
        ----------
        signal : señal analógica.
        sps_osc : muestras por slot en el osciloscopio.
        instant : posición del slot partir del cual se realiza el downsampling. Default: spb_osc//2-1.
        
        Retorna
        -------
        signal_down : señal analógica submuestreada (una muestra por slot).
    """
    if not instant:
        instant = sps_osc//2-1
    return signal[instant::sps_osc]


def decode(slots, M, decision):
    """
    Recibe una secuencia de bits codificada en PPM y la decodifica.

    Entradas:
    ppm_s: secuencia de bits codificada en PPM
    M: orden de modulación PPM
    decision: tipo de decisión

    Retorna:
    ook_s: secuencia de bits decodificada
    """
    if M > 256: error('El máximo orden de modulación PPM es 256!!')

    k = int(np.log2(M))
    n_simb = int(len(slots)/M)

    symbols = np.reshape(slots,(n_simb,M)) # convertimos la secueencia en una matriz de (n_simb x M)

    if decision == 'hard' or decision == 'HARD':
        i,j = np.where(symbols == 1) # filas y columnas de los 1's
        if len(set(i))!=n_simb: # si el número de filas no coincide con n_simb
            error('Error en la decodificación, hay símbolos que no contienen ningún pulso!!')
        decimal = np.array([np.random.choice(j[i==k]) for k in range(n_simb)], dtype=np.uint8).reshape(-1,1) # elegimos aleatoriamente una de las columnas que contiene más de un 1, para cada fila
    
    elif decision == 'soft' or decision == 'SOFT':
        decimal = np.argmax(symbols, axis=1).reshape(-1,1).astype(np.uint8) # elegimos la columna máxima para cada fila
    
    else:
        error('No existe el tipo de decisión seleccionada!!')

    bits = np.unpackbits(decimal, axis=1)[:,-k:].reshape(n_simb*k) # convertimos de decimal a binario y organizamos la matriz en una secuencia nuevamente
    return bits


def hard_decision(signal_downsampled, umbral, M):
    """ Parámetros
        ----------
        signal_downsampled : señal analógica, con una muestra por slot.
        umbral : umbral de decisión.
        M : cantidad de slots del símbolo PPM.

        Retorna
        -------
        seq_ook : secuencia digital decodificada.
    """
    return decode(signal_downsampled > umbral, M, decision='hard') 


def soft_decision(signal_downsampled, M):
    """ Parámetros
        ----------
        signal_downsampled : señal analógica, con una muestra por slot.
        M : cantidad de slots del símbolo PPM.

        Retorna
        -------
        seq_ook : secuencia digital decodificada.
    """
    return decode(signal_downsampled, M, decision='soft')


def error_count(bits_tx, bits_rx):
    """ Parámetros
        ----------
        bits_rx : secuencia de bits recuperada.
        bits_tx : secuencia de bits que se transmitió.
        
        Retorna
        -------
        err : número de errores dividido la cantidad de bits chequeados.
    """
    # se determina cuantas veces es más grande la secuencia recibida que la transmitida
    m = int(len(bits_rx)/len(bits_tx))-1 

    # Se repite m veces la seq_tx para que coincida con la señal del osciloscopio    
    bits_tx_m = np.kron(np.ones(m), bits_tx)

    # Se cuentan la cantidad de bits que no coinciden y se divide entre el total de bits
    err = np.sum(bits_tx_m != bits_rx[:len(bits_tx_m)])/len(bits_tx_m)  
    return err
    

def receptor_ppm(signal_rx, slots_tx, M, sps_osc):
    """ Parámetros
        ----------
        slots_tx : secuencia de slots transmitida.
        signal_rx : señal analógica recibida.
        M : orden de modulación PPM.
        sps_osc : muestras por slot en el osciloscopio.

        Retorna
        -------
        BER_H : tasa de error de bits para la decisión 'hard'.
        BER_S : tasa de error de bits para la decisión 'soft'.
        signal_rx_c : señal analógica corregida temporalmente.
        eye : parámetros del diagrama de ojo.
    """
    # Se corrige el error temporal de la señal recibida
    signal_rx_c,_ = delay_correction(signal_rx, slots_tx, sps_osc)

    # Se estiman el instante de decisión (i) y el umbral de decisión (umbral)
    eye = estimate_eye_params(signal_rx_c, sps_osc, M)

    # Se realiza un submuestreo de la señal 
    signal_down = down_sampled(signal_rx_c, sps_osc, eye.i)

    # Se realiza la decodificación dura
    bits_rx_H = hard_decision(signal_down, eye.umbral, M)

    # Se realiza la decodificación blanda
    bits_rx_S = soft_decision(signal_down, M)

    # Se decodifica la secuencia transmitida (seq_tx)
    bits_tx = decode(slots_tx, M, decision='hard')

    # Se determinan las tasas de error
    BER_H = error_count(bits_tx, bits_rx_H)
    BER_S = error_count(bits_tx, bits_rx_S)
    return BER_H, BER_S, signal_rx_c, eye


def gen_hist(data):
    """ Genera un histograma a partir de una secuencia de datos con amplitudes discretas."""
    x = list(set(data)); x.sort()
    y = []
    for X in x:
        y.append(len(data[data == X]))
    return x,y


def EYE_DIAGRAM(eye, show_=True, title ='', save=False):
    _,ax = subplots(1,2, sharey=True, gridspec_kw={'width_ratios': [4,1], 
                                                    'height_ratios': [4], 
                                                    'wspace': 0.1})

    ax[0].plot(eye.x, eye.y, 'k.', lw=0.2, alpha=0.15)
        
    suptitle('Diagrama de ojo')
    ax[0].set_title(title)
    ax[0].set_xlim(-1,1)
    ax[0].set_xlabel('Tiempo [t/T_b]', fontsize=12)
    ax[0].set_ylabel('Amplitud [mV]', fontsize=12)
    ax[0].grid()
    ax[0].set_xticks([-1,-0.5,0,0.5,1])
    ax[0].text(eye.t_opt, eye.mu0-4*eye.s0, 't_opt = ' + str(round(eye.t_opt,1)), ha="center", va="center", size=10,
    bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))

    ax[0].axhline(eye.umbral, color = 'g', ls = '--', alpha = 0.5)
    ax[1].axhline(eye.umbral, color = 'g', ls = '--', alpha = 0.5)
    ax[0].axhline(eye.mu0, color = '#00054F', ls = '--', alpha = 0.5)
    ax[1].axhline(eye.mu0, color = '#00054F', ls = '--', alpha = 0.5)
    ax[0].axhline(eye.mu1, color = '#054F00', ls = '--', alpha = 0.5)
    ax[1].axhline(eye.mu1, color = '#054F00', ls = '--', alpha = 0.5)

    if eye.height>0:
        v0_3s = eye.mu0+2.5*eye.s0
        v1_3s = eye.mu1-2.5*eye.s1

        ax[0].axhline(v0_3s, xmin = 0.375, xmax = 0.625, color = 'r', ls = '--', alpha = 0.8)
        ax[0].axhline(v1_3s, xmin = 0.375, xmax = 0.625, color = 'r', ls = '--', alpha = 0.8)
        ax[0].annotate("",xy=(eye.t_opt, v0_3s), xytext=(eye.t_opt, v1_3s), arrowprops=dict(arrowstyle="<->", color="r", alpha=0.8))

    _,bins,_ = ax[1].hist(eye.y[(eye.x==eye.t_opt) | (eye.x==eye.t_opt-1) | (eye.x==eye.t_opt + 1)], bins=100, density=True, orientation = 'horizontal', color = 'b', alpha = 0.5)
    x = bins[:-1] + np.diff(bins)/2; x = x.astype(np.float)
    
    plot(1/eye.M*gaus(x, eye.mu1, eye.s1), x, color = '#E18F00')
    plot((eye.M-1)/eye.M*gaus(x, eye.mu0, eye.s0), x, color = '#E18F00')

    if save: 
        if title:
            savefig(title+'.png', dpi=300)
        else:
            savefig('eye_diagram.png', dpi=300)
    if show_: show()


def SHOW_SIGNAL_DECISION(eye, nbits, title = '', show_=True, save=False):
    """ Muestra la señal recibida y los instantes de decisión 

        Parámetros
        ----------
        eye : parámetros del diagrama de ojo.
        nbits : int. Número de bits a mostrar
        title : str. Título de la gráfica
        show_ : bool. Si se muestra la gráfica
        save : bool. Si se guarda la gráfica

        Retornos
        ----------
        None
    """
    t = np.linspace(0,nbits,nbits*eye.sps, endpoint=False)
    y = np.roll(eye.y, eye.sps//2-1)

    figure()
    plot(t[eye.i::eye.sps], y[eye.i:nbits*eye.sps:eye.sps], 'ro')
    plot(t, y[:nbits*eye.sps], '.b')
    xticks(np.arange(0,nbits+1,1))
    grid()
    xlabel('Tiempo [t/T_b]')
    ylabel('Amplitud [mV]')
    suptitle(title)

    if save:
        if title:
            savefig(title+'.png', dpi=300)
        else:
            savefig('signal_decision.png', dpi=300)
    if show_: show()


                                                                      
