from OOK_lib import eye_params, error, gaus
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
        data : señal analógica en [mV].
    """
    t,x = pd.read_csv(filename, skiprows=skiprows).to_numpy().T
    return t, x*1000


def load_seq_tx(filename, spb_gen):
    """ Parámetros
        ----------
        filename : nombre del archivo con la secuencia de bits transmitidos.
        spb_gen : muestras por bit en el generador.

        Retorna
        -------
        seq_tx : secuencia de bits transmitida.
    """
    return np.loadtxt(filename)[::spb_gen] == 0


def delay_correction(signal, seq_tx, spb_osc):
    """ Corrige el error temporal entre la señal transmitida y recibida.

        Parámetros
        ----------
        signal : señal analógica.
        seq_tx : secuencia de bits que se transmitió.
        spb_osc : muestras por bit en el osciloscopio.

        Retorna
        -------
        signal_corr : señal analógica corregida.
        i : posición a partir de la cual se realiza la corrección.
    """
    seq_tx = np.kron(seq_tx, np.ones(spb_osc))

    if len(signal)<len(seq_tx): error('La longitud de los datos debe ser mayor al de la prbs!!')
    l = len(seq_tx)

    corr = sg.fftconvolve(signal[:2*l], seq_tx[::-1], mode='valid')
    i = np.argmax(corr)
    signal_corr = signal[i:]
    return signal_corr, i


def estimate_eye_params(signal, spb_osc):
    """
    Estima los parámetros del ojo de la señal recibida.

    Entradas:
    signal: señal analógica recibida
    spb_osc: muestras por bit en el osciloscopio
    M: orden de modulación PPM
    
    Retorna:
    eye: parámetros del ojo
    """
    l = min( len(signal), 2**21 )
    return eye_params(signal[:l], spb_osc)


def down_sampled(signal, spb_osc, instant = None):
    """ Parámetros
        ----------
        signal : señal analógica.
        spb_osc : muestras por bit en el osciloscopio.
        instant : posición del bit partir del cual se realiza el downsampling. Default: spb_osc//2-1.
        
        Retorna
        -------
        signal_down : señal analógica submuestreada (una muestra por bit).
    """
    if not instant:
        instant = spb_osc//2-1
    return signal[instant::spb_osc]


def decision(signal_downsampled, umbral):
    """ Parámetros
        ----------
        signal_downsampled : señal analógica, con una muestra por bit.
        umbral : umbral de decisión.

        Retorna
        -------
        seq_ook : secuencia digital binaria
    """
    return signal_downsampled > umbral 


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
    m = len(bits_rx)/len(bits_tx)

    # Se repite m veces la seq_tx para que coincida con la señal del osciloscopio    
    if m > 1:
        seq_tx_m = np.kron(np.ones(int(m)), bits_tx)
    elif m < 1:
        seq_tx_m = bits_tx[0:len(bits_rx)]
    else:
        seq_tx_m = bits_tx
    # Se cuentan la cantidad de bits que no coinciden y se divide entre el total de bits
    err = np.sum(seq_tx_m != bits_rx[:len(seq_tx_m)]) 
    if err < 10:
        return 0
    return err/len(seq_tx_m)
    

def receptor_ook(signal_rx, bits_tx, spb_osc):
    """ Parámetros
        ----------
        bits_tx : secuencia de bits transmitida.
        signal_rx : señal analógica recibida.
        spb_osc : muestras por bit en el osciloscopio.

        Retorna
        -------
        BER : tasa de error de bits.
        signal_rx_c : señal analógica corregida temporalmente.
        eye : parámetros del diagrama de ojo.
    """
    # Se corrige el error temporal de la señal recibida
    signal_rx_c,_ = delay_correction(signal_rx, bits_tx, spb_osc)

    # Se estiman el instante de decisión (i) y el umbral de decisión (umbral)
    eye = estimate_eye_params(signal_rx_c, spb_osc)

    # Se realiza un submuestreo de la señal, una muestra por bit
    signal_down = down_sampled(signal_rx_c, spb_osc, eye.i)

    # Se toma la decisión a partir del umbral
    bits_rx = decision(signal_down, eye.umbral)

    # Se determinan las tasas de error
    BER = error_count(bits_tx, bits_rx)
    return BER, signal_rx_c, eye


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
    t = np.linspace(0,nbits,nbits*eye.spb, endpoint=False)
    y = np.roll(eye.y, (eye.spb//2-1))

    figure()
    plot(t[eye.i::eye.spb], y[eye.i:nbits*eye.spb:eye.spb], 'ro')
    plot(t, y[:nbits*eye.spb], '.b')
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
