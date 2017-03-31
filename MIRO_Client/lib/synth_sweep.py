"""Synth_sweep.

Python translations of Matlab SYNTHSWEEP.

Usage:
  synth_sweep.py [--T=t] [--FS=fs]
                 [--low=f1] [--high=f2] [--tail] FILENM
  synth_sweep.py (-h | --help)

Arguments:
  FILENM  name of file in which to store output

Options:
  -h --help     Show this screen.
  --T=t       Duration of sweep in seconds [default: 10].
  --FS=fs     Sampling frquency [default: 48000].
  --low=f1    Start frequency in Hz [default: 200].
  --high=f2   End frequency in Hz [default: 10000].
  --tail      Apply a tail
"""
import numpy as np
import scipy.signal
from docopt import docopt
import librosa

# function [sweep invsweepfft sweepRate] = synthSweep(T,FS,f1,f2,tail,magSpect)

# https://uk.mathworks.com/matlabcentral/fileexchange/29187-swept-sine-analysis
# % SYNTHSWEEP Synthesize a logarithmic sine sweep.
# %   [sweep invsweepfft sweepRate] = SYNTHSWEEP(T,FS,f1,f2,tail,magSpect)
# %   generates a logarithmic sine sweep that starts at frequency f1 (Hz),
# %   stops at frequency f2 (Hz) and duration T (sec) at sample rate FS (Hz).
# %
# %   usePlots indicates whether to show frequency characteristics of the
# %   sweep, and the optional magSpect is a vector of length T*FS+1 that
# %   indicates an artificial spectral shape for the sweep to have
# %
# %   Developed at Oygo Sound LLC
# %
# %   Equations from Muller and Massarani, "Transfer Function Measurement
# %   with Sweeps."


def grpdelay2phase(grd):
    """Group delay to phase conversion."""
    ph = -np.cumsum(grd)
    ph = 2 * np.pi * ph / len(grd)
    return ph


def synth_sweep(T=10.0, FS=48000.0, f1=200.0, f2=10000.0, tail=0):
    """Synthesie a log sine sweep."""
    # %%% number of samples / frequency bins
    # N = real(round(T*FS));
    N = T * FS

    # %%% make sure start frequency fits in the first fft bin
    # f1 = ceil( max(f1, FS/(2*N)) );
    f1 = max(f1, FS / (2 * N))

    # %%% set group delay of sweep's starting freq to one full period length of
    # %%% the starting frequency, or N/200 if thats too small, or N/10 if its
    # %%% big
    # Gd_start = ceil(min(N/10,max(FS/f1, N/200)));
    Gd_start = min(N / 10, max(FS / f1, N / 200))

    # %%% set fadeout length
    # postfade = ceil(min(N/10,max(FS/f2,N/200)));
    postfade = min(N / 10, max(FS / f2, N / 200))

    # %%% find the length of the actual sweep when its between f1 and f2
    # Nsweep = N - tail - Gd_start - postfade;
    Nsweep = N - tail - Gd_start - postfade

    # %%% length in seconds of the actual sweep
    # tsweep = Nsweep/FS;
    tsweep = Nsweep / FS

    # # sweepRate = log2(f2/f1)/tsweep;
    # sweepRate = np.log2(f2 / f1) / tsweep

    # %%% make a frequency vector for calcs (This  has length N+1) )
    # f = ([0:N]*FS)/(2*N);
    f = np.linspace(0, N, int(N) + 1) * FS / (2 * N)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%             CALCULATE DESIRED MAGNITUDE
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # %%% create pink (-10dB per decade, or 1/(sqrt(f)) spectrum
    # mag = [sqrt(f1./f(1:end))];
    # mag(1) = mag(2);
    f[0] = f[1]
    mag = np.sqrt(f1 / f)

    # %%% Create band pass magnitude to start and stop at desired frequencies
    # [B1 A1] = butter(2,f1/(FS/2),'high' );  %%% HP at f1
    # [B2 A2] = butter(2,f2/(FS/2));          %%% LP at f2
    B1, A1 = scipy.signal.butter(2, f1 / (FS / 2), 'highpass')
    B2, A2 = scipy.signal.butter(2, f2 / (FS / 2), 'lowpass')

    # %%% convert filters to freq domain
    # [H1 W1] = freqz(B1,A1,N+1,FS);
    # [H2 W2] = freqz(B2,A2,N+1,FS);
    W1, H1 = scipy.signal.freqz(B1, A1, worN=int(N + 1))
    W2, H2 = scipy.signal.freqz(B2, A2, worN=int(N + 1))
    W1 = FS * W1 / (2 * np.pi)
    W2 = FS * W2 / (2 * np.pi)

    # %%% multiply mags to get final desired mag spectrum
    # mag = mag.*abs(H1)'.*abs(H2)';
    mag = mag * np.abs(H1) * np.abs(H2)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%            CALCULATE DESIRED GROUP DELAY
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # % calc group delay for arbitrary mag spectrum with contant time envelope
    # % from Muller eq's 11 and 12
    # C = tsweep ./ sum(mag.^2);
    # Gd = C * cumsum(mag.^2);
    # Gd = Gd + Gd_start/FS; % add predelay
    # Gd = Gd*FS/2;   % convert from secs to samps
    C = tsweep / sum(np.square(mag))
    Gd = C * np.cumsum(np.square(mag))
    Gd += Gd_start / FS  # add predelay
    Gd = Gd * FS / 2   # convert from secs to samps

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%            CALCULATE DESIRED PHASE
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # %%% integrate group delay to get phase
    # ph = grpdelay2phase(Gd);
    ph = grpdelay2phase(Gd)

    # %%% force the phase at FS/2 to be a multiple of 2pi using Muller eq 10
    # %%% (but ending with mod 2pi instead of zero ...)
    ph = ph - (f / (FS / 2)) * np.mod(ph[-1], 2 * np.pi)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%             SYNTHESIZE COMPLEX FREQUENCY RESPONSE
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # cplx = mag.*exp(sqrt(-1)*ph); %%% put mag and phase together in polar for
    # cplx = [cplx conj(fliplr(cplx(2:end-1)))]; %%% conjugate, flip, ...
    #  put mag and phase together in polar form
    cplx = mag * np.exp(np.sqrt(-1 + 0j) * ph)
    # conjugate, flip, append for whole spectrum
    cplx = np.append(cplx, np.flip(np.conj(cplx[1:-1]), 0))

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%             EXTRACT IMPULSE RESPONSE WITH IFFT AND WINDOW
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # ir = real(ifft(cplx));
    # err = max(abs(imag(ifft(cplx))));  %%% if this is not really ...
    ir = np.real(scipy.fftpack.ifft(cplx))
    # # if this is not really tiny then something is wrong
    # err = np.max(np.abs(np.imag(scipy.fftpack.ifft(cplx))))

    # %%% create window for fade-in and apply
    # w = hann(2*Gd_start)';
    # I = 1:Gd_start;
    # ir(I) = ir(I).*w(I);
    w = scipy.signal.hann(2 * Gd_start)
    r = range(int(Gd_start))
    ir[r] = ir[r] * w[r]

    # %%% create window for fade-out and apply
    # w = hann(2*postfade)';
    # I = Gd_start+Nsweep+1:Gd_start+Nsweep+postfade;
    # ir(I) = ir(I).*w(postfade+1:end);
    w = scipy.signal.hann(2 * postfade)
    I = range(int(Gd_start + Nsweep + 1), int(Gd_start + Nsweep + postfade))
    ir[I] = ir[I] * w[int(postfade + 1):]

    # %%% force the tail beyond the fadeout to zeros
    # I = Gd_start+Nsweep+postfade+1:length(ir);
    # ir(I) = zeros(1,length(I));
    I = range(int(Gd_start + Nsweep + postfade + 1), len(ir))
    ir[I] = np.zeros(len(I))

    # %%% cut the sweep down to its correct size
    # ir = ir(1:end/2);
    ir = ir[0:len(ir) / 2]

    # %%% normalize
    # ir = ir/(max(abs(ir(:))));
    ir = ir / max(np.abs(ir))

    # %%% get fft of sweep to verify that its okay and to use for inverse
    # irfft = fft(ir);
    irfft = scipy.fftpack.fft(ir)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%             CREATE INVERSE SPECTRUM
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # %%% start with the true inverse of the sweep fft
    # %%% this includes the band-pass filtering, whos inverse could go to
    # %%% infinity!!!
    # invirfft = 1./irfft;
    invirfft = 1.0 / irfft

    # %%% so we need to re-apply the band pass here to get rid of that
    # [H1 W1] = freqz(B1,A1,length(irfft),FS,'whole');
    # [H2 W2] = freqz(B2,A2,length(irfft),FS,'whole');
    W1, H1 = scipy.signal.freqz(B1, A1, len(irfft), whole=True)
    W2, H2 = scipy.signal.freqz(B2, A2, len(irfft), whole=True)

    W1 = FS * W1 / (2 * np.pi)
    W2 = FS * W2 / (2 * np.pi)

    # %%% apply band pass filter to inverse magnitude
    # invirfftmag  = abs(invirfft).*abs(H1)'.*abs(H2)';
    invirfftmag = np.abs(invirfft) * np.abs(H1) * np.abs(H2)

    # %%% get inverse phase
    # invirfftphase = angle(invirfft);
    invirfftphase = np.angle(invirfft)

    # %%% re-synthesis inverse fft in polar form
    # invirfft = invirfftmag.*exp(sqrt(-1)*invirfftphase);
    invirfft = invirfftmag * np.exp(np.sqrt(-1 + 0j) * invirfftphase)

    # %%% assign outputs
    # invsweepfft = invirfft;
    # sweep = ir;
    invsweepfft = invirfft
    sweep = ir

    return sweep, invsweepfft


def main():
    """Main method called from commandline."""
    arguments = docopt(__doc__)
    t = float(arguments['--T'])
    fs = float(arguments['--FS'])
    f1 = float(arguments['--low'])
    f2 = float(arguments['--high'])
    tail = arguments['--tail']
    filenm = arguments['FILENM']

    sweep, invsweepfft = synth_sweep(T=t, FS=fs, f1=f1, f2=f2, tail=tail)

    librosa.output.write_wav(filenm + '.wav', sweep, fs, norm=False)

    np.save(filenm + '.inv.npy', invsweepfft)


if __name__ == '__main__':
    main()
