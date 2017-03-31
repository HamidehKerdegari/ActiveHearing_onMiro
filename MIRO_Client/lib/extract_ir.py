"""extract_ir.

Python translations of Matlab EXTRACTIR.

Usage:
  extract_ir.py SWEEPFILE INVFILE OUTFILE
  extract_ir.py (-h | --help)

Arguments:
  SWEEPFILE  name of wav file storing sweep recording
  INVFILE    name of npy file storing inverse filter
  OUTFILE    name of npy file in which to store impulse response

"""
import numpy as np
import scipy.signal
from docopt import docopt
import librosa

# function [irLin, irNonLin] = extractIR(sweep_response, invsweepfft)

# https://uk.mathworks.com/matlabcentral/fileexchange/29187-swept-sine-analysis
# % EXTRACTIR Extract impulse response from swept-sine response.
# %   [irLin, irNonLin] = extractIR(sweep_response, invsweepfft)
# %   Extracts the impulse response from the swept-sine response.  Use
# %   synthSweep.m first to create the stimulus; then pass it through the
# %   device under test; finally, take the response and process it with the
# %   inverse swept-sine to produce the linear impulse response and
# %   non-linear simplified Volterra diagonals.  The location of each
# %   non-linear order can be calculated with the sweepRate - this will be
# %   implemented as a future revision.
# %
# %   Developed at Oygo Sound LLC
# %
# %   Equations from Muller and Massarani, "Transfer Function Measurement
# %   with Sweeps."


def extract_ir(sweep_response, invsweepfft):
    """Compute the inpulse response from the sweep response."""
    # if(size(sweep_response,1) > 1)
    #     sweep_response = sweep_response';
    # end
    # N = length(invsweepfft);
    # sweepfft = fft(sweep_response,N);
    N = len(invsweepfft)
    sweepfft = scipy.fftpack.fft(sweep_response, N)

    # %%% convolve sweep with inverse sweep (freq domain multiply)

    # ir = real(ifft(invsweepfft.*sweepfft));
    ir = np.real(scipy.fftpack.ifft(invsweepfft * sweepfft))

    # ir = circshift(ir', length(ir)/2);
    ir = np.roll(ir, len(ir) / 2)

    # irLin = ir(end/2+1:end);
    # irNonLin = ir(1:end/2);
    len_ir = len(ir)
    ir_lin = ir[len_ir / 2 + 1:-1]
    ir_non_lin = ir[0:len_ir / 2]

    return ir_lin, ir_non_lin


def main():
    """Main method called from commandline."""
    arguments = docopt(__doc__)
    wavefn = arguments['SWEEPFILE']
    invfn = arguments['INVFILE']
    outfn = arguments['OUTFILE']

    sweep_response, sr = librosa.core.load(wavefn, sr=None)
    invsweepfft = np.load(invfn)
    print(sweep_response.shape, sr)
    print(invsweepfft.shape)
    ir_lin, ir_non_lin = extract_ir(sweep_response, invsweepfft)

    import matplotlib.pyplot as plt

    plt.plot(ir_lin)
    plt.show()

    np.save(outfn, ir_lin)


if __name__ == '__main__':
    main()
