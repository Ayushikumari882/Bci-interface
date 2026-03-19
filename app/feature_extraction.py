from mne.decoding import CSP
import numpy as np

def extract_features(X, y, n_components=4):
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    X_csp = csp.fit_transform(X, y)
    return X_csp

def compute_band_power(epochs, bands={'mu': (8, 12), 'beta': (12, 30)}):
    psds, freqs = mne.time_frequency.psd_welch(epochs, fmin=1, fmax=40, n_fft=256)
    band_powers = {}
    for band, (fmin, fmax) in bands.items():
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        band_powers[band] = np.mean(psds[:, :, band_mask], axis=(1, 2))
    return band_powers, freqs