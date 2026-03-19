import mne
from mne.datasets import eegbci
import numpy as np

def load_and_preprocess(subject=1, runs=[4, 8, 12]):
    raw_fnames = eegbci.load_data(subject=subject, runs=runs, path='./data')
    raw = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in raw_fnames])
    raw.filter(8, 30, fir_design='firwin')
    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id, tmin=1, tmax=2, baseline=None, preload=True)
    X = epochs.get_data()
    y = epochs.events[:, 2]
    info = {
        'subject': subject,
        'n_epochs': len(epochs),
        'n_channels': raw.info['nchan'],
        'sfreq': raw.info['sfreq'],
        'ch_names': raw.ch_names
    }
    return X, y, epochs, info
