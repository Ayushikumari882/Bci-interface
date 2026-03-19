from mne.decoding import CSP

def extract_features(X, y, n_components=4):
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    X_csp = csp.fit_transform(X, y)
    return X_csp