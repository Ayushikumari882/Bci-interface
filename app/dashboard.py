import streamlit as st
import matplotlib.pyplot as plt
from preprocessing import load_and_preprocess
from feature_extraction import extract_features, compute_band_power
from classifier import train_and_evaluate
import numpy as np
import mne

st.set_page_config(page_title="NeuroSense", page_icon="🧠", layout="wide")

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #000000 0%, #000033 50%, #000066 100%);
    color: white;
    font-family: 'Arial', sans-serif;
}
.sidebar .sidebar-content {
    background-color: rgba(0, 0, 51, 0.9);
    color: white;
}
.stButton>button {
    background-color: #007bff;
    color: white;
    border-radius: 5px;
    border: none;
}
.stButton>button:hover {
    background-color: #0056b3;
}
.stTextInput, .stNumberInput, .stSelectbox {
    background-color: rgba(255, 255, 255, 0.1);
    color: white;
}
.stHeader, .stSubheader {
    color: white;
}
</style>
"", unsafe_allow_html=True)

st.sidebar.title("NeuroSense EEG System")

# Sidebar
if st.sidebar.button("Download & Load Dataset"):
    with st.spinner("Downloading and loading dataset..."):
        X, y, epochs, info = load_and_preprocess()
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.epochs = epochs
        st.session_state.info = info
        st.session_state.data_loaded = True
        st.success("Dataset loaded!")

uploaded_file = st.sidebar.file_uploader("Upload EDF File", type=['edf'])
if uploaded_file is not None:
    st.sidebar.write("EDF upload not implemented yet.")

if st.sidebar.button("Generate Synthetic Data"):
    st.sidebar.write("Synthetic data generation not implemented yet.")

if 'data_loaded' in st.session_state:
    st.sidebar.subheader("Dataset Info")
    st.sidebar.write(f"Subject: {st.session_state.info['subject']}")
    st.sidebar.write(f"Epochs: {st.session_state.info['n_epochs']}")
    st.sidebar.write(f"Channels: {st.session_state.info['n_channels']}")
    st.sidebar.write(f"Sampling Rate: {st.session_state.info['sfreq']} Hz")

    if st.sidebar.button("Run Classification"):
        with st.spinner("Running classification..."):
            X_feat = extract_features(st.session_state.X, st.session_state.y)
            clf, acc, y_test, y_pred, y_prob, cm, cv_mean = train_and_evaluate(X_feat, st.session_state.y)
            st.session_state.clf = clf
            st.session_state.acc = acc
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            st.session_state.y_prob = y_prob
            st.session_state.cm = cm
            st.session_state.cv_mean = cv_mean
            st.session_state.model_trained = True
            st.success("Classification complete!")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("EEG Signal Monitor")
    if 'data_loaded' in st.session_state:
        n_channels_to_plot = min(8, st.session_state.info['n_channels'])
        fig, axes = plt.subplots(n_channels_to_plot, 1, figsize=(12, 8), sharex=True)
        fig.patch.set_facecolor('#000000')
        for ax in axes:
            ax.set_facecolor('#000000')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
        time = np.arange(st.session_state.X.shape[2]) / st.session_state.info['sfreq']
        for i in range(n_channels_to_plot):
            axes[i].plot(time, st.session_state.X[0, i, :], color='cyan')
            axes[i].set_ylabel(st.session_state.info['ch_names'][i], color='white')
        axes[-1].set_xlabel('Time (s)', color='white')
        st.pyplot(fig)

    if 'model_trained' in st.session_state:
        st.subheader("Feature Visualization")
        st.write("Spectrogram")
        fig_spec, ax_spec = plt.subplots()
        ax_spec.set_facecolor('#000000')
        psds, freqs = mne.time_frequency.psd_welch(st.session_state.epochs, fmax=40)
        ax_spec.semilogy(freqs, np.mean(psds, axis=(0,1)), color='cyan')
        ax_spec.set_xlabel('Frequency (Hz)', color='white')
        ax_spec.set_ylabel('Power Spectral Density', color='white')
        ax_spec.tick_params(colors='white')
        ax_spec.spines['bottom'].set_color('white')
        ax_spec.spines['top'].set_color('white')
        ax_spec.spines['right'].set_color('white')
        ax_spec.spines['left'].set_color('white')
        st.pyplot(fig_spec)
        
        st.write("Band Power")
        band_powers, freqs = compute_band_power(st.session_state.epochs)
        fig_bp, ax_bp = plt.subplots()
        ax_bp.set_facecolor('#000000')
        ax_bp.bar(['Mu (8-12 Hz)', 'Beta (12-30 Hz)'], [np.mean(band_powers['mu']), np.mean(band_powers['beta'])], color='blue')
        ax_bp.set_ylabel('Power', color='white')
        ax_bp.tick_params(colors='white')
        ax_bp.spines['bottom'].set_color('white')
        ax_bp.spines['top'].set_color('white')
        ax_bp.spines['right'].set_color('white')
        ax_bp.spines['left'].set_color('white')
        st.pyplot(fig_bp)
        
        st.write("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        ax_cm.set_facecolor('#000000')
        im = ax_cm.imshow(st.session_state.cm, cmap='Blues', interpolation='nearest')
        ax_cm.set_title('Confusion Matrix', color='white')
        ax_cm.set_xlabel('Predicted', color='white')
        ax_cm.set_ylabel('True', color='white')
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(['Left Hand', 'Right Hand'])
        ax_cm.set_yticklabels(['Left Hand', 'Right Hand'])
        ax_cm.tick_params(colors='white')
        for i in range(2):
            for j in range(2):
                text = ax_cm.text(j, i, st.session_state.cm[i, j], ha="center", va="center", color="white")
        plt.colorbar(im, ax=ax_cm)
        st.pyplot(fig_cm)

with col2:
    st.header("Prediction Panel")
    if 'model_trained' in st.session_state:
        idx = 0
        pred = st.session_state.y_pred[idx]
        conf = np.max(st.session_state.y_prob[idx])
        class_name = 'Left Hand' if pred == 1 else 'Right Hand'
        st.write(f"**Predicted Class:** {class_name}")
        st.write(f"**Confidence Score:** {conf:.2f}")
        st.write(f"**Model Accuracy:** {st.session_state.acc:.2f}")
        st.write(f"**Cross-validation Score:** {st.session_state.cv_mean:.2f}")
    
    if 'model_trained' in st.session_state:
        st.subheader("Final Result")
        st.write(f"**Predicted Class:** {class_name}")
        st.write(f"**Confidence Score:** {conf:.2f}%")
        st.write(f"**Accuracy:** {st.session_state.acc:.2f}%")
        st.write(f"**Cross-validation:** {st.session_state.cv_mean:.2f}%")