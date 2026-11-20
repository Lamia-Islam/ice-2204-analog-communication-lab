"""Analog communication lab: AM, DSB-SC, SSB, FM, demodulation, spectra, bandwidth and noise."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import signal
OUT = os.path.join(os.path.dirname(__file__), "..", "results"); lines = []
def say(s=""): print(s); lines.append(s)
fs = 100000; t = np.arange(0, 0.1, 1 / fs); fm, fc = 500, 10000; m = np.cos(2 * np.pi * fm * t)
def spec(x, n=2 ** 16): X = abs(np.fft.rfft(x * np.hanning(len(x)), n)) / (len(x) / 4); return np.fft.rfftfreq(n, 1 / fs), X
def peaks(x, thr=.08):
    f, X = spec(x); idx = [i for i in range(1, len(X) - 1) if X[i] > X[i-1] and X[i] > X[i+1] and X[i] > thr * X.max()]; return [(int(round(f[i])), float(round(X[i] / X.max(), 2))) for i in idx]
lp = lambda x, c: signal.filtfilt(*signal.butter(5, c, fs=fs), x)
fig, ax = plt.subplots(3, 3, figsize=(15, 11)); ax = ax.ravel()

say("== Amplitude modulation (message 500 Hz, carrier 10 kHz) ==")
for mu in (0.5, 1.0, 1.5):
    am = (1 + mu * m) * np.cos(2 * np.pi * fc * t); env = np.abs(signal.hilbert(am)); rec = lp(env, 1500); rec -= rec.mean(); rec /= (mu if mu else 1)
    err = np.sqrt(np.mean((rec[2000:-2000] - m[2000:-2000]) ** 2)); eff = mu ** 2 / (2 + mu ** 2)
    say("mu = %.1f: power efficiency %.1f%%; envelope detector error (RMS vs message) %.3f%s" % (mu, 100 * eff, err, "  <- over-modulated, envelope distorted" if mu > 1 else ""))
am = (1 + 0.5 * m) * np.cos(2 * np.pi * fc * t); ax[0].plot(t[:600], am[:600], "k", label="AM, mu=0.5"); ax[0].plot(t[:600], 1 + .5 * m[:600], "C7", label="envelope"); ax[0].legend(loc="upper right"); ax[0].set_title("AM waveform")
f, X = spec(am); k_ = (f > 8000) & (f < 12000); ax[1].plot(f[k_], X[k_], "k"); ax[1].set_title("AM spectrum: carrier and two sidebands"); ax[1].set_xlabel("Hz")
say("AM spectrum peaks (Hz, relative amplitude): %s; bandwidth = 2 x fm = %d Hz" % (peaks(am), 2 * fm))

say("\n== DSB-SC and coherent detection ==")
dsb = m * np.cos(2 * np.pi * fc * t); rec = lp(dsb * np.cos(2 * np.pi * fc * t), 1500) * 2; say("DSB-SC peaks: %s (no carrier line); coherent demodulation error %.4f" % (peaks(dsb), np.sqrt(np.mean((rec[2000:-2000] - m[2000:-2000]) ** 2))))
for ph in (30, 60, 90):
    r = lp(dsb * np.cos(2 * np.pi * fc * t + np.radians(ph)), 1500) * 2; say("  local oscillator phase error %2d deg: recovered amplitude %.3f (theory cos(phase) = %.3f)" % (ph, np.max(np.abs(r[2000:-2000])), np.cos(np.radians(ph))))
ax[2].plot(t[:600], dsb[:600], "k"); ax[2].set_title("DSB-SC waveform (phase reversals)")

say("\n== SSB ==")
ssb = m * np.cos(2 * np.pi * fc * t) - np.imag(signal.hilbert(m)) * np.sin(2 * np.pi * fc * t); say("Upper sideband only: peaks %s, bandwidth = fm = %d Hz, half that of DSB" % (peaks(ssb), fm))
f, X = spec(ssb); k_ = (f > 8000) & (f < 12000); ax[3].plot(f[k_], X[k_], "k"); ax[3].set_title("SSB (upper sideband) spectrum"); ax[3].set_xlabel("Hz")

say("\n== Frequency modulation ==")
kf = 2500; phase = 2 * np.pi * kf * np.cumsum(m) / fs; fmod = np.cos(2 * np.pi * fc * t + phase); beta = kf / fm
say("Frequency deviation %d Hz, modulation index beta = %.1f; Carson bandwidth = 2(df + fm) = %d Hz" % (kf, beta, 2 * (kf + fm)))
f, X = spec(fmod); sel = X > 0.1 * X.max(); say("Measured: lines above 10%% of the peak span %.0f Hz to %.0f Hz (width %.0f Hz). Carson's rule is an approximation that captures about 98%% of the power, so the two need not match" % (f[sel].min(), f[sel].max(), f[sel].max() - f[sel].min()))
k_ = (f > 6000) & (f < 14000); ax[4].plot(f[k_], X[k_], "k"); ax[4].set_title("FM spectrum, beta = 5"); ax[4].set_xlabel("Hz")
dem = np.diff(np.unwrap(np.angle(signal.hilbert(fmod)))) * fs / (2 * np.pi) - fc; rec = lp(dem / kf, 1500); say("FM demodulation by phase differentiation (noise-free): error %.4f" % np.sqrt(np.mean((rec[2000:-2000] - m[1:][2000:-2000]) ** 2)))
def carrier_line(kf_):
    y = np.cos(2 * np.pi * fc * t + 2 * np.pi * kf_ * np.cumsum(m) / fs); f_, X_ = spec(y); return X_[abs(f_ - fc) < 100].max()
from scipy.special import jv
say("Carrier line amplitude against Bessel J0(beta), carrier amplitude 1 for an unmodulated carrier:")
for b_ in (0.5, 1.0, 2.4048, 5.0): say("  beta = %.4f: measured %.3f, |J0| = %.3f" % (b_, carrier_line(b_ * fm) / carrier_line(0.001), abs(jv(0, b_))))
say("  (beta = 2.4048 is the first zero of J0: the carrier line nearly vanishes)")

say("\n== Noise: output SNR of AM versus FM ==")
rng = np.random.default_rng(0)
def snr_out(ref, x): return 10 * np.log10(np.sum(ref ** 2) / np.sum((x - ref) ** 2))
res = {}
for snr_db in (5, 10, 15, 20):
    nv = np.sqrt(0.5 / 10 ** (snr_db / 10)); a = am + nv * rng.standard_normal(len(am)); rec_a = lp(np.abs(signal.hilbert(a)), 1500); rec_a = (rec_a - rec_a.mean()) / 0.5
    f_ = fmod + nv * rng.standard_normal(len(fmod)); d = np.diff(np.unwrap(np.angle(signal.hilbert(f_)))) * fs / (2 * np.pi) - fc; rec_f = lp(d / kf, 1500)
    res[snr_db] = (snr_out(m[2000:-2000], rec_a[2000:-2000]), snr_out(m[1:][2000:-2000], rec_f[2000:-2000]))
    say("input SNR %2d dB: AM output SNR %5.1f dB, FM (beta=5) output SNR %5.1f dB" % (snr_db, *res[snr_db]))
ax[5].plot(list(res), [v[0] for v in res.values()], "o-", c="C7", label="AM"); ax[5].plot(list(res), [v[1] for v in res.values()], "ks-", label="FM"); ax[5].set_xlabel("input SNR (dB)"); ax[5].set_ylabel("output SNR (dB)"); ax[5].legend(); ax[5].grid(alpha=.4); ax[5].set_title("Noise performance")
fa = np.linspace(0, 1, 300); ax[6].plot(fa, np.sqrt(np.maximum(0, 1 - fa)) * 0 + fa, "k"); ax[6].set_title("AM efficiency vs modulation index (mu^2/(2+mu^2))"); ax[6].plot(np.linspace(0, 1.5, 100), np.linspace(0, 1.5, 100) ** 2 / (2 + np.linspace(0, 1.5, 100) ** 2), "k", label="efficiency"); ax[6].cla(); mu_ = np.linspace(0, 1.5, 100); ax[6].plot(mu_, mu_ ** 2 / (2 + mu_ ** 2), "k"); ax[6].axvline(1, c="C7", ls="--"); ax[6].set_xlabel("mu"); ax[6].set_title("AM power efficiency (max 33% at mu = 1)")
ax[7].plot(t[:800], fmod[:800], "k"); ax[7].set_title("FM waveform"); ax[8].axis("off")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "analog_communication.png"), dpi=120)
open(os.path.join(OUT, "analog_communication_output.txt"), "w").write("\n".join(lines) + "\n")
