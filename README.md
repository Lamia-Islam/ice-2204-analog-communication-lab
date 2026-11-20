# ICE-2204: Analog Communication Sessional

Lab program built from the course syllabus: amplitude, double-sideband, single-sideband and frequency modulation, demodulation, spectra, bandwidth and noise performance.

These are written from the syllabus topics. They are not my original lab questions, which I no longer have.

## Contents

`python/analog_communication.py` covers:

- AM at three modulation indices, with power efficiency and envelope detection (over-modulation at mu = 1.5 distorts the envelope)
- DSB-SC with coherent detection and the effect of local oscillator phase error (amplitude follows cos of the error)
- SSB generated with a Hilbert transform: half the bandwidth of DSB
- FM with modulation index 5: Carson bandwidth against the measured spectrum, demodulation, and the carrier line against Bessel J0
- Output SNR of AM against FM as input SNR rises from 5 to 20 dB

Outputs: `results/analog_communication_output.txt` and `results/analog_communication.png`.

## Run

```bash
pip install numpy scipy matplotlib
python python/analog_communication.py
```

## Notes and limitations

- Demodulation errors are noise-free, so they are near zero by design. Only the noise section adds noise.
- At 5 dB input SNR, FM is barely better than AM. That is the FM threshold effect, and the large FM advantage only appears above it.
- Carson's rule gives 6000 Hz and the measured 10% width is about 7000 Hz. They measure different things.
- Results come from one random noise seed.
