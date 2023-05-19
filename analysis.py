import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random

files = os.listdir("data")
fileNames = np.char.strip(files, files[0][:9])

while True:
    userIn = input(
        f"Which would you like to look at? Type in a list of integers between 1 and {len(fileNames)}, separated by spaces: "
    )
    try:
        graphNums = [int(x) for x in userIn.split()]
        if any(np.array(graphNums) > len(fileNames)):
            continue
        break
    except ValueError:
        continue

for num in graphNums:
    df = pd.read_table("data\\" + files[num - 1])
    hdrTxt = df.columns[0]  # get the header
    sampleFreq = 1 / float(hdrTxt.split()[-1])
    startTime = np.datetime64(hdrTxt.split()[4].replace("/", "-").replace(",", "T"))
    locCode = hdrTxt.split()[1:4]

    x = df[hdrTxt].to_numpy()

    if len(x) % 2:
        x = x[:-1]  # cut last element if odd

    dftObj = np.fft.fft(x)

    numSamples = len(x)  # number of samples
    duration = (numSamples - 1) / sampleFreq

    ## 2 sided
    mags2S = np.abs(dftObj / numSamples)  # amplitude of each singal
    phases2S = np.angle(dftObj) / np.pi  # phase shift
    freqs2S = np.arange(-numSamples // 2, numSamples // 2) / numSamples * sampleFreq

    ## 1 sided
    mags1S = 2 * mags2S[: numSamples // 2]  # amplitude of each singal
    mags1S[0] = mags1S[0] / 2
    phases1S = phases2S[: numSamples // 2]  # phase shift
    freqs1S = np.arange(numSamples // 2) / duration
    # frequency (hz)

    plt.style.use(["dark_background"])
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.color": [0.25, 0.25, 0.25],
            "lines.linewidth": 0.75,
            "lines.markersize": 3,
        }
    )

    fig, axs = plt.subplots(
        2, 1, num=("Reading from " + str(num) + ": " + " ".join(locCode))
    )

    axs[0].set_title("Raw Seismic Data")
    axs[0].plot(np.arange(numSamples) / sampleFreq, x, "-c")
    axs[0].set_ylabel("Amplitude ($x$)")
    axs[0].set_xlabel("Time ($t$) [$s$]")

    axs[1].set_title("One-Sided Fourier Transform")
    axs[1].plot(freqs1S, mags1S, "-c")
    axs[1].set_ylabel("Amplitude ($|X|$)")
    axs[1].set_xlabel("Frequency (f) [$s^{-1}$]")

    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()

plt.show()
