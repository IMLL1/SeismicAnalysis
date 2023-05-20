import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from plotly.subplots import make_subplots
import plotly.graph_objs as go

pd.options.plotting.backend = "plotly"

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
    df = df.rename(columns={df.columns[0]: "x"})

    sampleFreq = 1 / float(hdrTxt.split()[-1])
    startTime = np.datetime64(hdrTxt.split()[4].replace("/", "-").replace(",", "T"))
    locCode = hdrTxt.split()[1:4]

    if len(df) % 2:
        df = df[:-1]  # cut last element if odd

    dftObj = np.fft.fft(df.x)

    numSamples = len(df)  # number of samples
    duration = (numSamples - 1) / sampleFreq
    df["t"] = np.arange(numSamples) / sampleFreq

    ## 2 sided
    mags2S = np.abs(dftObj / numSamples)  # amplitude of each singal
    phases2S = np.angle(dftObj) / np.pi  # phase shift
    freqs2S = np.arange(-numSamples // 2, numSamples // 2) / numSamples * sampleFreq

    ## 1 sided
    mags1S = 2 * mags2S[: numSamples // 2]  # amplitude of each singal
    mags1S[0] = mags1S[0] / 2
    phases1S = phases2S[: numSamples // 2]  # phase shift
    freqs1S = np.arange(numSamples // 2) / duration

    fftDF = pd.DataFrame({"mags": mags1S, "freqs": freqs1S})

    # plt.style.use(["dark_background"])
    # plt.rcParams.update({"axes.grid": True,"grid.color": [0.25, 0.25, 0.25],"lines.linewidth": 0.75,"lines.markersize": 3,})

    # df.plot(grid=True, title="Raw Seismograph Data", xlabel="Time ($t$) [$s$]",ylabel="Reading (x)", legend=[])
    # fftDF.plot(grid=True,title="Fourier Transform",xlabel="Frequency ($f$) [$s^{-1}$]",ylabel="Magnitude ($|X|$)",legend=[])

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Raw Seismograph Data", "Fourier Transform"),
        vertical_spacing=0.1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.t,
            y=df.x,
            # labels={"t": "Time (t) [s]", "x": "Reading Magnitude (x)"}
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            y=fftDF.mags,
            x=fftDF.freqs,
            # labels={"freqs": "Frequency (f) [1/s]", "mags": "Magnitude (|X|)"},
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title_text="Reading from " + str(num) + ": " + " ".join(locCode),
        title_xanchor="left",
        title_xref="container",
        template="plotly_dark",
        #margin=dict(r=10, t=50, b=25, l=10),
        showlegend=False,
    )
    fig.show()

plt.show()
