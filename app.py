import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import folium
from streamlit_folium import st_folium
from scipy.signal import butter, filtfilt
from math import radians, cos, sin, asin, sqrt

st.title("Fysiikan loppuprojekti - Miro Lampela")

# Ladataan data
gps_url = "https://raw.githubusercontent.com/mirolampela/streamlit_projekti/refs/heads/main/Location_18122025.csv"
acc_url = "https://raw.githubusercontent.com/mirolampela/streamlit_projekti/refs/heads/main/Linear%20Accelerometer_18122025.csv"
df_gps = pd.read_csv(gps_url)
df_acc = pd.read_csv(acc_url)

# Poistetaan epätarkka mittaus datan alusta (20s)
df_acc = df_acc[df_acc['Time (s)'] > 20].copy()
df_gps = df_gps[df_gps['Time (s)'] > 20].copy()

# Laske parametrit
t_tot = df_acc['Time (s)'].max()
N_data = len(df_acc)
fs = N_data / t_tot
dt = t_tot / N_data

st.write(f"Näytteenottotaajuus: {fs:.1f} Hz | Kesto: {t_tot:.1f} s | Mittauspisteitä: {N_data}")

# Alipäästösuodatin
def butter_lowpass_filter(data, cutoff, nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Suodatetaan Z-kiihtyvyysakseli
data = df_acc['Z (m/s^2)'].values
nyq = fs / 2
cutoff = 3
order = 3
data_filt = butter_lowpass_filter(data, cutoff, nyq, order)
df_acc['Z_filt'] = data_filt

# Suodatetut askeleet
N_filt = len(data_filt)
jaksot = 0
for i in range(N_filt - 1):
    if data_filt[i]/data_filt[i+1] < 0:
        jaksot = jaksot + 0.5
suodatetut_askeleet = round(jaksot)

# Fourier-muunnos
fourier = np.fft.fft(data, N_data)
psd = fourier * np.conj(fourier) / N_data
freq = np.fft.fftfreq(N_data, dt)
L = np.arange(1, int(N_data/2))

# Etsitään dominoiva taajuus
f_max = freq[L][psd[L] == np.max(psd[L])][0]
T = 1/f_max
askeleet_fourier = int(np.round(f_max * t_tot))

# Lasketaan matka Haversinella
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # Haversinen kaava 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371  # Maan säde kilometreinä
    return c * r

# Lasketaan kuljettu matka
distances = [0]

# Lasketaan välimatka havaintopisteiden välillä
for i in range(len(df_gps)-1):
    lon1 = df_gps['Longitude (°)'].iloc[i]
    lon2 = df_gps['Longitude (°)'].iloc[i+1]
    lat1 = df_gps['Latitude (°)'].iloc[i]
    lat2 = df_gps['Latitude (°)'].iloc[i+1]
    distances.append(haversine(lon1,lat1,lon2,lat2))

# Lasketaan kokonaismatka
df_gps['Distance_calc'] = distances
df_gps['total_distance'] = df_gps['Distance_calc'].cumsum()

# Matka metreinä
matka_total = df_gps['total_distance'].iloc[-1] * 1000

# Keskinopeus
keskinopeus = df_gps['Velocity (m/s)'].mean()

# Askelpituus
askelpituus = matka_total / suodatetut_askeleet

# Yhteenveto Streamlitillä tulostettuna
st.header("Tulokset")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Askeleet (suodatettu)", f"{suodatetut_askeleet}")
col2.metric("Askeleet (Fourier)", f"{askeleet_fourier}")
col3.metric("Matka", f"{matka_total:.0f} m")
col4.metric("Keskinopeus", f"{keskinopeus*3.6:.1f} km/h")
col5.metric("Askelpituus", f"{askelpituus:.2f} m")


# Kuvaajat
st.header("Analyysit")

# Kuvaaja 1a: Suodatettu kiihtyvyys
st.subheader("1a. Suodatettu kiihtyvyysdata")
fig1, ax1 = plt.subplots(figsize=(15, 5))
ax1.plot(df_acc['Time (s)'], data_filt, linewidth=0.5)
ax1.set_xlabel('Aika [s]')
ax1.set_ylabel('Kiihtyvyys Z [m/s²]')
st.pyplot(fig1)

# Kuvaaja 1b: Suodatettu kiihtyvyys, mittauspisteistä joka 50.
st.subheader("1b. Suodatettu kiihtyvyysdata, mittauspisteistä joka 50.")
df_plot = df_acc[['Time (s)', 'Z_filt']].iloc[::50].copy()
st.line_chart(df_plot, x='Time (s)', y='Z_filt')

# Kuvaaja 2: Tehospektri (ei toiminut st.line_chart)
st.subheader("2. Tehospektritiheys")
fig2, ax2 = plt.subplots(figsize=(15, 5))
ax2.plot(freq[L], psd[L].real)
ax2.set_xlabel('Taajuus [Hz]')
ax2.set_ylabel('Teho')
ax2.set_xlim(0, 10)
st.pyplot(fig2)

# Kuvaaja 3: Kartta
st.subheader("3. Reitti kartalla")

start_lat = df_gps['Latitude (°)'].mean()
start_long = df_gps['Longitude (°)'].mean()
map = folium.Map(location=[start_lat, start_long], zoom_start=14)

# Piirretään reitti
folium.PolyLine(df_gps[['Latitude (°)', 'Longitude (°)']], 
                color='blue', weight=3.5, opacity=1).add_to(map)

# Näytä kartta
st_map = st_folium(map, width=900, height=650)