#%%
import glob
import madmom
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy
import time
import os

#%%
dcp = madmom.audio.chroma.DeepChromaProcessor()

time_b, time_e = 0, 0

wav_file = 'dataset/pop/despacito.wav'
# wav_file = 'dataset/classical/Hungarian Dance No. 5.wav'
# wav_file = 'dataset/classical/La campanella.wav'

print('-'*60)
print(wav_file)
y, sr = librosa.load(wav_file, sr=None, mono=True)

#%%
#### Chromagram ####
time_b      = time.time()
chroma      = dcp(wav_file)
time_e      = time.time()
print(f'Chromagram done. {(time_e-time_b):.2f} seconds passed.')

time_b      = time.time()
## power chroma ##
# chroma = np.square(chroma)

## norm of chroma ##
nrow, ncol  = chroma.shape
resolution  = len(y) / nrow / sr
chroma_norm = np.zeros(nrow)
for i in range(nrow):
    chroma_norm[i] = np.linalg.norm(chroma[i,:], ord=2)

time_e      = time.time()
print(f'Chromagram norm done. {(time_e-time_b):.2f} seconds passed.')


#%%
#### SSM ####
time_b = time.time()
ssm    = np.zeros(shape=(nrow, nrow))
for i in range(nrow):
    v1 = chroma[i,:]
    v1_norm = chroma_norm[i]
    for j in range(nrow):
        if ssm[i,j] != 0:
            continue
        else:
            v2 = chroma[j,:]
            v2_norm = chroma_norm[j]
            dot = np.sum(v1 * v2) # element-wise product
            cos = dot / (v1_norm * v2_norm)
            ssm[i,j] = cos
            ssm[j,i] = cos

time_e = time.time()
print(f'SSM done. {(time_e-time_b):.2f} seconds passed.')

#%%
#### normalize SSM ####
ssm_normalized = (ssm - float(np.min(ssm)) / (float(np.max(ssm))) - float(np.min(ssm)))

#### filter SSM by medium filter ####
time_b = time.time()
# ssm_smooth = ssm_normalized
ssm_smooth = scipy.signal.medfilt2d(ssm_normalized, kernel_size=5)
time_e = time.time()
print(f'SSM filter done. {(time_e-time_b):.2f} seconds passed.')

#%%
#### Flip up-side-down and plot ####
ssm_flipped = np.flipud(ssm_smooth)
# ssm_flipped = ssm_flipped[-300:, 0:300] # plot first 1000 frames
plt_nrow, plt_ncol = ssm_flipped.shape

plt.figure(figsize=(15,15))
plt.imshow(ssm_flipped, cmap='plasma', interpolation='none')
MINSEC_FORMATTER     = lambda val, pos: '{0:d}:{1:02d}'.format(int(val*resolution/60), int((val*resolution)%60))
INV_MINSEC_FORMATTER = lambda val, pos: '{0:d}:{1:02d}'.format(int((plt_nrow-val-1)*resolution/60), int(((plt_nrow-val-1)*resolution)%60))
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(MINSEC_FORMATTER))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(INV_MINSEC_FORMATTER))
plt.xticks(np.arange(0, plt_nrow, step=int(plt_nrow/28)))
plt.yticks(np.arange(0, plt_nrow, step=int(plt_nrow/28)))
# plt.grid(axis='x')
# plt.show()
plt.savefig(f'{os.path.basename(wav_file)}-ssm.pdf')

#%%
## NOTE maximal phrase size ~ 30 frames ##
half_filter_size = 20 # user-defined parameter
quad_shape       = (half_filter_size, half_filter_size)
half_filter      = np.hstack((np.zeros(shape=quad_shape), np.ones(shape=quad_shape)))
phrase_filter    = np.vstack((half_filter, np.fliplr(half_filter)))

## Scan diagonal of SSM ##
filter_size = len(phrase_filter)
energy_arr  = []
for i in range(0, len(ssm_smooth)):
    try:
        sub_matrix = ssm_smooth[i:(i+filter_size), i:(i+filter_size)]
        energy_arr.append(float(np.sum(np.abs(sub_matrix - phrase_filter))))
    except (IndexError, ValueError) as e:
        break

energy_arr = np.array(energy_arr) / float(np.max(energy_arr))
energy_arr = np.concatenate(([np.min(energy_arr)]*half_filter_size, energy_arr))

## smoothen ##
smoothed_energy = np.array(energy_arr)
# smoothed_energy = scipy.signal.savgol_filter(energy_arr, window_length=7, polyorder=2)
# smoothed_energy = smoothed_energy - half_filter_size

## Plot smoothened energy ##
plt.figure(figsize=(15,3))
plt.plot(range(0,len(energy_arr[:])), energy_arr[:])
MINSEC_FORMATTER     = lambda val, pos: '{0:d}:{1:02d}'.format(int(val*resolution/60), int((val*resolution)%60))
# plt.plot(range(0,len(energy_arr[:int(200)])), energy_arr[:int(200)])
# MINSEC_FORMATTER     = lambda val, pos: '{0:d}:{1:02d}'.format(int(val*resolution/60), int((val*resolution)%60))
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(MINSEC_FORMATTER))
plt.xticks(np.arange(0, len(energy_arr[:int(200)]), step=int(len(energy_arr[:int(200)])/25)))
# plt.show()
plt.savefig(f'{os.path.basename(wav_file)}-energy.pdf')


#%%
## Find energy peaks ##
peaks, _ = scipy.signal.find_peaks(smoothed_energy, height=None, threshold=None, distance=20)
peaks    = peaks[1:] # discard the first peak

## Discard those less than a specific percentile ##
threshold = np.percentile(smoothed_energy[peaks], 15)
peaks = peaks[smoothed_energy[peaks]>=threshold]

## Frame to time ##
peaks_frame = peaks
peaks_time  = peaks_frame  * resolution

clicks = librosa.clicks(times=peaks_time, sr=sr, length=len(y))
normal_clicks = (clicks - min(clicks)) / 2.0
scaled_clicks = normal_clicks * (max(y) - min(y)) + min(y)
WAV = y * 0.8 + scaled_clicks * 1.2
librosa.output.write_wav(f'annotated-{os.path.basename(wav_file)}', WAV, sr)

## print peak time and energy ##

with open(f'{os.path.basename(wav_file)}.txt', mode='w') as f:
    lines = []
    for i, ptime in enumerate(peaks_time):
        minute = int(ptime/60)
        second = float(ptime%60)
        lines.append(f'{minute: 2d}:{second:04.1f} | {smoothed_energy[peaks_frame[i]]:.2f}\n')
        print(f'{minute: 2d}:{second:04.1f} | {smoothed_energy[peaks_frame[i]]:.2f}')

    f.writelines(lines)


#%%

fig, ax = plt.subplots(figsize=(15,3))
# ax.set_facecolor('w')
# fig.set_facecolor('tab:gray')

ax.plot(range(0,len(energy_arr[:])), energy_arr[:])
ax.plot(peaks, smoothed_energy[peaks], ls="", marker="x", ms=10,  color="crimson")


MINSEC_FORMATTER = lambda val, pos: '{0:d}:{1:02d}'.format(int(val*resolution/60), int((val*resolution)%60))
fig.gca().xaxis.set_major_formatter(ticker.FuncFormatter(MINSEC_FORMATTER))
ax.set_xticks(np.arange(0, len(energy_arr[:]), step=int(len(energy_arr[:])/30)))

# plt.show()
plt.savefig(f'{os.path.basename(wav_file)}-energy.pdf')



plt.figure(figsize=(15,3))
librosa.display.specshow(chroma.T, y_axis='chroma')
plt.savefig(f'{os.path.basename(wav_file)}-chroma.pdf')









# onset_env = librosa.onset.onset_strength(y=y, sr=sr,
#                                          feature=librosa.cqt)
# times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)

#%%
# plt.figure(figsize=(15,3))
# plt.plot(times[:6280], onset_env[:6280] / onset_env.max(), alpha=0.8,
#          label='Mean (CQT)')
# plt.legend(frameon=True, framealpha=0.75)
# plt.ylabel('Normalized strength')
# plt.yticks([])
# plt.xticks(np.arange(0, 65, step=2))
# plt.axis('tight')
# plt.tight_layout()

#%%
