"""Monkey-patch torchaudio to use soundfile instead of torchcodec (no FFmpeg needed)."""
import torch
import soundfile as sf
import torchaudio


def _load_soundfile(filepath, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None):
    data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
    if frame_offset > 0:
        data = data[frame_offset:]
    if num_frames > 0:
        data = data[:num_frames]
    tensor = torch.from_numpy(data).T if channels_first else torch.from_numpy(data)
    return tensor, sr


def _save_soundfile(filepath, src, sample_rate, channels_first=True, format=None, encoding=None, bits_per_sample=None):
    if channels_first:
        data = src.T.numpy()
    else:
        data = src.numpy()
    sf.write(str(filepath), data, sample_rate)


torchaudio.load = _load_soundfile
torchaudio.save = _save_soundfile
print("[torchaudio_compat] Patched torchaudio.load/save to use soundfile")
