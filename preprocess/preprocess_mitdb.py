import os
import numpy as np
import wfdb
from scipy.signal import resample, kaiserord, firwin, butter, sosfiltfilt

# ----------------------------
# Configurable parameters
# ----------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.environ.get("RAW_DATA_ROOT", os.path.join(project_root, "data", "raw"))
target_fs = 128                                 # target sampling rate in Hz
window_lengths = [10]                       # window lengths in seconds

# Example: Preprocess LTAFDB dataset
dataset_name = "MITDB"
dataset_dir = os.environ.get("MITDB_RAW_DIR", os.path.join(data_path, dataset_name))
dataset_tag = "MITDB"                           # used for output file names

# Class ids (reduced rhythm)
# 0: SINUS, 1: AFIB, 2: OTHR
CLASS_ID = {"SINUS": 0, "AFIB": 1, "OTHR": 2}

# Canonical tokens for rhythm classification (case-insensitive after cleaning)
SINUS_TAGS = {"N", "NSR", "SINUS"}               # rhythm annotations meaning sinus rhythm
AFIB_TAGS  = {"AFIB", "AF"}                      # AF / AFIB tagged as AFIB
# Everything else -> OTHR

# ----------------------------
# Helpers
# ----------------------------
def pick_single_lead(sig, fields):
    """
    Pick a single lead, prioritizing Lead II / MLII if available.
    Fallback to channel 0 if not found.
    """
    idx = 0
    try:
        names = [str(n).upper() for n in (fields.get("sig_name") or [])]
        preferred = {"II", "MLII", "LEAD II", "LEADII", "ECG II", "ECG2"}
        for i, n in enumerate(names):
            if n in preferred:
                idx = i
                break
    except Exception:
        pass
    return sig[:, idx]

def design_kaiser_bandpass(fs, low=1.0, high=30.0, ripple_db=60.0, trans_width=1.0):
    """
    based Kaiser window design 1 -30 Hz FIR bandpass filter.
    Parameters
    ----------
    fs : float
        Sampling frequency (Hz).
    low : float
        Low cutoff frequency (Hz).
    high : float
        High cutoff frequency (Hz).
    ripple_db : float
        Desired stopband attenuation (in dB). 
        60 dB is a typical safe choice.
    trans_width : float
        Transition width (Hz). 
        Smaller values -> sharper transition -> higher filter order.
    """
    nyq = fs / 2.0
    width = trans_width / nyq
    N, beta = kaiserord(ripple_db, width)
    if N % 2 == 0:
        N += 1
    taps = firwin(N, [low, high], window=('kaiser', beta), pass_zero=False, fs=fs)
    return taps

def simple_bandpass_ecg(x, fs, low=1.0, high=30.0, order=4):
    """
    x    : 1D numpy array
    """
    nyq = fs * 0.5
    low  = max(1e-6, float(low))
    high = min(float(high), nyq * 0.999)

    if not (0 < low < high < nyq):
        return x

    sos = butter(order, [low/nyq, high/nyq], btype="bandpass", output="sos")
    y = sosfiltfilt(sos, x, axis=0)
    return y


def normalize_zscore(x):

    """Per-record z-score normalization."""

    m, s = np.mean(x), np.std(x)
    return (x - m) / (s + 1e-8)


def is_rhythm_annot(ann, i):
    """
    Return True if this annotation index i carries a rhythm change label.
    In PhysioNet, rhythm labels are typically in aux_note and start with '('.
    """
    if ann.aux_note is None or len(ann.aux_note) <= i:
        return False
    raw = ann.aux_note[i]
    if raw is None:
        return False
    txt = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
    txt = txt.strip() #used .strip() to avoid leading spaces or trailing spaces
    return len(txt) > 0 and txt[0] == "("  # e.g. "(N", "(AFIB", "(AFL", ...)


def clean_token(raw):
    # 1) Decode bytes if needed
    if isinstance(raw, (bytes, bytearray)):
        txt = raw.decode("latin1", errors="ignore")
    else:
        txt = str(raw)

    # 2) Remove non-printable characters
    # some aux_note contain non-printable characters like \x00
    # ex "(\x00AFIB" -> "(AFIB"
    # ex "(AFIB\x00" -> "(AFIB"
    txt = txt.replace("\x00", "")
    txt = "".join(ch for ch in txt if ch.isprintable())

    # 3) remove leading '(' and trailing ')' and uppercase
    # e.g. "(afib)" -> "AFIB"
    txt = txt.strip()
    if txt.startswith("("):
        txt = txt[1:]
    if txt.endswith(")"):
        txt = txt[:-1]
    txt = txt.upper().strip()

    # 4) Map known variants to canonical forms
    # but in mitdb, N has already been used, so we don't need this mapping
    if txt in {"NORMAL", "NORMAL SINUS RHYTHM"}:
        txt = "N"

    if txt == "AF":  
        txt = "AFIB"

    return txt



def map_token_to_group(tok):
    """
    Map a canonical rhythm token into one of SINUS / AFIB / OTHR groups.
    for example 
    """
    if tok in SINUS_TAGS:
        return "SINUS"
    if tok in AFIB_TAGS:
        return "AFIB"   
    return "OTHR"


def build_rhythm_intervals(ann, rec_len, fs_orig, fs_target):
    """
    Build rhythm intervals in the target sampling domain.
    - ann.sample indices are at original fs; rescale them to target fs.
    - Create half-open intervals [start, end) with a group label per interval.
    """
    changes = []
    for i in range(len(ann.sample)):
        if is_rhythm_annot(ann, i):
            tok = clean_token(ann.aux_note[i])  #(AFIB -> AFIB
            grp = map_token_to_group(tok)
            s = ann.sample[i]
            s = int(round(s * (float(fs_target) / float(fs_orig))))
            s = max(0, s)
            changes.append((s, grp))

    # If none found, assume whole record is SINUS
    if not changes:
        return [(0, rec_len, CLASS_ID["SINUS"])]

    changes.sort(key=lambda x: x[0])
    intervals = []
    for k, (start, grp) in enumerate(changes):
        end = changes[k + 1][0] if (k + 1) < len(changes) else rec_len
        end = min(end, rec_len) #rec_len is the end of the record
        if end > start:
            intervals.append((start, end, CLASS_ID[grp]))
    # From the intervals for example [(100, 200, AFIB), (300, 400, SINUS)]
    # we label the interval segments with intervals = [(0,100,SINUS), (100,200,AFIB), (200,300,SINUS), (300,400,SINUS), (400,rec_len,SINUS)]

    # If first change starts after 0, prepend SINUS as default
    if intervals[0][0] > 0:
        intervals.insert(0, (0, intervals[0][0], CLASS_ID["SINUS"]))

    return intervals


def label_segment_by_rules(start_idx, end_idx, intervals):
    """
    Label a segment according to the required rules:
    - AFIB if ANY overlap with an AFIB interval (AFIB has priority).
    - SINUS only if ALL overlapped rhythm is SINUS (no AFIB and no OTHR).
    - Otherwise OTHR.

    If there is no overlap with any interval (shouldn't happen), default to SINUS.
    """
    overlapped_classes = set()
    for (interval_start, interval_end, interval_label) in intervals:
        overlap_length = max(0, min(end_idx, interval_end) - max(start_idx, interval_start))
        if overlap_length > 0:
            overlapped_classes.add(interval_label)

    if not overlapped_classes:
        return CLASS_ID["SINUS"]

    if CLASS_ID["AFIB"] in overlapped_classes:
        return CLASS_ID["AFIB"]

    # If the set contains only SINUS, it's SINUS
    if overlapped_classes == {CLASS_ID["SINUS"]}:
        return CLASS_ID["SINUS"]

    # Otherwise, anything mixed with OTHR (and without AFIB) is OTHR
    return CLASS_ID["OTHR"]


# ----------------------------
# Main preprocessing
# ----------------------------
out_segments = {w: [] for w in window_lengths}
out_labels   = {w: [] for w in window_lengths}

records = [f for f in os.listdir(dataset_dir) if f.endswith(".dat")]
records.sort()

for dat in records:
    rec_id = dat[:-4]
    record_path = os.path.join(dataset_dir, rec_id)

    # Read signal
    try:
        sig, fields = wfdb.rdsamp(record_path)
    except Exception as e:
        print(f"[Skip] Cannot read signal: {record_path}. Error: {e}")
        continue

    # Read annotations (rhythm usually in 'atr')
    try:
        ann = wfdb.rdann(record_path, "atr")
    
    except Exception as e:
        print(f"[Skip] Cannot read annotations 'atr': {record_path}. Error: {e}")
        continue

    fs_orig = float(fields["fs"])

    # Pick lead (prefer II/MLII)
    signal = pick_single_lead(sig, fields)

    # Resample
    if fs_orig != target_fs:
        n_new = int(round(len(signal) * (target_fs / fs_orig)))
        signal = resample(signal, n_new)
    rec_len = len(signal)

    # Normalize
    signal = normalize_zscore(signal).astype(np.float32)

    # Bandpass filter ""new""
    signal = simple_bandpass_ecg(signal, fs=target_fs, low=1.0, high=30.0, order=4)

    # Build rhythm intervals at target fs
    intervals = build_rhythm_intervals(ann, rec_len, fs_orig, target_fs)

    # build_rhythm_intervals is labeling the entire record
    # label_segment_by_rules is labeling a fixed_length segment based on the rule set

    # Segment with 50% overlap and label
    for w in window_lengths:
        win_samples = int(round(w * target_fs))
        step = max(1, win_samples // 2)  # 50% overlap

        # start_idx goes from 0 to rec_len - win_samples, step by 'step'
        # end_idx = start_idx + win_samples

        for start_idx in range(0, rec_len - win_samples + 1, step):
            end_idx = start_idx + win_samples
            seg = signal[start_idx:end_idx] #build up the 50% overlapping segments
            lbl = label_segment_by_rules(start_idx, end_idx, intervals)

            out_segments[w].append(seg.astype(np.float32))
            out_labels[w].append(lbl)
    
    for w in window_lengths:
        win_samples = int(round(w * target_fs))
        step = max(1, win_samples // 2)
        n_seg = (rec_len - win_samples) // step + 1 if rec_len >= win_samples else 0

        record_labels = out_labels[w][-n_seg:] if n_seg > 0 else []
        record_labels = np.asarray(record_labels, dtype=np.int32)

        uniq, cnt = np.unique(record_labels, return_counts=True)
        stats = ", ".join([f"class{u}:{c}" for u, c in zip(uniq, cnt)])
        print(f"[Record {rec_id}] win={w}s total={len(record_labels)} | {stats}")



# Save outputs
output_dir = os.path.join(
    os.environ.get("PROCESSED_DATA_ROOT", os.path.join(project_root, "data", "processed")),
    dataset_tag,
)
os.makedirs(output_dir, exist_ok=True)
for w in window_lengths:
    win_tag = f"{int(w*10)}s"  # 2.5s -> 25s; 10s -> 100s
    #since 2.5 is a float, we multiply by 10 to avoid the decimal point in the filename
    segs = np.stack(out_segments[w], axis=0) if len(out_segments[w]) else np.empty((0, int(w*target_fs)), np.float32)
    labs = np.asarray(out_labels[w], dtype=np.int32)

    # Append an explicit suffix to indicate 50% overlap and reduced 3-class labels
    np.save(os.path.join(output_dir, f"{dataset_tag}_segments_reduced3_overlap50_{win_tag}_bp.npy"), segs)
    np.save(os.path.join(output_dir, f"{dataset_tag}_labels_reduced3_overlap50_{win_tag}_bp.npy"), labs)

print(f"[Done] {dataset_tag}: reduced-rhythm (SINUS/AFIB/OTHR) with 50% overlap saved.")
