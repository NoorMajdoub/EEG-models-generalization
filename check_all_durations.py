from moabb.datasets import Yang2025
import pickle

dataset = Yang2025()
durations = {}

for subj in range(1, 63):
    try:
        data = dataset.get_data(subjects=[subj])
        total_min = 0
        for session_id, session_data in data[subj].items():
            for run_id, raw in session_data.items():
                total_min += raw.n_times / raw.info['sfreq'] / 60
        durations[subj] = total_min
        print(f"subject {subj}: {total_min:.2f} min ({total_min/60:.2f}h)")
    except Exception as e:
        print(f"subject {subj} failed: {e}")

with open("yang2025_durations.pkl", "wb") as f:
    pickle.dump(durations, f)

total_hours = sum(durations.values()) / 60
print(f"\nGrand total across {len(durations)} subjects: {total_hours:.2f}h")