from pathlib import Path
import pyarrow.parquet as pq


PARQUET_FILE = Path("../data/audio/asvspoof_raw/test-00000-of-00009.parquet")

REAL_DIR = Path("data/audio/real")
FAKE_DIR = Path("data/audio/fake")

REAL_DIR.mkdir(parents=True, exist_ok=True)
FAKE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_REAL = 100
TARGET_FAKE = 100

real_count = 0
fake_count = 0

parquet = pq.ParquetFile(PARQUET_FILE)

print("=" * 50)
print("ASVSPOOF AUDIO EXTRACTION")
print("=" * 50)

for group_number in range(parquet.num_row_groups):

    if real_count >= TARGET_REAL and fake_count >= TARGET_FAKE:
        break

    print(f"Processing row group {group_number + 1}/{parquet.num_row_groups}...")

    table = parquet.read_row_group(
        group_number,
        columns=["path", "audio", "label"]
    )

    paths = table["path"].to_pylist()
    audios = table["audio"].to_pylist()
    labels = table["label"].to_pylist()

    for path, audio, label in zip(paths, audios, labels):

        # ASVspoof labels:
        # 0 = bonafide (real)
        # 1 = spoof (fake)

        if isinstance(label, str):
            label_lower = label.lower()

            if label_lower in ["bonafide", "real"]:
                is_real = True
            elif label_lower in ["spoof", "fake"]:
                is_real = False
            else:
                continue
        else:
            is_real = int(label) == 0

        # Extract audio bytes
        audio_bytes = None

        if isinstance(audio, dict):
            audio_bytes = audio.get("bytes")

        if audio_bytes is None:
            continue

        filename = Path(path).name

        if is_real and real_count < TARGET_REAL:
            output_file = REAL_DIR / filename
            output_file.write_bytes(audio_bytes)
            real_count += 1

        elif not is_real and fake_count < TARGET_FAKE:
            output_file = FAKE_DIR / filename
            output_file.write_bytes(audio_bytes)
            fake_count += 1

        if real_count >= TARGET_REAL and fake_count >= TARGET_FAKE:
            break

print()
print("=" * 50)
print("EXTRACTION COMPLETE")
print("=" * 50)
print(f"Real files : {real_count}")
print(f"Fake files : {fake_count}")
print()
print(f"Real folder: {REAL_DIR}")
print(f"Fake folder: {FAKE_DIR}")
