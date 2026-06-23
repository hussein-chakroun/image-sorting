# image-sorting
Sort photos by face recognition with automatic similar-face grouping and an optional desktop UI.

## Install

```powershell
pip install -r requirements.txt
```

Supports JPG, PNG, WEBP, HEIC (iPhone photos), and more.

## Desktop UI (recommended)

```powershell
python app.py
```

1. Choose **From** — the folder with photos to sort (must be different from **To**).
2. Choose **To** — where grouped copies will be written.
3. Click **Start sorting**.

The UI shows live progress: which images are scanned, how faces are compared, and where files are copied. Increase **parallel workers** for faster processing on multi-core machines.

### Reference mode (optional)

Enable **Use reference folder** when you have labeled reference photos:

```text
references/
  Alice/
    alice_1.jpg
    alice_2.jpg
  Bob/
    bob_1.jpg
```

Known matches go to `output/Alice/`, `output/Bob/`, etc. Unmatched similar faces still land in `unknown_group_001/`, and so on.

## Command line

Auto-group similar faces (no reference folder):

```powershell
python main.py "C:\Users\husse\Pictures\photos" "C:\Users\husse\Pictures\sorted"
```

With named references:

```powershell
python main.py "C:\Users\husse\Pictures\photos" "C:\Users\husse\Pictures\sorted" --reference-folder "C:\Users\husse\Pictures\references"
```

## Output

- Similar faces are grouped into `person_001/`, `person_002/`, … (auto mode).
- Named matches use the reference folder names.
- A CSV audit report is written to `output/face_report.csv`.

## Performance

- Face detection runs in parallel across CPU cores (default up to 8 workers).
- Large images are downscaled before detection to keep runs fast.
- OpenCV YuNet + SFace models are downloaded into `models/` on first run.

## Notes

- Reference images with multiple faces use the largest face only.
- Small faces are ignored to reduce bad matches.
- Images with no detectable face are skipped and counted in the report.
