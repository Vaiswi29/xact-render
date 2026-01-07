import os, re, sys, subprocess

def extract_file_id(url: str) -> str:
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    if re.fullmatch(r"[a-zA-Z0-9_-]{20,}", url):
        return url
    raise ValueError("Could not extract Google Drive file id.")

if __name__ == "__main__":
    drive_url = os.environ.get("MODEL_DRIVE_URL", "").strip()
    if not drive_url:
        print("ERROR: MODEL_DRIVE_URL env var is not set.", file=sys.stderr)
        sys.exit(1)

    file_id = extract_file_id(drive_url)
    out = "xact_finetuned_model.zip"

    cmd = ["gdown", "--id", file_id, "-O", out]
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print("ERROR: gdown failed. Check sharing = Anyone with the link (Viewer).", file=sys.stderr)
        sys.exit(res.returncode)

    print("Download complete ✅")
