"""
Téléchargement Allociné depuis GitHub (TheophileBlard).
Le dataset est dans data.tar.bz2 contenant des fichiers .jsonl

Source : https://github.com/TheophileBlard/french-sentiment-analysis-with-bert

Usage : python data/download_data.py
"""

from pathlib import Path
import requests
import tarfile
import io
import pandas as pd

DATA_RAW = Path(__file__).parent / "raw"

URL = "https://github.com/TheophileBlard/french-sentiment-analysis-with-bert/raw/master/allocine_dataset/data.tar.bz2"


def main():
    print("=" * 60)
    print("📥 Téléchargement Allociné Reviews")
    print("   Source : GitHub (TheophileBlard)")
    print("=" * 60)

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    output = DATA_RAW / "allocine_dataset.csv"

    if output.exists() and output.stat().st_size > 0:
        print(f"\n✓ Déjà présent : {output.name}")
        return

    # Télécharger l'archive
    print("\n  Téléchargement de data.tar.bz2...")
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    print(f"  ✓ Téléchargé ({len(r.content) / 1e6:.1f} Mo)")

    # Extraire
    print("  Extraction...")
    tar = tarfile.open(fileobj=io.BytesIO(r.content), mode="r:bz2")
    tar.extractall(path=DATA_RAW, filter="data")
    tar.close()

    # Charger uniquement les fichiers .jsonl (ignorer le .pickle)
    frames = []
    for f in sorted(DATA_RAW.rglob("*.jsonl")):
        try:
            df_split = pd.read_json(f, lines=True)
            df_split["split"] = f.stem
            frames.append(df_split)
            print(f"  ✓ {f.stem}.jsonl → {len(df_split):,} lignes")
        except Exception as e:
            print(f"  ✗ {f.name} : {e}")

    if not frames:
        print("\n❌ Aucun fichier .jsonl trouvé")
        return

    df = pd.concat(frames, ignore_index=True)

    # Harmoniser les colonnes
    if "polarity" in df.columns and "label" not in df.columns:
        df["label"] = df["polarity"]
    if "review" not in df.columns and "text" in df.columns:
        df["review"] = df["text"]

    df.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"\n✓ Exporté : {output.name} ({len(df):,} lignes)")
    print(f"  Colonnes : {df.columns.tolist()}")
    if "label" in df.columns:
        print(f"  Labels : {df['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
