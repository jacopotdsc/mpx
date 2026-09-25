"""Riduce l'ampiezza del terreno rough del Lite3 (scene_rough.xml).

Il rough e' un XML statico di ~2394 box a pos_z=-0.25 e size_z variabile: la
sommita' di ogni box (pos_z+size_z) forma la superficie. Le sommita' sono
simmetriche attorno a ~0 (mean -0.0003, std 0.0144, range +-0.025 m). Questo
script scala la deviazione di ogni sommita' dal livello medio per un fattore k
(k<1 = terreno piu' facile), preservando il livello medio del suolo, e riscrive
SOLO l'attributo size dei geom box (il piano 'floor' e tutto il resto invariati).

E' l'unico modo pulito: non esiste un generatore per questo scene.
Backup automatico in analysis_lite3/mpc_tuning/rough_backup/ (l'originale e'
comunque nel git di mujoco_playground). NIENTE commit/push.

Uso:
    python scale_rough.py <k>            # applica (default k=0.6)
    python scale_rough.py <k> --dry      # solo statistiche, non scrive
    python scale_rough.py --restore      # ripristina dal backup
"""
import os, re, shutil, sys
import statistics as st

from mujoco_playground._src.locomotion.lite3 import lite3_constants as consts

XML = consts.task_to_xml("rough_terrain").as_posix()
BACKUP_DIR = os.path.join(os.path.dirname(__file__), "rough_backup")
BACKUP = os.path.join(BACKUP_DIR, "scene_rough.xml.orig")

# match a box geom, capturing the whole size="sx sy sz" so we can rewrite sz.
BOX_RE = re.compile(
    r'(<geom pos="(-?\d+\.?\d*) (-?\d+\.?\d*) (-?\d+\.?\d*)" type="box" '
    r'size="(\d+\.?\d*) (\d+\.?\d*) )(\d+\.?\d*)(")'
)


def stats(text):
    tops = [float(m.group(4)) + float(m.group(7)) for m in BOX_RE.finditer(text)]
    return dict(n=len(tops), mean=st.mean(tops), std=st.pstdev(tops),
                mn=min(tops), mx=max(tops))


def main():
    args = sys.argv[1:]
    if "--restore" in args:
        assert os.path.exists(BACKUP), "nessun backup da ripristinare"
        shutil.copy2(BACKUP, XML)
        print(f"ripristinato {XML} dal backup")
        return
    dry = "--dry" in args
    k = next((float(a) for a in args if not a.startswith("--")), 0.6)

    text = open(XML).read()
    before = stats(text)
    mean_top = before["mean"]
    print(f"PRIMA : n={before['n']} mean={before['mean']:.4f} std={before['std']:.4f} "
          f"min={before['mn']:.4f} max={before['mx']:.4f}")

    def repl(m):
        pos_z = float(m.group(4)); size_z = float(m.group(7))
        top = pos_z + size_z
        new_top = mean_top + k * (top - mean_top)   # scala verso la media
        new_size_z = new_top - pos_z
        return f"{m.group(1)}{new_size_z:.6f}{m.group(8)}"

    new_text = BOX_RE.sub(repl, text)
    after = stats(new_text)
    print(f"DOPO  : n={after['n']} mean={after['mean']:.4f} std={after['std']:.4f} "
          f"min={after['mn']:.4f} max={after['mx']:.4f}   (k={k})")
    if dry:
        print("[dry-run] nessuna scrittura")
        return
    os.makedirs(BACKUP_DIR, exist_ok=True)
    if not os.path.exists(BACKUP):
        shutil.copy2(XML, BACKUP)
        print(f"backup creato: {BACKUP}")
    else:
        print(f"backup gia' presente (non sovrascritto): {BACKUP}")
    open(XML, "w").write(new_text)
    print(f"scritto: {XML}")


if __name__ == "__main__":
    main()
