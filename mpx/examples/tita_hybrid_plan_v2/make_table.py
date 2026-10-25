"""Markdown table of the v2 robust evaluations (robust_results/*.json)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ["mpc", "hrc_RA3", "hrc_RA4", "hrc_RA5", "v1"]
NAMES = {"mpc": "solo MPC", "hrc_RA3": "RA3 (stanotte)", "hrc_RA4": "RA4 (fine-tune)", "hrc_RA5": "RA5 (pendenza)",
         "v1": "tita_v1"}


def fmt(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def main():
    res = {k: json.loads((HERE / "robust_results" / f"{k}.json").read_text())
           for k in ORDER if (HERE / "robust_results" / f"{k}.json").exists()}
    terrains = ["flat", "rough", "perlin", "slope10"]
    lines = ["| Terreno | Controllore | Cadute | Tempo alla caduta [s] | \\|vx err\\| | \\|wz err\\| | "
             "\\|roll\\| [rad] | \\|pitch\\| [rad] | Deriva heading [rad] | ‖τ_res‖/‖τ_MPC‖ |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for t in terrains:
        for k, d in res.items():
            if t not in d:
                continue
            r = d[t]
            lines.append(
                f"| {t} | {NAMES[k]} | {r['falls']}/{r['episodes']} | {fmt(r['mean_time_to_fall_s'], 1)} | "
                f"{fmt(r['mean_abs_vx_err'])} | {fmt(r['mean_abs_wz_err'])} | {fmt(r.get('mean_abs_roll'))} | "
                f"{fmt(r.get('mean_abs_pitch'))} | {fmt(r.get('final_heading_drift'))} | {fmt(r['residual_ratio'])} |")
    out = "\n".join(lines)
    print(out)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(out + "\n")


if __name__ == "__main__":
    main()
