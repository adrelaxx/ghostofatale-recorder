"""
CTF SOLVER - Localisation par analyse solaire + image satellite
Village cible : Verdun Ciel (Doubs/Haute-Saône, France)
Coordonnées : LAT = 46.8945173 LON = 5.0232693
DÉPENDANCES :
    pip install opencv-python numpy matplotlib pysolar

USAGE :
    1. Mets tes vidéos et ton image satellite dans le même dossier
    2. Configure la section CONFIG ci-dessous
    3. Lance : python ctf_solver.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, timedelta, timezone
import json
import os
import sys

try:
    from pysolar.solar import get_altitude, get_azimuth
    PYSOLAR_OK = True
except ImportError:
    print("[WARN] pysolar non installé. Lance : pip install pysolar")
    print("       Le script tournera sans calcul solaire (mode debug uniquement)")
    PYSOLAR_OK = False


# ============================================================
#  CONFIGURATION — MODIFIE ICI
# ============================================================

# Coordonnées GPS de Verdun Ciel (centre du village)
LAT = 46.8945173
LON = 5.0232693


# --- SOURCE VIDÉO(s) ---
# Liste de dicts. Chaque dict = une vidéo + ses paramètres
VIDEO_SOURCES = [
    {
        "file": "ghostofatale - 2026-02-22_11-19-28.mp4",
        # Heure réelle de la PREMIÈRE frame de la vidéo
        "real_start": datetime(2026, 2, 22, 11, 19, 28, tzinfo=timezone.utc),
        # ROI : (y_start, x_start, hauteur, largeur)
        "roi": (306, 0, 50, 50),
        # Segments à analyser (en temps vidéo HH:MM:SS)
        "segments": [
            ("00:41:12", "00:51:12"),
            ("01:28:16", "01:34:41"),
            ("05:02:17", "05:14:02"),
            ("07:02:07", "07:15:00"),
        ],
        # Pas d'échantillonnage en secondes (5 = précis, 30 = rapide)
        "step": 10,
    },
    # Tu peux ajouter d'autres vidéos ici avec d'autres ROI
]

# --- IMAGE SATELLITE ---
SATELLITE_IMAGE = "village.png"   # Ton image satellite de Verjux
# Coordonnées GPS des coins de l'image satellite (TL = top-left, BR = bottom-right)
# Mesure sur Google Maps les 4 coins de ton screenshot
SAT_GPS_TL = (46.914627, 4.993286)
SAT_GPS_BR = (46.875213, 5.050964)

# --- SIMULATION ---
# Pas de la grille en pixels (5 = très précis mais lent, 10 = bon compromis)
GRID_STEP = 5
# Orientations fenêtre testées (0 = Nord, 90 = Est, 180 = Sud, 270 = Ouest)
ORIENTATION_STEP = 10   # teste toutes les 10°

# --- SORTIE ---
OUTPUT_HEATMAP = "heatmap_verjux.png"
OUTPUT_DATA    = "brightness_data.json"


# ============================================================
#  ÉTAPE 1 — EXTRACTION LUMIÈRE DEPUIS VIDÉO(s)
# ============================================================

def to_sec(t_str):
    """Convertit HH:MM:SS en secondes."""
    h, m, s = map(int, t_str.split(':'))
    return h * 3600 + m * 60 + s


def extract_brightness(source: dict) -> list:
    """
    Extrait mean/std/variation par step secondes sur les segments demandés.
    Retourne une liste de dicts avec 'dt' (datetime UTC), 'mean', 'std', 'variation'.
    """
    results = []
    file = source["file"]
    roi_y, roi_x, roi_h, roi_w = source["roi"]
    real_start = source["real_start"]
    step = source["step"]

    if not os.path.exists(file):
        print(f"[WARN] Vidéo introuvable : {file} — segment ignoré")
        return results

    cap = cv2.VideoCapture(file)
    if not cap.isOpened():
        print(f"[ERR] Impossible d'ouvrir : {file}")
        return results

    print(f"\n[VIDEO] {file}")
    prev_gray = None

    for seg_start, seg_end in source["segments"]:
        curr = to_sec(seg_start)
        limit = to_sec(seg_end)
        cap.set(cv2.CAP_PROP_POS_MSEC, curr * 1000)

        while curr <= limit:
            ret, frame = cap.read()
            if not ret:
                break

            roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Léger flou pour réduire le bruit capteur
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            mean_val = float(np.mean(gray))
            std_val  = float(np.std(gray))

            if prev_gray is not None and prev_gray.shape == gray.shape:
                variation = float(np.mean(cv2.absdiff(prev_gray, gray)))
            else:
                variation = 0.0
            prev_gray = gray.copy()

            real_dt = real_start + timedelta(seconds=curr)

            results.append({
                "dt": real_dt.isoformat(),
                "mean": mean_val,
                "std": std_val,
                "variation": variation,
            })

            print(f"  {real_dt.strftime('%H:%M:%S')} | mean={mean_val:6.1f} "
                  f"std={std_val:5.1f} var={variation:5.1f}")

            curr += step
            cap.set(cv2.CAP_PROP_POS_MSEC, curr * 1000)

    cap.release()
    return results


def load_or_extract_data() -> list:
    """
    Si brightness_data.json existe → on recharge (gain de temps).
    Sinon → on extrait depuis les vidéos et on sauvegarde.
    """
    if os.path.exists(OUTPUT_DATA):
        print(f"[INFO] Chargement données existantes : {OUTPUT_DATA}")
        with open(OUTPUT_DATA) as f:
            return json.load(f)

    all_data = []
    for source in VIDEO_SOURCES:
        all_data.extend(extract_brightness(source))

    if not all_data:
        print("[ERR] Aucune donnée extraite. Vérifie tes fichiers vidéo.")
        sys.exit(1)

    with open(OUTPUT_DATA, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"[OK] {len(all_data)} samples sauvegardés → {OUTPUT_DATA}")
    return all_data


# ============================================================
#  ÉTAPE 2 — NORMALISATION
# ============================================================

def normalize_data(data: list) -> list:
    """Ajoute un champ 'norm' [0,1] basé sur mean."""
    vals = np.array([d["mean"] for d in data], dtype=float)
    vmin, vmax = vals.min(), vals.max()
    denom = vmax - vmin if vmax != vmin else 1.0
    for i, d in enumerate(data):
        d["norm"] = float((vals[i] - vmin) / denom)
    return data


# ============================================================
#  ÉTAPE 3 — POSITION SOLAIRE
# ============================================================

def get_sun(lat, lon, dt_iso: str):
    """Retourne (altitude_deg, azimuth_deg) du soleil."""
    if not PYSOLAR_OK:
        return 45.0, 180.0   # valeur debug

    dt = datetime.fromisoformat(dt_iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    alt = get_altitude(lat, lon, dt)
    az  = get_azimuth(lat, lon, dt)
    return alt, az


# ============================================================
#  ÉTAPE 4 — SIMULATION LUMIÈRE
# ============================================================

def simulate_light(window_az: float, sun_az: float, sun_alt: float) -> float:
    """
    Modèle physique simplifié :
    - lumière directe si soleil dans le demi-espace de la fenêtre (±90°)
    - pondéré par l'élévation (soleil bas → lumière rasante → moins efficace)
    - zéro si soleil sous l'horizon ou très bas (<3°)
    """
    if sun_alt < 3.0:
        return 0.0

    diff = abs(window_az - sun_az) % 360
    diff = min(diff, 360 - diff)

    direct = max(0.0, np.cos(np.radians(diff)))
    elev   = np.sin(np.radians(min(sun_alt, 90.0)))

    return direct * elev


def simulate_series(data: list, window_az: float) -> np.ndarray:
    """Simule la luminosité normalisée pour une orientation de fenêtre."""
    preds = []
    for d in data:
        alt, az = get_sun(LAT, LON, d["dt"])
        preds.append(simulate_light(window_az, az, alt))

    preds = np.array(preds)
    denom = preds.max() - preds.min() + 1e-6
    return (preds - preds.min()) / denom


# ============================================================
#  ÉTAPE 5 — SCORING
# ============================================================

def compute_score(data: list, pred: np.ndarray) -> float:
    """
    Corrélation de Pearson + pénalité sur les erreurs quadratiques.
    Pearson → capte la forme du signal (timing des pics).
    MSE     → capte l'amplitude.
    """
    measured = np.array([d["norm"] for d in data])

    # Corrélation de Pearson (robuste aux décalages de gain)
    if measured.std() < 1e-6 or pred.std() < 1e-6:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(measured, pred)[0, 1])
        pearson = max(0.0, pearson)   # on ne garde que la corrélation positive

    mse = float(np.mean((measured - pred) ** 2))

    # Score combiné : corrélation forte = bon, erreur faible = bon
    score = pearson * np.exp(-mse * 5)
    return max(0.0, score)


def best_score_for_point(data: list) -> float:
    """
    Pour un point donné (même soleil partout dans le village,
    variation ~0 sur 200m), on teste toutes les orientations
    et on garde le meilleur score.
    
    Note : la position GPS n'influence le soleil que de façon
    négligeable à l'échelle d'un village → on peut pré-calculer
    les positions solaires une seule fois.
    """
    best = 0.0
    for az in range(0, 360, ORIENTATION_STEP):
        pred = simulate_series(data, float(az))
        s = compute_score(data, pred)
        if s > best:
            best = s
    return best


# ============================================================
#  ÉTAPE 6 — CHARGEMENT IMAGE SATELLITE
# ============================================================

def load_satellite():
    if not os.path.exists(SATELLITE_IMAGE):
        print(f"[WARN] Image satellite introuvable : {SATELLITE_IMAGE}")
        print("       Génération d'une image placeholder...")
        img = np.ones((500, 500, 3), dtype=np.uint8) * 200
        cv2.putText(img, "Ajoutez village.png", (50, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        return img
    return cv2.cvtColor(cv2.imread(SATELLITE_IMAGE), cv2.COLOR_BGR2RGB)


# ============================================================
#  ÉTAPE 7 — GRILLE + HEATMAP
# ============================================================

def build_heatmap(data: list, sat_img: np.ndarray) -> np.ndarray:
    """
    Parcourt l'image satellite pixel par pixel (par pas de GRID_STEP)
    et calcule un score de probabilité pour chaque position.
    
    Comme le soleil est quasi-identique pour tout le village (< 200m),
    on pré-calcule les positions solaires UNE FOIS et on mappe le
    résultat sur l'image.
    
    Le score dépend uniquement de l'orientation de la fenêtre, pas
    de la position GPS exacte → la heatmap reflète les zones où
    l'orientation des bâtiments est cohérente avec le signal lumière.
    """
    h, w = sat_img.shape[:2]
    heatmap = np.zeros((h, w), dtype=float)

    # Pré-calcul positions solaires (une seule fois)
    sun_positions = []
    for d in data:
        alt, az = get_sun(LAT, LON, d["dt"])
        sun_positions.append((alt, az))

    # Pré-calcul du meilleur score par orientation
    print("\n[SIMULATION] Calcul des scores par orientation...")
    orientation_scores = {}
    measured = np.array([d["norm"] for d in data])

    for window_az in range(0, 360, ORIENTATION_STEP):
        preds = []
        for alt, az in sun_positions:
            preds.append(simulate_light(float(window_az), az, alt))
        preds = np.array(preds)
        denom = preds.max() - preds.min() + 1e-6
        preds = (preds - preds.min()) / denom
        orientation_scores[window_az] = compute_score(data, preds)

    best_global = max(orientation_scores.values()) + 1e-9
    print(f"[OK] Meilleure orientation : "
          f"{max(orientation_scores, key=orientation_scores.get)}° "
          f"(score={best_global:.4f})")

    # --- MASQUE BÂTIMENT depuis image satellite ---
    # On détecte les zones construites (toits clairs) vs végétation/routes
    print("\n[SATELLITE] Détection des bâtiments...")
    gray_sat = cv2.cvtColor(sat_img, cv2.COLOR_RGB2GRAY)
    edges    = cv2.Canny(gray_sat, 40, 120)
    kernel   = np.ones((7, 7), np.uint8)
    building_mask = cv2.dilate(edges, kernel, iterations=2)

    # Détection des orientations locales via lignes de Hough
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=60)
    # Convertit les angles Hough en azimuts de façades
    facade_orientations = set()
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            az_facade = int(np.degrees(theta)) % 180
            facade_orientations.add(az_facade)
            facade_orientations.add((az_facade + 90) % 360)

    print(f"[OK] {len(facade_orientations)} orientations de façades détectées")

    # Si pas de lignes détectées → on teste toutes les orientations
    if not facade_orientations:
        facade_orientations = set(range(0, 360, ORIENTATION_STEP))

    # Calcul du score pour chaque orientation de façade présente
    facade_scores = {
        az: orientation_scores.get(az - az % ORIENTATION_STEP, 0)
        for az in facade_orientations
    }
    best_facade_score = max(facade_scores.values()) if facade_scores else 0

    # --- REMPLISSAGE HEATMAP ---
    print(f"\n[HEATMAP] Calcul sur grille {w}x{h} (step={GRID_STEP}px)...")
    total = (h // GRID_STEP) * (w // GRID_STEP)
    done = 0

    for py in range(0, h, GRID_STEP):
        for px in range(0, w, GRID_STEP):

            # Score de base : meilleure orientation globale
            base_score = best_global

            # Bonus si la zone correspond à un bâtiment détecté
            is_building = float(building_mask[py, px]) / 255.0
            building_bonus = is_building * 0.3

            # Bonus si l'orientation locale des façades est compatible
            facade_bonus = 0.0
            if is_building > 0.1 and facade_scores:
                facade_bonus = (best_facade_score / best_global) * 0.2

            score = base_score * (1 + building_bonus + facade_bonus)

            # Remplissage du bloc
            y2 = min(py + GRID_STEP, h)
            x2 = min(px + GRID_STEP, w)
            heatmap[py:y2, px:x2] = score

            done += 1
            if done % 5000 == 0:
                pct = 100 * done / total
                print(f"  {pct:.0f}%...", end="\r")

    print("[OK] Heatmap calculée.           ")

    # Normalisation finale [0, 1]
    hmax = heatmap.max()
    if hmax > 1e-9:
        heatmap /= hmax

    return heatmap, orientation_scores


# ============================================================
#  ÉTAPE 8 — VISUALISATION
# ============================================================

def visualize(sat_img: np.ndarray, heatmap: np.ndarray,
              orientation_scores: dict, data: list):

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor('#0d1117')

    # --- (1) Image satellite + heatmap ---
    ax = axes[0]
    ax.imshow(sat_img)
    im = ax.imshow(heatmap, alpha=0.55, cmap='inferno',
                   vmin=0.3, vmax=1.0)
    plt.colorbar(im, ax=ax, label='Score probabilité')
    ax.set_title('Heatmap localisation\n(rouge = zone probable)',
                 color='white', fontsize=11)
    ax.axis('off')

    # Marque le top-3 des pixels
    flat_idx = np.argsort(heatmap.ravel())[::-1][:3]
    h, w = heatmap.shape
    for rank, idx in enumerate(flat_idx):
        py, px = divmod(idx, w)
        ax.plot(px, py, marker='*', markersize=14 - rank*3,
                color=['cyan', 'lime', 'yellow'][rank],
                label=f"Top-{rank+1}")
    ax.legend(loc='lower right', fontsize=8)

    # --- (2) Courbe lumière + simulation meilleure orientation ---
    ax2 = axes[1]
    ax2.set_facecolor('#0d1117')

    times = list(range(len(data)))
    measured_norm = [d["norm"] for d in data]
    best_az = max(orientation_scores, key=orientation_scores.get)

    sun_pos_list = [get_sun(LAT, LON, d["dt"]) for d in data]
    preds = [simulate_light(float(best_az), az, alt)
             for alt, az in sun_pos_list]
    preds = np.array(preds)
    denom = preds.max() - preds.min() + 1e-6
    preds_norm = (preds - preds.min()) / denom

    ax2.plot(times, measured_norm, color='#00d4ff', lw=1.5,
             label='Luminosité mesurée')
    ax2.plot(times, preds_norm, color='#ff6b35', lw=1.5,
             linestyle='--', label=f'Simulation (az={best_az}°)')
    ax2.set_title(f'Signal lumière vs simulation\nMeilleure orientation : {best_az}°',
                  color='white')
    ax2.set_xlabel('Sample index', color='grey')
    ax2.set_ylabel('Luminosité normalisée', color='grey')
    ax2.legend()
    ax2.tick_params(colors='grey')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#333')

    # --- (3) Score par orientation ---
    ax3 = axes[2]
    ax3.set_facecolor('#0d1117')

    azs    = list(orientation_scores.keys())
    scores = list(orientation_scores.values())
    colors = ['#ff6b35' if s == max(scores) else '#00d4ff' for s in scores]

    ax3.bar(azs, scores, width=ORIENTATION_STEP * 0.8, color=colors)
    ax3.set_title('Score par orientation de fenêtre\n(barre orange = meilleure)',
                  color='white')
    ax3.set_xlabel('Azimut fenêtre (°)', color='grey')
    ax3.set_ylabel('Score', color='grey')
    ax3.tick_params(colors='grey')
    for spine in ax3.spines.values():
        spine.set_edgecolor('#333')

    # Compass labels
    compass = {0: 'N', 90: 'E', 180: 'S', 270: 'O'}
    ax3.set_xticks([0, 90, 180, 270, 360])
    ax3.set_xticklabels(['N', 'E', 'S', 'O', 'N'], color='grey')

    fig.suptitle('CTF Solver — Verjux — Localisation par analyse solaire',
                 color='white', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_HEATMAP, dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    print(f"\n[OK] Heatmap sauvegardée → {OUTPUT_HEATMAP}")
    plt.show()


# ============================================================
#  ÉTAPE 9 — RAPPORT CONSOLE
# ============================================================

def print_report(data: list, orientation_scores: dict, heatmap: np.ndarray,
                 sat_img: np.ndarray):
    print("\n" + "="*55)
    print("  RAPPORT CTF SOLVER — VERJUX")
    print("="*55)

    best_az = max(orientation_scores, key=orientation_scores.get)
    best_s  = orientation_scores[best_az]
    print(f"\n  Orientation fenêtre probable   : {best_az}° "
          f"({az_to_compass(best_az)})")
    print(f"  Score de confiance             : {best_s:.4f}")
    print(f"  Nb samples analysés            : {len(data)}")

    # Top-5 pixels
    flat = np.argsort(heatmap.ravel())[::-1][:5]
    h_px, w_px = heatmap.shape
    print("\n  Top-5 positions candidates (pixels → GPS approx) :")
    for rank, idx in enumerate(flat):
        py, px = divmod(idx, w_px)
        lat = gps_from_pixel(py, px, h_px, w_px, "lat")
        lon = gps_from_pixel(py, px, h_px, w_px, "lon")
        sc  = heatmap[py, px]
        print(f"    [{rank+1}] pixel=({px},{py})  GPS≈({lat:.5f}, {lon:.5f})"
              f"  score={sc:.4f}")
        print(f"         https://maps.google.com/?q={lat},{lon}")

    print("\n" + "="*55)


def az_to_compass(az: int) -> str:
    dirs = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
    return dirs[int((az + 22.5) / 45) % 8]


def gps_from_pixel(py, px, h, w, component):
    """Convertit un pixel de l'image satellite en coordonnée GPS."""
    lat_tl, lon_tl = SAT_GPS_TL
    lat_br, lon_br = SAT_GPS_BR
    if component == "lat":
        return lat_tl + (lat_br - lat_tl) * (py / h)
    return lon_tl + (lon_br - lon_tl) * (px / w)


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  CTF SOLVER — LOCALISATION PAR ANALYSE SOLAIRE")
    print(f"  Village : Verjux ({LAT}, {LON})")
    print("=" * 55)

    # 1. Extraction / chargement des données lumière
    data = load_or_extract_data()
    data = normalize_data(data)
    print(f"\n[OK] {len(data)} samples chargés")

    # 2. Chargement image satellite
    sat_img = load_satellite()
    print(f"[OK] Image satellite : {sat_img.shape[1]}x{sat_img.shape[0]} px")

    # 3. Heatmap
    heatmap, orientation_scores = build_heatmap(data, sat_img)

    # 4. Rapport
    print_report(data, orientation_scores, heatmap, sat_img)

    # 5. Visualisation
    visualize(sat_img, heatmap, orientation_scores, data)
