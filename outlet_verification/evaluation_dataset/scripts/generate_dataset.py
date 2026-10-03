import csv
import json
import math
import os
import random
import shutil
import uuid
from pathlib import Path
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

# Set deterministic random seed
random.seed(42)

DATASET_ROOT = Path(__file__).resolve().parent.parent
BASE_OUTLETS_DIR = DATASET_ROOT / "base_outlets"
BASE_IMAGES_DIR = BASE_OUTLETS_DIR / "images"
SUBMISSIONS_DIR = DATASET_ROOT / "submissions"
SUB_IMAGES_DIR = SUBMISSIONS_DIR / "images"
GROUND_TRUTH_DIR = DATASET_ROOT / "ground_truth"
GENERATED_DIR = DATASET_ROOT / "generated"

# Ensure all directories exist
for d in [BASE_IMAGES_DIR, SUB_IMAGES_DIR, GROUND_TRUTH_DIR, GENERATED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Plausible synthetic city anchor coordinates in Southern India
CITY_ANCHORS = {
    "Bengaluru_Indiranagar": (12.9716, 77.5946),
    "Bengaluru_Koramangala": (12.9352, 77.6245),
    "Bengaluru_Jayanagar": (12.9308, 77.5838),
    "Bengaluru_Whitefield": (12.9698, 77.7500),
    "Chennai_AnnaNagar": (13.0850, 80.2101),
    "Chennai_TNagar": (13.0418, 80.2341),
    "Chennai_Adyar": (13.0012, 80.2565),
    "Madurai_KKNagar": (9.9252, 78.1198),
    "Madurai_Simmakkal": (9.9290, 78.1250),
    "Coimbatore_RSPuram": (11.0089, 76.9490),
    "Coimbatore_Gandhipuram": (11.0168, 76.9676),
    "Trichy_ThillaiNagar": (10.8282, 78.6865),
    "Virudhunagar_Market": (9.5872, 77.9578),
}

# Synthetic Brand Profiles
BRAND_SPECS = [
    {"group": "SUNRISE", "base_name": "Sunrise Medicals", "sign_bg": (20, 50, 140), "sign_fg": (255, 230, 80), "wall": (235, 230, 220), "cross_color": (30, 180, 50)},
    {"group": "GREENCARE", "base_name": "GreenCare Pharmacy", "sign_bg": (20, 120, 50), "sign_fg": (255, 255, 255), "wall": (240, 245, 240), "cross_color": (220, 40, 40)},
    {"group": "LAKSHMI", "base_name": "Lakshmi Health Point", "sign_bg": (140, 30, 30), "sign_fg": (255, 240, 180), "wall": (245, 235, 225), "cross_color": (30, 160, 50)},
    {"group": "CITYMED", "base_name": "CityMed Pharmacy", "sign_bg": (30, 100, 160), "sign_fg": (255, 255, 255), "wall": (230, 235, 240), "cross_color": (220, 30, 30)},
    {"group": "BLUECROSS", "base_name": "BlueCross Medicals", "sign_bg": (15, 60, 150), "sign_fg": (255, 255, 255), "wall": (235, 240, 245), "cross_color": (20, 80, 200)},
    {"group": "SRIKRISHNA", "base_name": "Sri Krishna Medical Store", "sign_bg": (160, 80, 20), "sign_fg": (255, 255, 200), "wall": (245, 240, 230), "cross_color": (30, 160, 40)},
    {"group": "MEDPLUS_SYNTH", "base_name": "MedPlus Synthetic Care", "sign_bg": (180, 20, 20), "sign_fg": (255, 255, 255), "wall": (240, 235, 235), "cross_color": (255, 255, 255)},
    {"group": "APOLLO_SYNTH", "base_name": "Apollo QuickMeds Synthetic", "sign_bg": (20, 100, 80), "sign_fg": (255, 220, 80), "wall": (235, 245, 240), "cross_color": (220, 40, 40)},
    {"group": "RELIEF", "base_name": "Relief Pharmacy", "sign_bg": (70, 40, 120), "sign_fg": (255, 255, 255), "wall": (240, 235, 245), "cross_color": (40, 180, 60)},
    {"group": "WELLCARE", "base_name": "WellCare Drug Store", "sign_bg": (30, 140, 130), "sign_fg": (255, 255, 255), "wall": (235, 245, 245), "cross_color": (220, 30, 30)},
]


def render_storefront_image(
    shop_name: str,
    sign_bg: tuple[int, int, int],
    sign_fg: tuple[int, int, int],
    wall_color: tuple[int, int, int],
    cross_color: tuple[int, int, int],
    awning_color: tuple[int, int, int] | None = None,
    size: tuple[int, int] = (256, 256),
) -> Image.Image:
    """
    Renders a realistic synthetic storefront photograph with building façade,
    signboard with shop text, pharmacy cross emblem, entrance door, and display windows.
    """
    w, h = size
    img = Image.new("RGB", size, color=wall_color)
    draw = ImageDraw.Draw(img)

    # 1. Pavement / Sidewalk at bottom
    pavement_top = int(h * 0.82)
    draw.rectangle([0, pavement_top, w, h], fill=(180, 180, 185))
    draw.line([0, pavement_top, w, pavement_top], fill=(130, 130, 135), width=2)
    # Pavement tiles
    for x in range(0, w, 40):
        draw.line([x, pavement_top, x, h], fill=(160, 160, 165), width=1)

    # 2. Awning (optional striped canopy)
    awning_top = int(h * 0.22)
    awning_bottom = int(h * 0.32)
    c_awning = awning_color or sign_bg
    draw.rectangle([10, awning_top, w - 10, awning_bottom], fill=c_awning)
    for x in range(10, w - 10, 20):
        draw.rectangle([x, awning_top, min(x + 10, w - 10), awning_bottom], fill=(255, 255, 255))
    draw.line([10, awning_bottom, w - 10, awning_bottom], fill=(100, 100, 100), width=2)

    # 3. Signboard
    sign_top = int(h * 0.05)
    sign_bottom = int(h * 0.22)
    draw.rectangle([12, sign_top, w - 12, sign_bottom], fill=sign_bg, outline=(255, 255, 255), width=2)

    # Cross Emblem on signboard (Left)
    cx, cy = 32, (sign_top + sign_bottom) // 2
    arm_w, arm_h = 4, 10
    draw.rectangle([cx - arm_w, cy - arm_h, cx + arm_w, cy + arm_h], fill=cross_color)
    draw.rectangle([cx - arm_h, cy - arm_w, cx + arm_h, cy + arm_w], fill=cross_color)

    # Text rendering
    display_name = shop_name.upper()
    if len(display_name) > 22:
        display_name = display_name[:20] + ".."
    
    # Simple default font rendering centered
    text_bbox = draw.textbbox((0, 0), display_name)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]
    tx = max(50, 52 + (w - 70 - tw) // 2)
    ty = (sign_top + sign_bottom - th) // 2
    draw.text((tx, ty), display_name, fill=sign_fg)

    # 4. Entrance Door (center)
    door_left = int(w * 0.38)
    door_right = int(w * 0.62)
    door_top = int(h * 0.42)
    draw.rectangle([door_left, door_top, door_right, pavement_top], fill=(60, 65, 75), outline=(30, 30, 35), width=3)
    # Glass panes on door
    draw.rectangle([door_left + 4, door_top + 6, door_right - 4, int(pavement_top * 0.75)], fill=(160, 200, 220))
    # Door handle
    draw.line([door_right - 8, int(h * 0.62), door_right - 8, int(h * 0.68)], fill=(220, 220, 80), width=3)

    # 5. Display Windows (left and right)
    win_top = int(h * 0.42)
    win_bottom = int(pavement_top * 0.90)

    # Left window
    draw.rectangle([18, win_top, door_left - 12, win_bottom], fill=(175, 210, 225), outline=(40, 45, 50), width=2)
    # Window shelf lines
    draw.line([20, (win_top + win_bottom) // 2, door_left - 14, (win_top + win_bottom) // 2], fill=(130, 160, 175), width=2)
    # Shelf items (small medicine box silhouettes)
    for bx in range(24, door_left - 20, 12):
        draw.rectangle([bx, (win_top + win_bottom) // 2 - 8, bx + 8, (win_top + win_bottom) // 2], fill=(220, 80, 80))

    # Right window
    draw.rectangle([door_right + 12, win_top, w - 18, win_bottom], fill=(175, 210, 225), outline=(40, 45, 50), width=2)
    draw.line([door_right + 14, (win_top + win_bottom) // 2, w - 20, (win_top + win_bottom) // 2], fill=(130, 160, 175), width=2)
    for bx in range(door_right + 18, w - 24, 12):
        draw.rectangle([bx, (win_top + win_bottom) // 2 - 8, bx + 8, (win_top + win_bottom) // 2], fill=(60, 180, 90))

    return img


def apply_near_duplicate_transform(img: Image.Image) -> Image.Image:
    """
    Applies small, realistic photo capture variations:
    crop jitter, slight brightness/contrast variance, and JPEG recompression.
    """
    w, h = img.size
    # 1. Random mild crop (2% to 6%)
    crop_x = int(w * random.uniform(0.02, 0.05))
    crop_y = int(h * random.uniform(0.02, 0.05))
    cropped = img.crop((crop_x, crop_y, w - crop_x, h - crop_y))
    resized = cropped.resize((w, h), Image.Resampling.BILINEAR)

    # 2. Slight brightness & contrast variance (0.94 - 1.06)
    enhancer_b = ImageEnhance.Brightness(resized)
    b_img = enhancer_b.enhance(random.uniform(0.94, 1.06))
    enhancer_c = ImageEnhance.Contrast(b_img)
    final_img = enhancer_c.enhance(random.uniform(0.94, 1.06))

    return final_img


def offset_gps(lat: float, lng: float, distance_meters: float, bearing_deg: float | None = None) -> tuple[float, float]:
    """
    Offsets a coordinate by distance_meters in meters along bearing_deg.
    """
    r_earth = 6371000.0
    bearing = math.radians(bearing_deg if bearing_deg is not None else random.uniform(0, 360))
    lat_rad = math.radians(lat)
    lng_rad = math.radians(lng)

    delta = distance_meters / r_earth
    new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(delta) + math.cos(lat_rad) * math.sin(delta) * math.cos(bearing))
    new_lng_rad = lng_rad + math.atan2(
        math.sin(bearing) * math.sin(delta) * math.cos(lat_rad),
        math.cos(delta) - math.sin(lat_rad) * math.sin(new_lat_rad),
    )
    return round(math.degrees(new_lat_rad), 6), round(math.degrees(new_lng_rad), 6)


def generate_synthetic_dataset():
    print("========================================================")
    print("  PHASE 6A: GENERATING SYNTHETIC EVALUATION DATASET")
    print("========================================================")

    # --------------------------------------------------------------------------
    # 1. GENERATE 55 BASE OUTLETS
    # --------------------------------------------------------------------------
    print("\n1. Generating 55 Base Outlets...")
    base_outlets: list[dict] = []
    base_images_map: dict[str, Image.Image] = {}

    city_keys = list(CITY_ANCHORS.keys())

    for i in range(1, 56):
        outlet_id = f"SYNTH_OUTLET_{i:03d}"
        # Deterministic UUID for database seeding (namespace DNS)
        db_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"synthetic.base.{outlet_id}"))

        brand_spec = BRAND_SPECS[(i - 1) % len(BRAND_SPECS)]
        city_name = city_keys[(i - 1) % len(city_keys)]
        anchor_lat, anchor_lng = CITY_ANCHORS[city_name]

        # Scatter base outlets around city anchor (100m to 1500m)
        lat, lng = offset_gps(anchor_lat, anchor_lng, distance_meters=random.uniform(100.0, 1500.0))

        # Distinct name variation for base outlet
        branch_qualifier = city_name.split("_")[-1]
        name = f"{brand_spec['base_name']} - {branch_qualifier} Branch"
        if i > 30:
            name = f"{brand_spec['base_name']} #{i}"

        img_filename = f"{outlet_id}.jpg"
        img_path = BASE_IMAGES_DIR / img_filename

        # Render storefront
        store_img = render_storefront_image(
            shop_name=brand_spec["base_name"],
            sign_bg=brand_spec["sign_bg"],
            sign_fg=brand_spec["sign_fg"],
            wall_color=brand_spec["wall"],
            cross_color=brand_spec["cross_color"],
        )
        store_img.save(img_path, format="JPEG", quality=92)
        base_images_map[outlet_id] = store_img

        base_outlets.append({
            "outlet_id": outlet_id,
            "db_uuid": db_uuid,
            "name": name,
            "latitude": lat,
            "longitude": lng,
            "image_file": img_filename,
            "brand_group": brand_spec["group"],
            "city": city_name,
            "description": f"Synthetic pharmacy storefront in {city_name}",
        })

    with open(BASE_OUTLETS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(base_outlets, f, indent=2)

    print(f"   Created {len(base_outlets)} base outlets and saved metadata.")

    # --------------------------------------------------------------------------
    # 2. GENERATE SUBMISSIONS ACROSS 11 CATEGORIES (Total: 155 submissions)
    # --------------------------------------------------------------------------
    print("\n2. Generating 155 Evaluation Submissions across 11 Scenarios...")

    submissions: list[dict] = []
    sub_counter = 1

    def add_sub(
        name: str,
        lat: float | None,
        lng: float | None,
        image_img: Image.Image | None,
        scenario: str,
        expected_decision: str,
        expected_match_id: str | None,
        notes: str = "",
    ):
        nonlocal sub_counter
        sub_id = f"SUB_{sub_counter:03d}"
        img_file = None
        if image_img is not None:
            img_file = f"{sub_id}.jpg"
            image_img.save(SUB_IMAGES_DIR / img_file, format="JPEG", quality=90)

        submissions.append({
            "submission_id": sub_id,
            "name": name,
            "latitude": lat,
            "longitude": lng,
            "image_file": img_file,
            "scenario": scenario,
            "expected_decision": expected_decision,
            "expected_match_outlet_id": expected_match_id,
            "notes": notes,
        })
        sub_counter += 1

    # --- CATEGORY A: STRONG DUPLICATES (25 cases) ---
    # Expected: DUPLICATE (same outlet, very close GPS <= 25m, same/near-duplicate image)
    for i in range(25):
        base = base_outlets[i]
        drift_dist = random.uniform(5.0, 25.0)
        s_lat, s_lng = offset_gps(base["latitude"], base["longitude"], drift_dist)
        s_img = apply_near_duplicate_transform(base_images_map[base["outlet_id"]])
        add_sub(
            name=base["name"],
            lat=s_lat,
            lng=s_lng,
            image_img=s_img,
            scenario="STRONG_DUPLICATE",
            expected_decision="DUPLICATE",
            expected_match_id=base["outlet_id"],
            notes=f"Strong duplicate of {base['outlet_id']} shifted {drift_dist:.1f}m",
        )

    # --- CATEGORY B: SAME OUTLET, DIFFERENT NAME (15 cases) ---
    # Expected: DUPLICATE (name altered e.g. 'Medicals' -> 'Pharmacy' or 'Drug Store', GPS <= 20m, near-duplicate image)
    name_suffixes = ["Medical Store", "Pharma & Care", "Health Pharmacy", "Druggists", "Medical Center"]
    for i in range(15):
        base = base_outlets[25 + i]
        s_lat, s_lng = offset_gps(base["latitude"], base["longitude"], random.uniform(3.0, 20.0))
        # Modify name
        tokens = base["name"].split(" - ")[0].split()
        alt_name = f"{tokens[0]} {name_suffixes[i % len(name_suffixes)]}"
        s_img = apply_near_duplicate_transform(base_images_map[base["outlet_id"]])
        add_sub(
            name=alt_name,
            lat=s_lat,
            lng=s_lng,
            image_img=s_img,
            scenario="SAME_OUTLET_DIFFERENT_NAME",
            expected_decision="DUPLICATE",
            expected_match_id=base["outlet_id"],
            notes=f"Altered name '{alt_name}' for {base['outlet_id']}",
        )

    # --- CATEGORY C: SAME BRAND / DIFFERENT BRANCH (15 cases) ---
    # Expected: GENUINE (similar brand name, far GPS > 2000m, distinct image)
    for i in range(15):
        base = base_outlets[i % 20]
        # Distant coordinates (2500m to 8000m away)
        s_lat, s_lng = offset_gps(base["latitude"], base["longitude"], random.uniform(2500.0, 8000.0))
        branch_name = f"{base['name'].split(' - ')[0]} - Remote Extension #{i+1}"
        # Distinct storefront image (different color scheme)
        diff_img = render_storefront_image(
            shop_name=branch_name,
            sign_bg=(random.randint(40, 180), random.randint(40, 180), random.randint(40, 180)),
            sign_fg=(255, 255, 255),
            wall_color=(240, 240, 245),
            cross_color=(200, 30, 30),
        )
        add_sub(
            name=branch_name,
            lat=s_lat,
            lng=s_lng,
            image_img=diff_img,
            scenario="SAME_BRAND_DIFFERENT_BRANCH",
            expected_decision="GENUINE",
            expected_match_id=None,
            notes=f"Different branch of brand {base['brand_group']} located >2.5km away",
        )

    # --- CATEGORY D: SIMILAR NAMES, DIFFERENT OUTLETS (15 cases) ---
    # Expected: GENUINE (name similarity high/moderate, different image, far GPS > 1500m)
    for i in range(15):
        base = base_outlets[15 + i]
        s_lat, s_lng = offset_gps(base["latitude"], base["longitude"], random.uniform(1500.0, 4000.0))
        sim_name = f"{base['name'].split()[0]} General Provisions & Care #{i}"
        diff_img = render_storefront_image(
            shop_name=sim_name,
            sign_bg=(random.randint(50, 150), random.randint(50, 150), random.randint(50, 150)),
            sign_fg=(255, 255, 220),
            wall_color=(230, 230, 230),
            cross_color=(40, 180, 60),
        )
        add_sub(
            name=sim_name,
            lat=s_lat,
            lng=s_lng,
            image_img=diff_img,
            scenario="SIMILAR_NAME_DIFFERENT_OUTLET",
            expected_decision="GENUINE",
            expected_match_id=None,
            notes=f"Unrelated outlet with name prefix sharing with {base['outlet_id']}",
        )

    # --- CATEGORY E: SAME OUTLET WITH GPS DRIFT (15 cases) ---
    # Expected: DUPLICATE (drift values: 5m, 15m, 30m, 50m, 75m, 100m, 150m)
    drift_levels = [5.0, 15.0, 30.0, 50.0, 75.0, 100.0, 150.0]
    for i in range(15):
        base = base_outlets[30 + (i % 25)]
        d_meters = drift_levels[i % len(drift_levels)]
        s_lat, s_lng = offset_gps(base["latitude"], base["longitude"], d_meters)
        s_img = apply_near_duplicate_transform(base_images_map[base["outlet_id"]])
        add_sub(
            name=base["name"],
            lat=s_lat,
            lng=s_lng,
            image_img=s_img,
            scenario="GPS_DRIFT",
            expected_decision="DUPLICATE",
            expected_match_id=base["outlet_id"],
            notes=f"Controlled GPS drift {d_meters}m from {base['outlet_id']}",
        )

    # --- CATEGORY F: CONFLICTING EVIDENCE (15 cases) ---
    # Expected: NEEDS_REVIEW
    # Case F1: High Name + Different Image + Close GPS (6 cases)
    # Case F2: Different Name + Similar Image + Far GPS (5 cases)
    # Case F3: High Name + Different Image + Far GPS (4 cases)
    for i in range(15):
        base = base_outlets[i]
        if i < 6:
            # High Name + Close GPS (15m) + completely different image
            s_lat, s_lng = offset_gps(base["latitude"], base["longitude"], 15.0)
            diff_img = render_storefront_image(
                shop_name="Random Brand Store",
                sign_bg=(240, 20, 20),
                sign_fg=(255, 255, 255),
                wall_color=(200, 200, 200),
                cross_color=(255, 255, 0),
            )
            add_sub(
                name=base["name"],
                lat=s_lat,
                lng=s_lng,
                image_img=diff_img,
                scenario="CONFLICTING_EVIDENCE",
                expected_decision="NEEDS_REVIEW",
                expected_match_id=None,
                notes="Conflicting: Identical name & close GPS but completely different storefront photo",
            )
        elif i < 11:
            # Different Name + Very Similar Image + Far GPS (3500m)
            s_lat, s_lng = offset_gps(base["latitude"], base["longitude"], 3500.0)
            s_img = base_images_map[base["outlet_id"]].copy()
            add_sub(
                name=f"Completely Different Electronics Shop #{i}",
                lat=s_lat,
                lng=s_lng,
                image_img=s_img,
                scenario="CONFLICTING_EVIDENCE",
                expected_decision="NEEDS_REVIEW",
                expected_match_id=None,
                notes="Conflicting: Identical image but completely different name and >3km away",
            )
        else:
            # High Name + Different Image + Far GPS
            s_lat, s_lng = offset_gps(base["latitude"], base["longitude"], 2000.0)
            diff_img = render_storefront_image(
                shop_name="Unique Store",
                sign_bg=(10, 10, 10),
                sign_fg=(255, 255, 255),
                wall_color=(220, 220, 220),
                cross_color=(255, 0, 0),
            )
            add_sub(
                name=base["name"],
                lat=s_lat,
                lng=s_lng,
                image_img=diff_img,
                scenario="CONFLICTING_EVIDENCE",
                expected_decision="NEEDS_REVIEW",
                expected_match_id=None,
                notes="Conflicting: Identical name but different image and far away",
            )

    # --- CATEGORY G: MISSING IMAGE (10 cases) ---
    # 5 Duplicates, 5 Genuine
    for i in range(10):
        base = base_outlets[10 + i]
        if i < 5:
            # Duplicate with missing image (close GPS 20m, same name)
            s_lat, s_lng = offset_gps(base["latitude"], base["longitude"], 20.0)
            add_sub(
                name=base["name"],
                lat=s_lat,
                lng=s_lng,
                image_img=None,
                scenario="MISSING_IMAGE",
                expected_decision="DUPLICATE",
                expected_match_id=base["outlet_id"],
                notes="Duplicate submission with missing image (name + close GPS)",
            )
        else:
            # Genuine with missing image (unrelated name, far GPS 5km)
            s_lat, s_lng = offset_gps(base["latitude"], base["longitude"], 5000.0)
            add_sub(
                name=f"Isolated Rural Clinic #{i}",
                lat=s_lat,
                lng=s_lng,
                image_img=None,
                scenario="MISSING_IMAGE",
                expected_decision="GENUINE",
                expected_match_id=None,
                notes="Genuine submission with missing image (unrelated name and far location)",
            )

    # --- CATEGORY H: MISSING GPS (10 cases) ---
    # 5 Duplicates, 3 Genuine, 2 Needs Review
    for i in range(10):
        base = base_outlets[20 + i]
        if i < 5:
            # Duplicate with missing GPS (exact same name + exact same image)
            s_img = base_images_map[base["outlet_id"]].copy()
            add_sub(
                name=base["name"],
                lat=None,
                lng=None,
                image_img=s_img,
                scenario="MISSING_GPS",
                expected_decision="DUPLICATE",
                expected_match_id=base["outlet_id"],
                notes="Duplicate submission with missing GPS (identical name + identical photo)",
            )
        elif i < 8:
            # Genuine with missing GPS (completely different name + different photo)
            diff_img = render_storefront_image(
                shop_name=f"New Remote Store #{i}",
                sign_bg=(50, 100, 200),
                sign_fg=(255, 255, 255),
                wall_color=(240, 240, 240),
                cross_color=(0, 255, 0),
            )
            add_sub(
                name=f"Novel Entity Healthcare #{i}",
                lat=None,
                lng=None,
                image_img=diff_img,
                scenario="MISSING_GPS",
                expected_decision="GENUINE",
                expected_match_id=None,
                notes="Genuine submission with missing GPS (unrelated name and image)",
            )
        else:
            # Needs Review with missing GPS (moderate name similarity + generic photo)
            gen_img = render_storefront_image(
                shop_name="Generic Medicals",
                sign_bg=(100, 100, 100),
                sign_fg=(255, 255, 255),
                wall_color=(230, 230, 230),
                cross_color=(255, 0, 0),
            )
            add_sub(
                name=f"{base['name'].split()[0]} Medicals",
                lat=None,
                lng=None,
                image_img=gen_img,
                scenario="MISSING_GPS",
                expected_decision="NEEDS_REVIEW",
                expected_match_id=None,
                notes="Ambiguous submission with missing GPS",
            )

    # --- CATEGORY I: WEAK EVIDENCE (10 cases) ---
    # Expected: NEEDS_REVIEW (moderate name ~0.55, missing image, moderate distance ~300m)
    for i in range(10):
        base = base_outlets[35 + (i % 20)]
        s_lat, s_lng = offset_gps(base["latitude"], base["longitude"], 300.0)
        weak_name = f"{base['name'].split()[0]} General Store"
        add_sub(
            name=weak_name,
            lat=s_lat,
            lng=s_lng,
            image_img=None,  # Missing image
            scenario="WEAK_EVIDENCE",
            expected_decision="NEEDS_REVIEW",
            expected_match_id=None,
            notes="Weak evidence: moderate name similarity, missing image, 300m distance",
        )

    # --- CATEGORY J: COMPLETELY NEW OUTLETS (20 cases) ---
    # Expected: GENUINE (novel name, novel image, distant/unique coordinates)
    novel_names = [
        "Aura Organic Herbals", "Blissful Ayur Care", "Chaitanya Dental Depot",
        "Dhanvantari Health Line", "Evergreen Super Pharmacy", "Frontier Surgicals",
        "Genesis Pharma Hub", "Harmony Holistic Wellness", "Infinity Diagnostics",
        "Jupiter Healthcare", "Kalyan Ayurvedic Point", "LifeSpring Homeopathy",
        "Maruti Medicine Center", "Narayana Health Oasis", "Omkar Medical Hall",
        "Prana Vitality Pharma", "Quantum Care Drugs", "Radiance Mother & Child Meds",
        "Shanti Wellness Dispensary", "Trinetra Eye & Health Pharmacy"
    ]
    for i in range(20):
        # Anchor around coordinates distinctly separated from base outlets
        base_anchor = CITY_ANCHORS["Chennai_Adyar"] if i % 2 == 0 else CITY_ANCHORS["Bengaluru_Koramangala"]
        s_lat, s_lng = offset_gps(base_anchor[0], base_anchor[1], random.uniform(12000.0, 25000.0))
        n_name = novel_names[i]
        n_img = render_storefront_image(
            shop_name=n_name,
            sign_bg=(random.randint(20, 160), random.randint(20, 160), random.randint(20, 160)),
            sign_fg=(255, 255, 255),
            wall_color=(random.randint(220, 250), random.randint(220, 250), random.randint(220, 250)),
            cross_color=(random.randint(20, 220), random.randint(20, 220), 40),
        )
        add_sub(
            name=n_name,
            lat=s_lat,
            lng=s_lng,
            image_img=n_img,
            scenario="NEW_OUTLET",
            expected_decision="GENUINE",
            expected_match_id=None,
            notes=f"Completely brand new outlet '{n_name}' with novel branding and location",
        )

    # --- CATEGORY K: PERFECT MATCH (5 cases) ---
    # Expected: DUPLICATE (exact identical name, exact identical image file, distance = 0)
    for i in range(5):
        base = base_outlets[i * 10]
        s_img = base_images_map[base["outlet_id"]].copy()
        add_sub(
            name=base["name"],
            lat=base["latitude"],
            lng=base["longitude"],
            image_img=s_img,
            scenario="PERFECT_MATCH",
            expected_decision="DUPLICATE",
            expected_match_id=base["outlet_id"],
            notes=f"Exact perfect clone of {base['outlet_id']} (identical name, image, coordinates)",
        )

    print(f"   Generated total of {len(submissions)} submissions.")

    # --------------------------------------------------------------------------
    # 3. SAVE SUBMISSION METADATA & GROUND TRUTH
    # --------------------------------------------------------------------------
    with open(SUBMISSIONS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(submissions, f, indent=2)

    # Ground truth JSON
    ground_truth_records = [
        {
            "submission_id": s["submission_id"],
            "name": s["name"],
            "scenario": s["scenario"],
            "expected_decision": s["expected_decision"],
            "expected_match_outlet_id": s["expected_match_outlet_id"],
            "has_image": s["image_file"] is not None,
            "has_gps": s["latitude"] is not None and s["longitude"] is not None,
            "notes": s["notes"],
        }
        for s in submissions
    ]

    with open(GROUND_TRUTH_DIR / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(ground_truth_records, f, indent=2)

    # Ground truth CSV
    csv_file = GROUND_TRUTH_DIR / "ground_truth.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "submission_id", "name", "scenario", "expected_decision",
            "expected_match_outlet_id", "has_image", "has_gps", "notes"
        ])
        writer.writeheader()
        writer.writerows(ground_truth_records)

    # --------------------------------------------------------------------------
    # 4. COMPUTE & SAVE DATASET STATISTICS
    # --------------------------------------------------------------------------
    decisions = [s["expected_decision"] for s in submissions]
    scenarios = [s["scenario"] for s in submissions]

    stats = {
        "dataset_name": "Synthetic Outlet Verification Evaluation Dataset (Phase 6A)",
        "total_base_outlets": len(base_outlets),
        "total_submissions": len(submissions),
        "ground_truth_distribution": {
            "DUPLICATE": decisions.count("DUPLICATE"),
            "GENUINE": decisions.count("GENUINE"),
            "NEEDS_REVIEW": decisions.count("NEEDS_REVIEW"),
        },
        "scenario_distribution": {
            "STRONG_DUPLICATE": scenarios.count("STRONG_DUPLICATE"),
            "SAME_OUTLET_DIFFERENT_NAME": scenarios.count("SAME_OUTLET_DIFFERENT_NAME"),
            "SAME_BRAND_DIFFERENT_BRANCH": scenarios.count("SAME_BRAND_DIFFERENT_BRANCH"),
            "SIMILAR_NAME_DIFFERENT_OUTLET": scenarios.count("SIMILAR_NAME_DIFFERENT_OUTLET"),
            "GPS_DRIFT": scenarios.count("GPS_DRIFT"),
            "CONFLICTING_EVIDENCE": scenarios.count("CONFLICTING_EVIDENCE"),
            "MISSING_IMAGE": scenarios.count("MISSING_IMAGE"),
            "MISSING_GPS": scenarios.count("MISSING_GPS"),
            "WEAK_EVIDENCE": scenarios.count("WEAK_EVIDENCE"),
            "NEW_OUTLET": scenarios.count("NEW_OUTLET"),
            "PERFECT_MATCH": scenarios.count("PERFECT_MATCH"),
        },
        "image_statistics": {
            "base_images_count": len(list(BASE_IMAGES_DIR.glob("*.jpg"))),
            "submission_images_count": len(list(SUB_IMAGES_DIR.glob("*.jpg"))),
            "missing_images_count": sum(1 for s in submissions if s["image_file"] is None),
        },
        "gps_statistics": {
            "submissions_with_gps": sum(1 for s in submissions if s["latitude"] is not None),
            "missing_gps_count": sum(1 for s in submissions if s["latitude"] is None),
        },
    }

    with open(GENERATED_DIR / "dataset_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n--------------------------------------------------------")
    print("      DATASET GENERATION COMPLETED SUCCESSFULLY!")
    print(f"      Base Outlets:   {stats['total_base_outlets']}")
    print(f"      Submissions:    {stats['total_submissions']}")
    print(f"      Decisions:      {stats['ground_truth_distribution']}")
    print(f"      Images:         Base={stats['image_statistics']['base_images_count']}, Submissions={stats['image_statistics']['submission_images_count']}")
    print(f"      Missing:        Images={stats['image_statistics']['missing_images_count']}, GPS={stats['gps_statistics']['missing_gps_count']}")
    print("--------------------------------------------------------\n")


if __name__ == "__main__":
    generate_synthetic_dataset()
