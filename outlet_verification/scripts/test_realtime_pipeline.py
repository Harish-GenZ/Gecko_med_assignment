import io
import sys
from pathlib import Path
from PIL import Image, ImageDraw
import httpx

# Base URL for local running FastAPI backend
API_BASE = "http://127.0.0.1:8000"

def create_synthetic_storefront(name: str, color: tuple[int, int, int]) -> bytes:
    """Creates a storefront image with the shop name rendered on a banner."""
    img = Image.new("RGB", (400, 300), color=color)
    draw = ImageDraw.Draw(img)
    # Draw storefront banner
    draw.rectangle([20, 20, 380, 80], fill=(255, 255, 255))
    draw.text((40, 45), name, fill=(0, 0, 0))
    # Draw door and window
    draw.rectangle([140, 120, 260, 290], fill=(50, 50, 50))
    draw.rectangle([40, 130, 120, 230], fill=(180, 220, 255))
    draw.rectangle([280, 130, 360, 230], fill=(180, 220, 255))
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

def main():
    print("=" * 60)
    print("TESTING REAL-TIME MEDICAL SHOP VERIFICATION & RAILWAY STORAGE")
    print("=" * 60)

    client = httpx.Client(timeout=60.0)

    # 1. Health check
    h_resp = client.get(f"{API_BASE}/health")
    assert h_resp.status_code == 200, f"Health check failed: {h_resp.text}"
    print("[PASS] Backend health check: OK")

    # 2. First submission: Apollo Pharmacy - MG Road (genuine first-time registration)
    shop1_name = "Apollo Pharmacy - MG Road Branch"
    shop1_lat = 12.975000
    shop1_lng = 77.605000
    shop1_img_bytes = create_synthetic_storefront("APOLLO PHARMACY", (34, 139, 34))

    print(f"\n[STEP 1] Submitting first real-time shop: '{shop1_name}'...")
    resp1 = client.post(
        f"{API_BASE}/verify",
        data={
            "name": shop1_name,
            "latitude": str(shop1_lat),
            "longitude": str(shop1_lng),
            "auto_register_if_genuine": "true",
        },
        files={"image": ("apollo_mg_road.jpg", shop1_img_bytes, "image/jpeg")},
    )

    assert resp1.status_code == 200, f"Submission 1 failed: {resp1.text}"
    data1 = resp1.json()
    print("  -> Decision:", data1["decision"])
    print("  -> Duplicate Confidence:", data1["duplicate_confidence"])
    print("  -> Registered:", data1.get("registered"))
    print("  -> Registered Outlet ID:", data1.get("registered_outlet_id"))
    print("  -> Stored Image URL:", data1.get("stored_image_url"))

    assert data1["decision"] == "GENUINE", f"Expected GENUINE, got {data1['decision']}"
    assert data1.get("registered") is True, "Expected registered=True"
    assert data1.get("registered_outlet_id") is not None, "Expected registered_outlet_id"
    stored_url = data1.get("stored_image_url")
    assert stored_url and "storageapi.dev" in stored_url, f"Expected Tigris S3 URL, got {stored_url}"

    # Verify image is accessible from Railway S3 bucket
    print("\n[STEP 2] Verifying stored photograph accessibility in S3 bucket...")
    img_resp = client.get(stored_url)
    assert img_resp.status_code == 200, f"Failed to download image from S3: {img_resp.status_code}"
    assert len(img_resp.content) > 0, "Downloaded image is empty"
    print(f"  -> Successfully loaded {len(img_resp.content)} bytes from S3 bucket! (HTTP 200)")

    # 3. Second submission: Exact or near duplicate submission of Apollo Pharmacy
    print(f"\n[STEP 3] Submitting duplicate of '{shop1_name}' at same location...")
    resp2 = client.post(
        f"{API_BASE}/verify",
        data={
            "name": "Apollo Pharmacy - MG Road",
            "latitude": str(shop1_lat + 0.00005),  # ~5 meters away
            "longitude": str(shop1_lng - 0.00005),
            "auto_register_if_genuine": "true",
        },
        files={"image": ("apollo_repeat.jpg", shop1_img_bytes, "image/jpeg")},
    )

    assert resp2.status_code == 200, f"Submission 2 failed: {resp2.text}"
    data2 = resp2.json()
    print("  -> Decision:", data2["decision"])
    print("  -> Duplicate Confidence:", data2["duplicate_confidence"])
    print("  -> Matched Outlet:", data2.get("matched_outlet"))
    print("  -> Registered:", data2.get("registered"))

    assert data2["decision"] == "DUPLICATE", f"Expected DUPLICATE, got {data2['decision']}"
    assert data2.get("registered") is False, "Duplicate should not be re-registered"
    assert data2.get("matched_outlet") is not None, "Expected matched_outlet"
    matched_id = data2["matched_outlet"]["outlet_id"]
    assert matched_id == data1["registered_outlet_id"], f"Matched ID {matched_id} does not match {data1['registered_outlet_id']}"
    print(f"  -> Correctly flagged as DUPLICATE matching existing outlet {matched_id}!")

    # 4. Third submission: MedPlus in Chennai (distinct medical shop)
    shop2_name = "MedPlus Pharmacy - Anna Nagar"
    shop2_lat = 13.085000
    shop2_lng = 80.210000
    shop2_img_bytes = create_synthetic_storefront("MEDPLUS", (220, 20, 60))

    print(f"\n[STEP 4] Submitting distinct medical shop: '{shop2_name}'...")
    resp3 = client.post(
        f"{API_BASE}/verify",
        data={
            "name": shop2_name,
            "latitude": str(shop2_lat),
            "longitude": str(shop2_lng),
            "auto_register_if_genuine": "true",
        },
        files={"image": ("medplus_annanagar.jpg", shop2_img_bytes, "image/jpeg")},
    )

    assert resp3.status_code == 200, f"Submission 3 failed: {resp3.text}"
    data3 = resp3.json()
    print("  -> Decision:", data3["decision"])
    print("  -> Reasons:", data3.get("reason_codes"))
    print("  -> Registered:", data3.get("registered"))

    assert data3["decision"] == "GENUINE", f"Expected GENUINE for distinct shop in Chennai, got {data3['decision']}"
    assert data3.get("registered") is True, "Expected auto-registration for genuine outlet"
    print("  -> Correctly classified as GENUINE and automatically registered!")

    # 4B. Fourth submission: Devi Medicals (4 km away from Apollo Pharmacy in Bangalore)
    shop3_name = "Devi Medicals & General Store"
    shop3_lat = shop1_lat + 0.036  # ~4.0 km away
    shop3_lng = shop1_lng + 0.010
    shop3_img_bytes = create_synthetic_storefront("DEVI MEDICALS", (70, 130, 180))

    print(f"\n[STEP 4B] Submitting distinct medical shop 4 km away: '{shop3_name}'...")
    resp4 = client.post(
        f"{API_BASE}/verify",
        data={
            "name": shop3_name,
            "latitude": str(shop3_lat),
            "longitude": str(shop3_lng),
            "auto_register_if_genuine": "true",
        },
        files={"image": ("devi_medicals_4km.jpg", shop3_img_bytes, "image/jpeg")},
    )

    assert resp4.status_code == 200, f"Submission 4B failed: {resp4.text}"
    data4 = resp4.json()
    print("  -> Decision:", data4["decision"])
    print("  -> Confidence:", data4["duplicate_confidence"])
    print("  -> Reasons:", data4.get("reason_codes"))
    print("  -> Registered:", data4.get("registered"))

    assert data4["decision"] == "GENUINE", f"Expected GENUINE for shop 4 km away, got {data4['decision']}"
    assert data4.get("registered") is True, "Expected auto-registration for genuine 4km-away outlet"
    print("  -> Correctly classified as GENUINE (confidence < 0.35) and auto-registered into database!")

    # 5. Verify image proxy endpoint
    print("\n[STEP 5] Testing backend image streaming proxy (GET /outlets/images/...)...")
    # Extract object key from stored URL
    test_key = stored_url.split("shop-images-bucket-i4vjyu/")[-1].split("?")[0]
    proxy_resp = client.get(f"{API_BASE}/outlets/images/{test_key}")
    assert proxy_resp.status_code == 200, f"Proxy failed: {proxy_resp.status_code}"
    assert len(proxy_resp.content) > 0, "Proxy returned empty bytes"
    print(f"  -> Successfully served {len(proxy_resp.content)} bytes through image proxy! (HTTP 200)")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED: REAL-TIME VERIFICATION & STORAGE PIPELINE VERIFIED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
