import cv2
import os
import numpy as np

def align_images(thermal_img, rgb_img):
    gray_thermal = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2GRAY)
    gray_rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray_thermal, None)
    kp2, des2 = orb.detectAndCompute(gray_rgb, None)

    if des1 is None or des2 is None:
        return thermal_img  # fallback if no features

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:50]

    if len(matches) < 4:
        return thermal_img  # not enough matches

    pts_thermal = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_rgb = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    matrix, _ = cv2.findHomography(pts_thermal, pts_rgb, cv2.RANSAC, 5.0)
    aligned_thermal = cv2.warpPerspective(thermal_img, matrix, (rgb_img.shape[1], rgb_img.shape[0]))
    return aligned_thermal

def overlay_images(thermal_img, rgb_img, alpha=0.5):
    thermal_resized = cv2.resize(thermal_img, (rgb_img.shape[1], rgb_img.shape[0]))
    return cv2.addWeighted(thermal_resized, alpha, rgb_img, 1 - alpha, 0)

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)
    thermal_images = [f for f in files if f.endswith('_T.JPG')]

    for thermal_name in thermal_images:
        identifier = thermal_name.replace('_T.JPG', '')
        rgb_name = identifier + '_Z.JPG'

        thermal_path = os.path.join(input_folder, thermal_name)
        rgb_path = os.path.join(input_folder, rgb_name)

        if not os.path.exists(rgb_path):
            print(f"Missing RGB image for: {identifier}")
            continue

        thermal_img = cv2.imread(thermal_path)
        rgb_img = cv2.imread(rgb_path)

        aligned_thermal = align_images(thermal_img, rgb_img)
        result = overlay_images(aligned_thermal, rgb_img)

        out_path = os.path.join(output_folder, f"{identifier}_overlay.jpg")
        cv2.imwrite(out_path, result)
        print(f"Saved overlay to: {out_path}")

# Run it
process_folder("input", "output")
