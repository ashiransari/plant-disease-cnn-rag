import os
import cv2
import numpy as np
from PIL import Image
import imagehash
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim
import shutil
import hashlib
import argparse

def calculate_md5(filepath):
    """Calculate MD5 hash for exact file matching."""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def find_duplicates(image_folder, output_folder=None, hash_size=8, threshold=5, 
                   similarity_threshold=0.95, move_duplicates=False, preview=False):
    """
    Advanced duplicate image detector with multiple comparison methods.

    Args:
        image_folder (str): Path to folder containing images.
        output_folder (str): Where to move duplicates (if None, deletes them).
        hash_size (int): Perceptual hash size (8, 16, or 32).
        threshold (int): Max hash difference for duplicates (0-64).
        similarity_threshold (float): Structural similarity index (0-1).
        move_duplicates (bool): If True, moves instead of deleting.
        preview (bool): Show duplicates before removal.
    """
    # Initialize
    hashes = defaultdict(list)
    md5_hashes = defaultdict(list)
    duplicates = set()
    total_files = 0
    supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')

    print(f"[+] Scanning {image_folder} for duplicates...")

    # First pass: MD5 exact duplicates
    print("[1/3] Checking for exact duplicates (MD5)...")
    for root, _, files in os.walk(image_folder):
        for filename in files:
            if filename.lower().endswith(supported_exts):
                filepath = os.path.join(root, filename)
                try:
                    md5 = calculate_md5(filepath)
                    if md5 in md5_hashes:
                        duplicates.add(filepath)
                        print(f"  Exact duplicate found: {filename}")
                    else:
                        md5_hashes[md5].append(filepath)
                    total_files += 1
                except Exception as e:
                    print(f"  Error processing {filename}: {e}")

    # Second pass: Perceptual hashing
    print("[2/3] Checking for perceptual duplicates (pHash)...")
    for file_list in md5_hashes.values():
        filepath = file_list[0]  # Only need one file per MD5 group
        try:
            img = Image.open(filepath)
            phash = str(imagehash.phash(img, hash_size=hash_size))
            hashes[phash].append(filepath)
        except Exception as e:
            print(f"  Error hashing {filepath}: {e}")

    # Find similar hashes
    hash_list = list(hashes.keys())
    for i in range(len(hash_list)):
        for j in range(i+1, len(hash_list)):
            hamming_dist = sum(c1 != c2 for c1, c2 in zip(hash_list[i], hash_list[j]))
            if hamming_dist <= threshold:
                for dup in hashes[hash_list[j]]:
                    duplicates.add(dup)

    # Third pass: Structural Similarity (SSIM) for borderline cases
    if similarity_threshold < 1.0:
        print("[3/3] Verifying with structural similarity...")
        borderlines = []
        for original, candidates in hashes.items():
            if len(candidates) > 1:
                try:
                    img1 = cv2.imread(candidates[0])
                    if img1 is None:
                        continue
                    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

                    for candidate in candidates[1:]:
                        img2 = cv2.imread(candidate)
                        if img2 is None:
                            continue
                        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

                        # Resize to same dimensions
                        h, w = img1.shape
                        img2 = cv2.resize(img2, (w, h))

                        # Calculate SSIM
                        score, _ = ssim(img1, img2, full=True)
                        if score >= similarity_threshold:
                            duplicates.add(candidate)
                        else:
                            borderlines.append((candidates[0], candidate, score))
                except Exception as e:
                    print(f"  SSIM error: {e}")

    # Handle duplicates
    print(f"\n[+] Found {len(duplicates)} duplicates among {total_files} images")
    if not duplicates:
        print("No duplicates found!")
        return

    # Create output folder if moving files
    if move_duplicates and output_folder:
        os.makedirs(output_folder, exist_ok=True)

    # Process duplicates
    for dup in sorted(duplicates):
        try:
            if preview:
                original = next((x for x in hashes.values() if dup in x), [None])[0]
                if original:
                    show_comparison(original, dup)

            if move_duplicates:
                dest = os.path.join(output_folder, os.path.basename(dup))
                shutil.move(dup, dest)
                print(f"Moved: {dup} → {dest}")
            else:
                os.remove(dup)
                print(f"Deleted: {dup}")
        except Exception as e:
            print(f"Error processing {dup}: {e}")

    print("\n[+] Duplicate removal complete!")
    if borderlines:
        print("\nBorderline cases (similar but not duplicates):")
        for orig, cand, score in borderlines[:5]:  # Show top 5
            print(f"  {os.path.basename(orig)} vs {os.path.basename(cand)}: SSIM={score:.2f}")

def show_comparison(img1_path, img2_path):
    """Display side-by-side comparison of images."""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        return

    h = min(img1.shape[0], img2.shape[0])
    img1 = cv2.resize(img1, (int(h*img1.shape[1]/img1.shape[0]), h))
    img2 = cv2.resize(img2, (int(h*img2.shape[1]/img2.shape[0]), h))

    comparison = np.hstack([img1, img2])
    cv2.imshow(f"Original vs Duplicate", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced duplicate image detector")
    parser.add_argument("image_folder", help="Folder containing images to scan")
    parser.add_argument("--output", help="Folder to move duplicates to (optional)")
    parser.add_argument("--hash_size", type=int, default=8, help="Perceptual hash size (8, 16, 32)")
    parser.add_argument("--threshold", type=int, default=5, help="Max hash difference (0-64)")
    parser.add_argument("--similarity", type=float, default=0.95, 
                       help="SSIM threshold (0.9-0.99)")
    parser.add_argument("--move", action="store_true", help="Move instead of delete")
    parser.add_argument("--preview", action="store_true", help="Preview before removal")

    args = parser.parse_args()

    find_duplicates(
        image_folder=args.image_folder,
        output_folder=args.output,
        hash_size=args.hash_size,
        threshold=args.threshold,
        similarity_threshold=args.similarity,
        move_duplicates=args.move,
        preview=args.preview
    )
