"""
Data Collection Script for Pakistani Politician Image Classification
Collects images for 16 politician classes using web crawlers
"""

import os
import cv2
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
from tqdm import tqdm

# 16 Pakistani Politicians
POLITICIANS = {
    "imran_khan": ["Imran Khan Pakistan PM face", "Imran Khan PTI"],
    "nawaz_sharif": ["Nawaz Sharif Pakistan PM", "Nawaz Sharif PML-N"],
    "asif_ali_zardari": ["Asif Ali Zardari Pakistan President", "Zardari PPP"],
    "bilawal_bhutto": ["Bilawal Bhutto Zardari Pakistan", "Bilawal PPP Chairman"],
    "shahbaz_sharif": ["Shahbaz Sharif Pakistan PM", "Shehbaz Sharif PML-N"],
    "maryam_nawaz": ["Maryam Nawaz Pakistan", "Maryam Nawaz PML-N"],
    "fazlur_rehman": ["Fazlur Rehman Pakistan", "Maulana Fazlur Rehman JUI"],
    "asfandyar_wali": ["Asfandyar Wali Khan Pakistan", "Asfandyar Wali ANP"],
    "altaf_hussain": ["Altaf Hussain MQM Pakistan", "Altaf Hussain London"],
    "chaudhry_shujaat": ["Chaudhry Shujaat Hussain Pakistan", "Shujaat Hussain PML-Q"],
    "pervez_musharraf": ["Pervez Musharraf Pakistan President", "General Musharraf"],
    "shehryar_afridi": ["Shehryar Afridi Pakistan PTI", "Shehryar Khan Afridi"],
    "khawaja_asif": ["Khawaja Asif Pakistan PML-N", "Khawaja Muhammad Asif"],
    "ahsan_iqbal": ["Ahsan Iqbal Pakistan PML-N", "Ahsan Iqbal Minister"],
    "barrister_gohar": ["Barrister Gohar Ali Khan PTI", "Gohar Ali Khan Pakistan"],
    "ahmed_sharif_chaudhry": ["Ahmed Sharif Chaudhry ISPR Pakistan", "DG ISPR Ahmed Sharif"]
}


def crawl_images(name, queries, max_per_query=60):
    """
    Crawl images for a politician using Bing and Google
    
    Args:
        name: Politician folder name
        queries: List of search queries
        max_per_query: Maximum images per query
    """
    out_dir = f"data/raw/{name}"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Collecting images for: {name}")
    print(f"{'='*60}")
    
    for idx, query in enumerate(queries):
        print(f"\nQuery {idx+1}/{len(queries)}: {query}")
        
        # Bing crawler
        try:
            bing_crawler = BingImageCrawler(
                storage={"root_dir": out_dir},
                downloader_threads=4
            )
            bing_crawler.crawl(keyword=query, max_num=max_per_query)
            print(f"  ✓ Bing crawl completed")
        except Exception as e:
            print(f"  ✗ Bing crawl failed: {e}")
        
        # Google crawler
        try:
            google_crawler = GoogleImageCrawler(
                storage={"root_dir": out_dir},
                downloader_threads=4
            )
            google_crawler.crawl(keyword=query, max_num=max_per_query//2)
            print(f"  ✓ Google crawl completed")
        except Exception as e:
            print(f"  ✗ Google crawl failed: {e}")


def filter_images_with_faces(data_dir="data/raw", min_face_ratio=0.15):
    """
    Filter images to keep only those with detectable faces
    
    Args:
        data_dir: Root directory containing class folders
        min_face_ratio: Minimum ratio of face area to image area
    """
    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    summary = {}
    
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        print(f"\nFiltering images for: {class_name}")
        
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        kept = 0
        removed = 0
        
        for img_file in tqdm(images, desc=f"  Processing {class_name}"):
            img_path = os.path.join(class_path, img_file)
            
            try:
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    os.remove(img_path)
                    removed += 1
                    continue
                
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
                
                # Check if face detected and meets size requirement
                if len(faces) == 0:
                    os.remove(img_path)
                    removed += 1
                    continue
                
                # Calculate face area ratio
                img_area = img.shape[0] * img.shape[1]
                face_area = sum([w * h for (x, y, w, h) in faces])
                face_ratio = face_area / img_area
                
                if face_ratio < min_face_ratio:
                    os.remove(img_path)
                    removed += 1
                else:
                    kept += 1
                    
            except Exception as e:
                print(f"    Error processing {img_file}: {e}")
                try:
                    os.remove(img_path)
                    removed += 1
                except:
                    pass
        
        summary[class_name] = kept
        print(f"  ✓ Kept: {kept} | Removed: {removed}")
    
    return summary


def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("PAKISTANI POLITICIAN IMAGE COLLECTION")
    print("="*60)
    
    # Step 1: Crawl images
    print("\n[STEP 1] Crawling images from web...")
    for politician_name, queries in POLITICIANS.items():
        crawl_images(politician_name, queries, max_per_query=60)
    
    # Step 2: Filter images with face detection
    print("\n" + "="*60)
    print("[STEP 2] Filtering images with face detection...")
    print("="*60)
    summary = filter_images_with_faces()
    
    # Step 3: Print summary report
    print("\n" + "="*60)
    print("COLLECTION SUMMARY")
    print("="*60)
    print(f"{'Class Name':<30} {'Image Count':>15}")
    print("-"*60)
    
    total_images = 0
    min_images = float('inf')
    max_images = 0
    
    for class_name, count in sorted(summary.items()):
        print(f"{class_name:<30} {count:>15}")
        total_images += count
        min_images = min(min_images, count)
        max_images = max(max_images, count)
    
    print("-"*60)
    print(f"{'TOTAL':<30} {total_images:>15}")
    print(f"{'AVERAGE':<30} {total_images/len(summary):>15.1f}")
    print(f"{'MIN':<30} {min_images:>15}")
    print(f"{'MAX':<30} {max_images:>15}")
    print("="*60)
    
    # Validation
    classes_below_threshold = [name for name, count in summary.items() if count < 80]
    if classes_below_threshold:
        print(f"\n⚠️  WARNING: {len(classes_below_threshold)} classes have < 80 images:")
        for name in classes_below_threshold:
            print(f"  - {name}: {summary[name]} images")
        print("\nConsider re-running collection for these classes.")
    else:
        print("\n✓ All classes have ≥80 images. Ready for next phase!")


if __name__ == "__main__":
    main()
