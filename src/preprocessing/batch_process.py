import cv2, os, pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing, threading
from .roi_cropper import crop_black_border
from .enhancement import apply_clahe
from .circular_mask import apply_circular_mask

cv2.setNumThreads(0)
failed_images = []
lock = threading.Lock()

def xu_ly_mot_anh(args):
    img_path, save_path = args
    try:
        img = cv2.imread(img_path)
        if img is None: return False
        img = crop_black_border(img)
        img = apply_circular_mask(img)
        img = apply_clahe(img)
        img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_AREA)
        cv2.imwrite(save_path, img)
        return True
    except Exception as e:
        with lock: failed_images.append((img_path, str(e)))
        return False

def run_preprocessing(base_dir, out_dir):
    for split in ["train", "val", "test"]:
        os.makedirs(f"{out_dir}/{split}", exist_ok=True)
        csv_path = f"{base_dir}/{split}/{split}.csv"
        img_dir = f"{base_dir}/{split}/images"
        if not os.path.exists(img_dir): img_dir = f"{base_dir}/{split}"
        
        df = pd.read_csv(csv_path)
        tasks = [(os.path.join(img_dir, f"{r['image_id']}.jpg"), os.path.join(out_dir, split, f"{r['image_id']}.jpg")) 
                 for _, r in df.iterrows() if not os.path.exists(os.path.join(out_dir, split, f"{r['image_id']}.jpg"))]
        
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            list(tqdm(executor.map(xu_ly_mot_anh, tasks), total=len(tasks)))
