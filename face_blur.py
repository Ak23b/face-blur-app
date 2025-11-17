"""
face_blur.py
Simple script to detect faces and blur or pixelate them in images.
Usage examples:
  pyhton face_blur.py --input images/photo.jpg --output out.jpg --mode blur --level 25
  python face_blur.py --input_folder images/ --output_folder out/ --mode pixelate --level 10
"""

import cv2
import numpy as np
import os
from PIL import Image
import argparse

def load_face_cascade():
  # OpenCV ships with Haar Cascade files accessible via cv2.data.haarcascades
  cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
  if not os.path.exists(cascade_path):
    raise RuntimeError(f"Haar cascade not found at {cascade_path}")
  return cv2.CascadeClassifier(cascade_path)

def blur_region(img, x, y, w, h, level=25):
  """Apply Gaussian blur to the ROI. Level controls kernel size (odd int)."""
  roi = img[y:y+h, x:x+w]
  # Compute kernel size from level (must be odd)
  k = max(1, int(level)// 2 * 2 + 1)
  blurred = cv2.GaussianBlur(roi,(k,k), 0)
  img[y:y+y+h, x:x+w] = blurred
  return img

def pixelate_region(img, x, y, w, h, level=10):
  """Pixelate the region by resizing down and up."""
  roi = img[y:y+h, x:x+w]
  # ensure level is at least 1
  level = max(1, int(level))
  
  #downscale
  small = cv2.resize(roi, (max(1, w//level), max(1, h//level)), interpolation=cv2.INTER_LINEAR)
  # Upscale back to original size
  pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
  img[y:y+h, x:x+w] = pixelated
  return img


def detect_faces(gray_img, cascade, scaleFactor=1.1, minNeighbors=5, minSize=(30,30)):
  faces = cascade.detectMultiScale(gray_img, scaleFactor=scaleFactor,minNeighbors=minNeighbors, minSize=minSize)
  return faces

def process_image(path_in, path_out, mode='blur', level=25, draw_rect=False):
  img = cv2.imread(path_in)
  if img is None:
    raise RuntimeError(f"Could not read image: {path_in}")
  orig = img.copy()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  cascade = load_face_cascade()
  faces = detect_faces(gray, cascade)
  for (x, y, w, h) in faces:
    if mode == 'blur':
      img = blur_region(img, x, y, w, h, level=level)
    elif mode == 'pixelate':
      img = pixelate_region(img, x, y, w, h, level=level)
    else:
      raise ValueError("mode must be 'blur' or 'pixelate'")
    if draw_rect:
      cv2.rectangle(img,(x, y), (x+w, y+h), (0, 255, 0), 2)
      
  # ensure output dir exists
  os.makedirs(os.path.dirname(path_out) or '.', exist_ok=True)
  # Write the result
  cv2.imencode('.jpg', img)[1].tofile(path_out) # safer for unique code paths on windows
  return len(faces)

def process_folder(input_folder, output_folder, **kwargs):
  os.makedirs(output_folder, exist_ok=True)
  supported = ('.jpg','.jpeg','.png','.bmp','.tiff','.webp')
  count = 0
  for fname in os.listdir(input_folder):
    if fname.lower().endswith(supported):
      in_path = os.path.join(input_folder, fname)
      out_path = os.path.join(output_folder, fname)
      faces = process_image(in_path, out_path, **kwargs)
      print(f"Processed {fname}: {faces} faces")
      count += 1
  return count
def main():
  parser = argparse.ArgumentParser(description="Detect and blur/pixelate faces in images.")
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('--input', help="Input image path")
  group.add_argument('--input_folder', help="Input folder with images.")
  parser.add_argument('--output', help="Output image path (for single input)")
  parser.add_argument('--output_folder', help="Output folder (for folder input)")
  parser.add_argument('--mode', choices=['blur','pixelate'], default='blur')
  parser.add_argument('--level',type=int, default=25, help="Blur strength or pixelation level")
  parser.add_argument('--draw_rect', action='store_true', help="Draw rectangle around detected faces for debugging")
  args = parser.parse_args()
  
  if args.input:
    out = args.output or (os.path.splittext(args.input)[0] + "_blurred.jpg")
    faces = process_image(args.input, out, mode=args.mode, level=args.level, draw_rect=args.draw_rect)
    print(f"Saved{out} - faces found: {faces}")
  else:
    out_folder = args.output_folder or (args.input_folder.rstrip('/\\') + "_out")
    processed = process_folder(args.input_folder, out_folder, mode=args.mode, level=args.level, draw_rect=args.draw_rect)
    print(f"Processed {processed} images. Output in: {out_folder}")
    
if __name__ == "__main__":
  main()
  

