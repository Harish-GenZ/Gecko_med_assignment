import sys
sys.path.insert(0, 'backend')
from PIL import Image
from app.services.ocr.ocr_engine import ocr_image
from app.services.preprocessing.image_preprocessor import preprocess

img_path = 'raw_datasets/UHSR/UHSR/WhatsApp Image 2026-09-26 at 06.32.03.jpeg'
img = Image.open(img_path)

print('Running OCR on RAW image...')
res_raw = ocr_image(img)
raw_len = len(res_raw['text'])
raw_conf = res_raw['confidence']
print(f'RAW OCR: {raw_len} chars, conf: {raw_conf:.3f}')
print('--- RAW OCR FIRST 500 CHARS ---')
print(res_raw['text'][:500])
print('\n' + '='*50 + '\n')

print('Running OCR on PREPROCESSED image...')
preprocessed = preprocess(img)
res_prep = ocr_image(preprocessed)
prep_len = len(res_prep['text'])
prep_conf = res_prep['confidence']
print(f'PREPROCESSED OCR: {prep_len} chars, conf: {prep_conf:.3f}')
print('--- PREPROCESSED OCR FIRST 500 CHARS ---')
print(res_prep['text'][:500])
