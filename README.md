# ShopSmart

## Project Summary

ShopSmart uses computer vision and AI to help consumers track and compare prices across various stores. By simply uploading or taking a photo of an item with its price tag, ShopSmart detects the product, identifies its category, extracts the price, and stores this information in a database. This enables easy price comparison for shoppers.

---

## 🎯 Minimum Viable Product (MVP) Checklist

1. Upload (or take) a photo
   - 1 prompt for item
   - 1 prompt for price tag
2. Do image processing for item
   - Run it through YoloV8 for bounding boxes
     - 1 bounding box per image based on distance from image center, and size of bounding box
   - Crop image based on bounding box
   - Compute pHash based on cropped image
   - Save pHash and cropped image into database (sql)
3. Do image processing for price tag
   - Run it through YoloV8 for bounding boxes
     - 1 bounding box per image based on distance from image center, and size of bounding box
   - Crop image based on bounding box
   - Extract number from image (Tesseract)
   - Save number and cropped image into database (sql)
4. Save results to the same column in a database

### MVP Pipeline

## ✅ First Steps: Web Demo

- Learn OpenCV, and Tesseract for image processing and text extraction
- Learn YOLOv8 for object detection and fine-tuning models
- Learn basics of SQLite for data storage

---

## ✅ Next Steps: Web Demo

- Learn Flask or FastAPI to build a web interface

---

## ✅ Future Plans: Mobile App

- Learn Flutter or React Native for mobile development

---
