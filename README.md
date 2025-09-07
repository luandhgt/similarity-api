# Image Similarity API

## 📋 Mô tả
API service để tìm kiếm ảnh tương đồng sử dụng FAISS và Places365 ResNet50 model. Tích hợp với Event Management System để phát hiện ảnh trùng lặp và đề xuất ảnh liên quan.

## 🎯 Tính năng chính
- **Vector Embedding**: Trích xuất features từ ảnh bằng Places365 ResNet50
- **Similarity Search**: Tìm ảnh tương đồng bằng FAISS với cosine similarity
- **PostgreSQL Integration**: Lưu trữ metadata và liên kết với Event Management DB
- **REST API**: Endpoints đơn giản cho Node.js integration
- **Real-time Processing**: Xử lý ảnh upload và search nhanh chóng

## 🛠️ Công nghệ sử dụng
- **Python 3.10+**
- **FAISS**: Facebook AI Similarity Search
- **Places365**: Pretrained ResNet50 model
- **FastAPI**: Modern web framework
- **PostgreSQL**: Database với psycopg2
- **OpenCV + Pillow**: Image processing

## 📦 Cài đặt

### 1. Setup Environment
```bash
# Tạo conda environment
conda create -n image-similarity-api python=3.10
conda activate image-similarity-api

# Install dependencies
conda install -c conda-forge faiss-cpu
pip install -r requirements.txt
```

### 2. Database Setup
```sql
-- Tạo table trong PostgreSQL
CREATE TABLE IF NOT EXISTS images (
  id SERIAL PRIMARY KEY,
  code UUID DEFAULT gen_random_uuid() UNIQUE,
  event_code UUID NOT NULL REFERENCES events(code) ON DELETE CASCADE,
  filename TEXT NOT NULL,
  file_path TEXT NOT NULL,
  file_size BIGINT,
  mime_type TEXT,
  width INTEGER,
  height INTEGER,
  faiss_index INTEGER,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_images_event_code ON images(event_code);
CREATE INDEX IF NOT EXISTS idx_images_faiss_index ON images(faiss_index);
```

### 3. Environment Variables
```bash
# Tạo file .env
DB_HOST=localhost
DB_NAME=event_management
DB_USER=your_username
DB_PASS=your_password
DB_PORT=5432
API_PORT=8000
```

## 🚀 Chạy ứng dụng

### Development
```bash
conda activate image-similarity-api
uvicorn main:app --reload --port 8000
```

### Production
```bash
conda activate image-similarity-api
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 📡 API Endpoints

### 1. Health Check
```bash
GET /
Response: {"status": "ready", "total_vectors": 1234}
```

### 2. Add Image
```bash
POST /add_image
Body: {
  "filename": "cat.jpg",
  "file_path": "/uploads/cat.jpg", 
  "vector": [0.1, 0.2, ...],  // 2048 dimensions
  "event_code": "uuid-here",
  "metadata": {"width": 800, "height": 600}
}
Response: {"id": 123, "faiss_index": 456, "message": "success"}
```

### 3. Search Similar
```bash
POST /search
Body: {
  "vector": [0.1, 0.2, ...],  // 2048 dimensions
  "top_k": 10,
  "filters": {"event_code": "uuid-here"}  // optional
}
Response: [
  {
    "id": 123,
    "filename": "similar1.jpg",
    "similarity_score": 0.95,
    "event_code": "uuid-here"
  }
]
```

### 4. Extract Features
```bash
POST /extract_features
Body: multipart/form-data với image file
Response: {
  "vector": [0.1, 0.2, ...],  // 2048 dimensions
  "processing_time": 0.5
}
```

### 5. Statistics
```bash
GET /stats
Response: {
  "total_vectors_faiss": 1234,
  "total_images_db": 1234,
  "index_size_mb": 50.2
}
```

## 🔗 Integration với Node.js

### Client Example
```javascript
const axios = require('axios');

class ImageSimilarityClient {
  constructor(baseURL = 'http://localhost:8000') {
    this.api = axios.create({ baseURL });
  }

  async searchSimilar(vector, topK = 10, filters = null) {
    const response = await this.api.post('/search', {
      vector, top_k: topK, filters
    });
    return response.data;
  }

  async addImage(imageData) {
    const response = await this.api.post('/add_image', imageData);
    return response.data;
  }
}

// Usage
const client = new ImageSimilarityClient();
const results = await client.searchSimilar(vectorArray, 5);
```

## 📁 Cấu trúc dự án
```
image-similarity-api/
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables
├── README.md              # Documentation
├── models/
│   ├── __init__.py
│   └── places365.py       # Places365 model wrapper
├── utils/
│   ├── __init__.py
│   ├── database.py        # PostgreSQL utilities
│   └── image_processing.py # Image preprocessing
├── data/
│   ├── image_index.faiss  # FAISS index file
│   └── resnet50_places365.pth.tar  # Model weights
└── tests/
    └── test_api.py        # Unit tests
```

## ⚡ Performance
- **Search Speed**: ~1-5ms cho 10K+ vectors
- **Memory Usage**: ~2GB RAM cho 100K vectors
- **Throughput**: 100+ requests/second
- **Vector Dimension**: 2048 (Places365 ResNet50)

## 🔧 Configuration

### Model Settings
- **Architecture**: ResNet50 pretrained on Places365
- **Input Size**: 224x224 pixels
- **Preprocessing**: Grayscale → Resize → Center pad
- **Feature Layer**: Average pooling before classification

### FAISS Settings
- **Index Type**: IndexFlatIP (Inner Product/Cosine)
- **Similarity Metric**: Cosine similarity
- **Auto-save**: Every 100 additions

## 🚨 Troubleshooting

### Common Issues
1. **Model download fails**: Check internet connection, retry
2. **Database connection error**: Verify PostgreSQL credentials in .env
3. **FAISS import error**: Reinstall with `conda install -c conda-forge faiss-cpu`
4. **Memory issues**: Use quantized index for large datasets

### Performance Tuning
- Use GPU version: `conda install -c conda-forge faiss-gpu`
- Optimize batch size for bulk operations
- Use IVF index for datasets > 1M vectors

## 📊 Monitoring

### Key Metrics
- API response times
- Search accuracy scores
- Memory usage trends
- Error rates by endpoint

### Logs Location
- Application logs: `/var/log/image-similarity-api/`
- FAISS operations: Console output
- Database queries: PostgreSQL logs

## 🛡️ Security
- Input validation for all endpoints
- File size limits for uploads
- SQL injection prevention
- Rate limiting (implement in production)

## 📈 Scaling
- **Horizontal**: Multiple API instances behind load balancer
- **Vertical**: Increase memory for larger FAISS index
- **Database**: Read replicas for search-heavy workloads
- **Storage**: Distributed file system for images

## 🔄 Backup & Recovery
- **FAISS Index**: Daily backup của .faiss files
- **Database**: Standard PostgreSQL backup
- **Model Files**: Version control cho model weights

## 👥 Team
- **Backend**: Integration với Event Management System
- **Infrastructure**: Docker deployment, monitoring
- **QA**: API testing, performance validation

## 📞 Support
- **Issues**: GitHub Issues
- **Documentation**: Wiki pages
- **Monitoring**: Application dashboard
- **Alerts**: Slack integration cho production

---

## 🚀 Quick Start
```bash
git clone <repo-url>
cd image-similarity-api
conda create -n image-similarity-api python=3.10
conda activate image-similarity-api
pip install -r requirements.txt
uvicorn main:app --reload
```

**Status**: ✅ Production Ready  
**Last Updated**: December 2024  
**Version**: 1.0.0