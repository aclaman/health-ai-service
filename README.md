# Health AI Service

A lightweight REST API that classifies short health-related text inputs into one of three categories:
- `low_concern`
- `needs_follow_up`
- `urgent_review`

## How it works

The service loads a pre-trained sentence embedding model (`all-MiniLM-L6-v2`). When a text input is received, it is converted into a vector and compared to a set of reference phrases for each category using cosine similarity. The category with the highest similarity score is returned as the classification.

## Model

[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) – a lightweight, freely available sentence embedding model that runs fully locally without any external API calls.

## How to build and run

**Build the Docker image:**
```bash
docker build -t health-ai-service .
```

**Run the container:**
```bash
docker run --rm -p 8000:8000 health-ai-service
```

## API Endpoints

### GET /health
Returns whether the service is running.

**Example:**
```bash
curl http://localhost:8000/health
```
**Response:**
```json
{"status": "ok"}
```

### POST /analyze
Receives a text input and returns a classification.

**Example 1 – urgent:**
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I have chest pain and shortness of breath."}'
```
**Response:**
```json
{"label": "urgent_review", "confidence": 1.0}
```

**Example 2 – low concern:**
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel great today, no complaints."}'
```
**Response:**
```json
{"label": "low_concern", "confidence": 0.85}
```

## Interactive API documentation

FastAPI provides a built-in visual interface. With the service running, open:
```
http://localhost:8000/docs
```