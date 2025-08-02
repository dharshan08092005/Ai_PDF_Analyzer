# PDF Bot API

A FastAPI-based PDF document analysis service that uses Gemini AI and FAISS for efficient document processing and question answering.

## Features

- PDF text extraction and chunking
- Vector-based similarity search using FAISS
- AI-powered question answering with Google Gemini
- High-performance concurrent processing
- Caching for improved response times
- Authentication support
- Health monitoring

## Docker Deployment

### Local Docker Testing

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

2. **Or build and run manually:**
   ```bash
   # Build the image
   docker build -t pdf-bot-api .
   
   # Run the container
   docker run -p 8000:8000 \
     -e GEMINI_API_KEY=your_gemini_api_key \
     -e AUTHORIZE_TOKEN=your_auth_token \
     pdf-bot-api
   ```

### Hugging Face Docker Deployment

1. **Push to Hugging Face:**
   ```bash
   # Login to Hugging Face
   docker login registry.hf.space
   
   # Tag your image
   docker tag pdf-bot-api:latest registry.hf.space/your-username/your-space-name:latest
   
   # Push to Hugging Face
   docker push registry.hf.space/your-username/your-space-name:latest
   ```

2. **Configure in Hugging Face Space:**
   - Create a new Space in Hugging Face
   - Choose "Docker" as the SDK
   - Set the Docker image to: `registry.hf.space/your-username/your-space-name:latest`
   - Add environment variables:
     - `GEMINI_API_KEY`: Your Google Gemini API key
     - `AUTHORIZE_TOKEN`: Your authentication token

## Environment Variables

- `GEMINI_API_KEY`: Google Gemini API key (required)
- `AUTHORIZE_TOKEN`: Authentication token for API access (required)
- `PORT`: Port to run the server on (default: 8000)

## API Usage

### Authentication

All API endpoints require Bearer token authentication:
```
Authorization: Bearer your_auth_token
```

### Endpoints

#### Health Check
```bash
GET /health
```

#### Process PDF and Answer Questions
```bash
POST /hackrx/run
Content-Type: application/json
Authorization: Bearer your_auth_token

{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the main topic?",
    "What are the key points?",
    "What is the conclusion?"
  ]
}
```

### Example Usage

```python
import requests

url = "https://your-hf-space-url.hf.space/hackrx/run"
headers = {
    "Authorization": "Bearer your_auth_token",
    "Content-Type": "application/json"
}

data = {
    "documents": "https://example.com/sample.pdf",
    "questions": [
        "What is the document about?",
        "What are the main requirements?",
        "What is the deadline?"
    ]
}

response = requests.post(url, json=data, headers=headers)
answers = response.json()["answers"]
print(answers)
```

## Performance Optimization

- Uses FAISS for fast vector similarity search
- Implements caching for documents and Q&A pairs
- Optimized concurrent processing for multiple questions
- HTTP/2 support for better performance
- Adaptive thread pooling based on CPU cores

## Health Monitoring

The application includes health checks and monitoring:
- `/health` endpoint for status monitoring
- Docker health checks
- Cache size monitoring
- Performance logging

## Troubleshooting

1. **Memory Issues**: The application is optimized for CPU usage. For large documents, consider increasing container memory limits.

2. **Timeout Issues**: The API is designed for 15-second response times. For very large documents or many questions, processing may take longer.

3. **Authentication Errors**: Ensure your `AUTHORIZE_TOKEN` is correctly set and included in requests.

4. **API Key Issues**: Verify your `GEMINI_API_KEY` is valid and has sufficient quota.

## Development

### Local Development Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables:
   ```bash
   export GEMINI_API_KEY=your_key
   export AUTHORIZE_TOKEN=your_token
   ```

3. Run the application:
   ```bash
   python main.py
   ```

### Testing

Run tests with:
```bash
pytest test_*.py
```

## License

This project is licensed under the MIT License. 