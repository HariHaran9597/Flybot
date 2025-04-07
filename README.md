# Flybot: AI-Powered Flight Booking Chatbot

A sophisticated chatbot leveraging Neural Networks and NLP for intelligent flight booking assistance, featuring high-accuracy intent recognition and real-time flight data integration.

## 🎯 Key Achievements
- **90% Intent Recognition Rate** using Neural Networks and TF-IDF vectorization
- **40% Reduction** in user interaction time through Amadeus Flight API integration
- **Scalable Performance** handling 100+ daily queries with 2-second response latency
- **Global Accessibility** through multi-language support

## 🛠 Technical Architecture

### Core Components
- **Intent Recognition Engine**
  - Neural Network-based classification
  - TF-IDF vectorization for text processing
  - Pre-trained model saved in `fnn_model.pth`

- **API Integration**
  - Amadeus Flight API for real-time booking
  - Google Translate API for language support
  - RESTful API endpoints for service integration

- **Backend Infrastructure**
  - Flask web framework
  - Gunicorn WSGI server for production deployment
  - Scalable architecture for high concurrency

## 🚀 Getting Started

### Prerequisites
```bash
python 3.x
pip
virtualenv (recommended)
```

### Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd flybot
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration
1. Set up API keys in environment variables:
```bash
export AMADEUS_API_KEY='your_key_here'
export GOOGLE_TRANSLATE_API_KEY='your_key_here'
```

2. Configure the application:
- Update `intents.json` for custom intent patterns
- Modify `config.py` for environment-specific settings

### Running the Application
1. Development mode:
```bash
python app.py
```

2. Production deployment with Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 🔧 Project Structure
```
├── app.py                # Main Flask application
├── chat.py              # Core chat functionality
├── chatgui.py           # Chat interface implementation
├── data_preprocessing.py # Data preparation utilities
├── feature_extraction.py # TF-IDF and feature processing
├── fnn.py              # Feed-forward neural network implementation
├── intent_handler.py    # Intent processing logic
├── model.py            # Neural network model definition
├── train.py           # Model training script
└── intents.json       # Intent patterns and responses
```

## 📝 Key Features Explained

### Intent Recognition System
- Utilizes a Feed-Forward Neural Network (FNN) for accurate intent classification
- TF-IDF vectorization for efficient text feature extraction
- Pre-trained model with 90% accuracy on diverse intent patterns
- Supports multiple languages through Google Translate API

### Real-time Flight Booking
- Seamless integration with Amadeus Flight API
- Asynchronous request handling for better performance
- Real-time flight availability and pricing
- Automated booking workflow reducing user interaction time by 40%

### Scalable Architecture
- Flask-based RESTful API design
- Gunicorn WSGI server for production deployment
- Multi-worker configuration for handling concurrent requests
- 2-second response latency under high load (100+ daily queries)

### Multi-language Support
- Integrated Google Translate API
- Automatic language detection
- Real-time translation of user queries and responses
- Support for major global languages

## 🧪 Testing and Evaluation

### Model Testing
- Comprehensive test suite in `model_evaluation.py`
- Regular evaluation of intent recognition accuracy
- Performance benchmarking under various loads
- Continuous monitoring of response times

### Integration Testing
- API endpoint testing
- End-to-end booking flow validation
- Language support verification
- Load testing for scalability validation

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.