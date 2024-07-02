# Digit Classification Suite

This project is a comprehensive digit classification suite that includes a model application for digit classification, a serving application, and a web application for interaction. It utilizes a machine learning model trained on the MNIST dataset to classify handwritten digits.

## Components

- **Model Application**: Located in `model-app/`, this component is responsible for training the digit classification model.
- **Serving Application**: Located in `serve-app/`, this component serves the trained model via a REST API.
- **Web Application**: Located in `web-app/`, this component provides a user interface for interacting with the digit classifier through a web browser.

## Getting Started

To get started with this project, clone the repository and follow the setup instructions for each component.

### Prerequisites

- Docker
- Docker Compose
- Node.js (for the web application)

### Setup

1. **Model Application**: Navigate to `model-app/` and follow the instructions in the README to train the model.
2. **Serving Application**: Navigate to `serve-app/` and follow the instructions in the README to start the serving application.
3. **Web Application**: Navigate to `web-app/` and follow the instructions in the README to start the web application.

### Running the Suite

To run the entire suite, use Docker Compose from the root directory:

```sh
docker-compose up