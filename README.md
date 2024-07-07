# Digit Classification Suite

This project is a comprehensive digit classification suite that includes a model application for digit classification, a serving application, and a web application for interaction. It utilizes a machine learning model trained on the MNIST dataset to classify handwritten digits.

## Components

- **Model Application**: Located in [`model-app/`], this component is responsible for training the digit classification model.
- **Serving Application**: Located in [`serve-app/`], this component serves the trained model via a REST API.
- **Web Application**: Located in [`web-app/`], this component provides a user interface for interacting with the digit classifier through a web browser.

## Getting Started

To get started with this project, clone the repository and follow the setup instructions for each component.

### Prerequisites

- Docker
- Docker Compose
- Node.js (for the web application)

### Running the Suite

To run the entire suite, use Docker Compose from the root directory:

```sh
docker-compose up
```

### Visualizing Training with TensorBoard

The Model Application uses TensorBoard to visualize the training process, including metrics such as loss and accuracy over epochs. To view these visualizations:

1. Ensure that the Model Application has been run at least once to generate TensorBoard logs.
2. Open a web browser and go to the URL provided by TensorBoard, typically `http://localhost:6006`, to view the training visualizations.