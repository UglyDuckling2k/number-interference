import pytest
from app import app as flask_app, load_model, DigitClassifier

@pytest.fixture
def app():
    yield flask_app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def model():
    model = DigitClassifier()
    model.eval()  # Set the model to evaluation mode
    return model

def test_digit_classifier_structure(model):
    # Test the structure of the CNN model
    assert hasattr(model, 'conv1'), "Model should have a conv1 layer"
    assert hasattr(model, 'fc2'), "Model should have an fc2 layer"
    # Add more assertions as needed to test the structure thoroughly