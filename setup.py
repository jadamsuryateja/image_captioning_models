from setuptools import setup, find_packages

setup(
    name="image-captioning-app",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "flask>=3.0.0",
        "torch>=2.8.0",
        "torchvision>=0.19.0",
        "Pillow>=10.0.0",
        "transformers>=4.36.2",
        "sentence-transformers>=2.2.2",
        "gunicorn>=21.2.0",
        "werkzeug>=3.0.0",
    ],
    python_requires=">=3.11.8",
)