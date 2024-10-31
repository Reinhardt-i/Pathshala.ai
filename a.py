import kagglehub

# Download latest version
path = kagglehub.dataset_download("furcifer/bangla-newspaper-dataset")

print("Path to dataset files:", path)