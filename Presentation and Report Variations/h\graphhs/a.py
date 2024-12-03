import matplotlib.pyplot as plt

# Sample data for perplexity over 10 epochs
epochs = list(range(1, 11))
perplexity = [51, 42, 35, 30, 24, 22, 20, 18, 16, 14.2]

# Plotting the perplexity over epochs
plt.figure(figsize=(10, 6))
plt.plot(epochs, perplexity, marker='o', linestyle='-', color='blue')
plt.title('Perplexity Over Training Epochs')
plt.xlabel('Epochs')
plt.ylabel('Perplexity')
plt.xticks(epochs)
plt.grid(True)
plt.show()



import matplotlib.pyplot as plt

# Data
aspects = ['Ease of Use', 'Helpfulness of Content', 'Response Time', 'Interface Design', 'Offline Functionality']
ratings = [3.5, 4.5, 4.5, 3.0, 4.0]

# Plotting the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(aspects, ratings, color='skyblue')
plt.title('User Satisfaction Ratings')
plt.ylabel('Average Rating (out of 5)')
plt.ylim(0, 5)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adding the rating values on top of each bar
for bar, rating in zip(bars, ratings):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{rating}', ha='center', va='bottom')

plt.show()




import matplotlib.pyplot as plt

# Data
subjects = ['Mathematics', 'Science', 'Bangla', 'Social Studies']
accuracy = [30, 82, 92, 86]

# Plotting the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(subjects, accuracy, color='green')
plt.title('Subject-wise Accuracy')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adding the accuracy values on top of each bar
for bar, acc in zip(bars, accuracy):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{acc}%', ha='center', va='bottom')

plt.show()





import matplotlib.pyplot as plt

# Data
models = ['Pathshala.ai', 'BanglaBERT', 'mBERT', 'DistilBERT', 'GPT-2 Bangla']
accuracy = [70, 88, 82, 80, 78]

# Plotting the bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(models, accuracy, color='orange')
plt.title('Accuracy Comparison of Language Models')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adding accuracy values on top of each bar
for bar, acc in zip(bars, accuracy):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{acc}%', ha='center', va='bottom')

plt.show()




import matplotlib.pyplot as plt

# Data
models = ['Pathshala.ai', 'BanglaBERT', 'mBERT', 'DistilBERT', 'GPT-2 Bangla']
inference_time = [1.7, 5.5, 6.0, 3.0, 7.0]

# Plotting the bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(models, inference_time, color='red')
plt.title('Inference Time Comparison of Language Models')
plt.ylabel('Inference Time (seconds)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adding inference time values on top of each bar
for bar, time in zip(bars, inference_time):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{time}s', ha='center', va='bottom')

plt.show()





import matplotlib.pyplot as plt

# Data
models = ['Pathshala.ai', 'BanglaBERT', 'mBERT', 'DistilBERT', 'GPT-2 Bangla']
model_size = [120, 420, 680, 250, 500]

# Plotting the bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(models, model_size, color='purple')
plt.title('Model Size Comparison of Language Models')
plt.ylabel('Model Size (MB)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adding model size values on top of each bar
for bar, size in zip(bars, model_size):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 10, f'{size} MB', ha='center', va='bottom')

plt.show()






import numpy as np
import matplotlib.pyplot as plt

# Data
models = ['Pathshala.ai', 'BanglaBERT', 'mBERT', 'DistilBERT', 'GPT-2 Bangla']
accuracy = [70, 88, 82, 80, 78]
perplexity = [14.2, 12.5, 15.8, 17.3, 16.5]
inference_time = [1.7, 5.5, 6.0, 3.0, 7.0]
model_size = [120, 420, 680, 250, 500]
memory_usage = [140, 800, 1100, 400, 900]

# Convert lists to numpy arrays for easier handling
indices = np.arange(len(models))
width = 0.15  # Width of each bar

# Plotting the grouped bar chart
plt.figure(figsize=(14, 8))

plt.bar(indices - 2*width, accuracy, width=width, label='Accuracy (%)')
plt.bar(indices - width, perplexity, width=width, label='Perplexity')
plt.bar(indices, inference_time, width=width, label='Inference Time (s)')
plt.bar(indices + width, model_size, width=width, label='Model Size (MB)')
plt.bar(indices + 2*width, memory_usage, width=width, label='Memory Usage (MB)')

plt.title('Detailed Comparison Metrics of Language Models')
plt.xlabel('Models')
plt.xticks(indices, models)
plt.ylabel('Values')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()