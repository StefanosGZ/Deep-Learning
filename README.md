Sure! Here is an improved `README.md` for your repository:

---

# Deep Learning Course - IST Lisbon

This repository contains the lab work and homework assignments completed by me and a colleague for the Deep Learning course at IST Lisbon. The primary coding language used in the assignments is Python, and the main library utilized for deep learning tasks is PyTorch.

## Homework 1

### Files:
- **DL_hmw1.pdf**: Submitted report on the homework.
- **homework1.pdf**: Homework instructions.
- **hw1-q1.py**: First task - Medical image classification using linear classifiers and neural networks.
- **hw1-q2.py**: Second task - Medical image classification using an autodiff toolkit.
- **utils.py**: Utility classes for data handling and preprocessing.

### Description:
In Homework 1, we explored:
1. Logistic Regression and Feedforward Neural Networks for medical image classification.
2. Implementing and training models using PyTorch, optimizing with SGD and Adam optimizers.
3. Evaluating models and visualizing training/validation losses and accuracies.

## Homework 2

### Files:
- **DL_hmw2.pdf**: Submitted report on the homework.
- **homework2.pdf**: Homework instructions.
- **hw2-q2.py**: Second task - Image classification using Convolutional Neural Networks (CNNs).
- **hw2-q3.ipynb**: Third task - Automatic Speech Recognition using sequence-to-sequence models.
- **utils.py**: Utility classes for data handling and transformations.

### Description:
In Homework 2, we focused on:
1. Designing and implementing CNN architectures for image classification.
2. Experimenting with dropout and max-pooling layers.
3. Developing an automatic speech recognition system using sequence-to-sequence models with RNN and Transformer-based decoders.
4. Analyzing model performance through various similarity metrics.

### Example Workflow:

#### Training & Testing with CNNs (hw2-q2.py):
```bash
python hw2-q2.py -epochs 20 -batch_size 8 -learning_rate 0.01 -dropout 0.7 -optimizer adam
```

#### Running ASR task (hw2-q3.ipynb):
1. Open the Jupyter notebook and follow the instructions to train and evaluate the ASR models.
2. Ensure all dependencies are installed, e.g., `ml_collections`, `textdistance`, etc.

## Plotting Results

```python
# main.py (Run for plotting training and validation metrics)

if __name__ == "__main__":
    config = get_config()
    config.unlock()
    # Customize your config details as needed
    config.lock()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {DEVICE}")

    trainer = Trainer(config, DEVICE)
    trainer.train()
    trainer.test()
    plotting(config)
```

Explore the plots generated during training and evaluation of models under `./storage/<config_name>/<seed>/plots`.

I hope this provides a clear and structured overview of your work for the Deep Learning course!
