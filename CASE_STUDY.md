# Classifying Vehicle Damage from Photographs: Building a CNN for Insurance Claim Verification

## Summary

I designed and trained a convolutional neural network to classify six types of vehicle damage from photographs, working through the full pipeline from data exploration to regularisation experiments, hyperparameter tuning, and test-set prediction. The project forced me to think carefully about class imbalance, generalisation, and the practical gap between training accuracy and real-world reliability.

## Context and Motivation

Insurance companies process thousands of damage claims daily. Each claim includes photographs that a human assessor must inspect, categorise, and verify. This process is slow, subjective, and inconsistent. Two assessors looking at the same scratched bumper might classify the damage differently, and the backlog grows faster than people can work through it.

The question I wanted to explore was straightforward but not simple: can a convolutional neural network learn to distinguish between crack, scratch, tire flat, dent, glass shatter, and broken damage from images alone? And if so, how much architectural and training discipline does it take to get the model to generalise rather than memorise?

## Problem Definition

This was a six-class image classification task. Given a photograph of a damaged vehicle, the model needed to output which category of damage was present. The training set was labelled but imbalanced, with some damage types appearing far more frequently than others. The test set was unlabelled, so final evaluation depended on validation performance and careful analysis of training dynamics.

## Why the Problem Mattered

Beyond the obvious business case for automating claim triage, this problem sits at an interesting intersection of computer vision challenges. Vehicle damage is visually subtle. A scratch and a crack can look similar at certain angles. Dents change appearance dramatically depending on lighting. Glass shatter patterns vary widely. The model needed to learn features that were genuinely discriminative, not just surface-level texture patterns that happened to correlate with labels in the training set.

## My Role and Contribution

I was solely responsible for the entire pipeline: data exploration, preprocessing, architecture design, training, regularisation, hyperparameter tuning, overfitting analysis, and final evaluation. Every design choice was mine to justify.

## Approach and Methodology

I structured the work as a sequence of deliberate experiments, each building on what the previous one revealed.

**Data exploration** came first. I examined the class distribution and found meaningful imbalance across the six damage types. I visualised sample images from each class to develop intuition about what the model would need to distinguish. This step matters more than people often acknowledge. Looking at the data before writing any model code changed how I thought about augmentation and architecture.

**Preprocessing** involved resizing all images to 224x224x3, a resolution that balances spatial detail against computational cost. I used stratified train-validation splits to preserve class proportions, which was essential given the imbalance. Training images were augmented on the fly with rotation, shifts, zoom, and horizontal flips through Keras ImageDataGenerator.

**The baseline CNN** used three convolutional blocks with 3x3 kernels, batch normalisation, and max-pooling, followed by a dense layer with ReLU activation and a softmax output. I deliberately kept this architecture simple. The goal was to establish a reference point before adding complexity, not to throw everything at the wall and hope something stuck.

**Regularisation experiments** introduced dropout after both convolutional and dense blocks, L2 weight decay on selected layers, and the data augmentation already built into the training pipeline. I trained this version for 20 epochs with early stopping (patience of 6) to let the regularisation effects manifest. The key observation was that dropout and L2 decay reduced the gap between training and validation accuracy, even when peak training accuracy dropped slightly. That tradeoff was exactly what I wanted.

**Hyperparameter tuning** explored combinations of learning rate (1e-3, 5e-4, 1e-4), batch size (16, 32), and base filter count (32, 64) for the first convolutional layer. I used validation accuracy to select the best configuration. This was a lightweight grid search, constrained by available compute, but it was enough to identify meaningful differences. The winning configuration was then used to train the final model.

**Overfitting analysis** compared training and validation curves across epochs. I looked for the classic signature: training accuracy climbing while validation accuracy plateaus or declines, and a growing gap between training and validation loss. The regularisation strategies and early stopping kept this under control, though I could see the model beginning to memorise toward later epochs.

## Data, Models, Systems, and Tools

- Python, TensorFlow/Keras, scikit-learn, NumPy, Pandas, Matplotlib, Seaborn, PIL
- ImageDataGenerator for augmentation and streaming from disk
- Computed class weights using scikit-learn to address imbalance
- EarlyStopping callback with best-weight restoration
- Confusion matrix and classification report for per-class evaluation

## Key Technical Decisions and Why They Were Made

**Class weights over resampling.** I used computed class weights to penalise misclassification of rare damage types more heavily rather than oversampling minority classes. Oversampling can introduce subtle duplication artifacts in image data, and class weighting achieves a similar effect without modifying the data distribution.

**Simple baseline first.** Starting with a minimal architecture and only adding regularisation after observing baseline behaviour gave me a clear picture of what each modification contributed. This is a habit I consider essential for trustworthy model development.

**Early stopping with weight restoration.** Rather than training for a fixed number of epochs and hoping for the best, I let the model train until validation loss stopped improving, then restored the weights from the best epoch. This is a small decision that makes a large practical difference.

**224x224 input resolution.** Large enough to preserve the visual details that distinguish damage types, small enough to keep training times reasonable. This is one of those decisions that seems obvious in retrospect but required thinking about the specific characteristics of vehicle damage images.

## Challenges, Tradeoffs, and Constraints

The most significant challenge was the class imbalance. Some damage types had substantially more training examples than others, which meant the model could achieve reasonable overall accuracy by performing well on majority classes while failing on minority ones. Class weights helped, but the per-class performance gap never disappeared entirely.

Compute constraints limited the depth of hyperparameter search. With more resources, I would have explored deeper architectures, transfer learning from ImageNet-pretrained models, and a wider range of augmentation strategies. The grid I used was sufficient to demonstrate the principle but not exhaustive.

The absence of test labels meant I could not compute a final test accuracy. I had to rely on validation performance as my best estimate of generalisation. This is common in real-world settings, but it means the reported numbers carry some uncertainty.

## What I Learned

The most lasting lesson was about the relationship between model complexity and useful generalisation. Adding more layers or parameters does not automatically improve a model. What matters is whether the added capacity lets the model learn features that transfer to unseen data. Regularisation is not just a technique to apply mechanically; it is a way of encoding the belief that the model should not fit the training data too closely.

I also gained a sharper appreciation for systematic experimentation. Running each modification as a distinct experiment with a clear baseline comparison made it possible to attribute improvements (or regressions) to specific decisions. This is the same logic that underlies scientific experimentation, and it transfers directly to research.

## Outcome

The final regularised and tuned CNN showed improved generalisation over the baseline, with a reduced gap between training and validation metrics. Per-class performance was strongest on visually distinctive damage types and weaker where categories overlapped visually. The complete pipeline, from data loading through to test-set prediction export, was implemented in a single reproducible notebook.

## Why This Work Matters for AI and Research Readiness

This project required the kind of iterative, evidence-based thinking that defines applied machine learning research. I did not just build a model; I systematically investigated how architectural and training choices affected the model's ability to generalise. The work demonstrates comfort with deep learning fundamentals, experimental methodology, and the discipline of evaluating models honestly rather than chasing headline accuracy numbers.

## What I Would Investigate Next

Transfer learning is the obvious next step. Fine-tuning a ResNet or EfficientNet pretrained on ImageNet would likely outperform my custom architecture significantly, and the comparison would be informative. I would also explore Grad-CAM or similar interpretability methods to understand which image regions the model attends to when making classifications. For insurance applications, being able to explain why a model assigned a particular damage category matters as much as the classification itself.
