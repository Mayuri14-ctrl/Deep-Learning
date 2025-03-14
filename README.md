# Deep-Learning

### What are the main components of a neural network?
Inout Layer
Hidden Layer
Output Layer
Weights & biases
Activation function

### What is the purpose of an activation function
It introduces non linearity to learn complex patterns
Sigmoid: Used for probabilities, but suffers from vanishing gradient
TanH: Used in hidden layers, zero-centered
Relu
Leaky Relu
Softmax: Used in multi-class classification

### Why do we prefer ReLU over Sigmoid and Tanh?
To avoid vanishing gradient problem

### What is backpropagation?
Backpropagation is used to train a neural network by minimizing the loss function. During forward propagation, data flows from the input to the output layer, making predictions using initial random weights and biases. The loss is then calculated by comparing predictions with actual values. Using the chain rule, the gradient of the loss is computed, and the error propagates backward layer by layer. Weights are updated using gradient descent to minimize the error. This process repeats until the model converges

### What is the vanishing gradient problem?
The vanishing gradient problem occurs when gradients become too small during backpropagation, especially in deep networks. This happens because activation functions like Sigmoid or Tanh squash values between a limited range, causing gradients to shrink as they pass through multiple layers. As a result, earlier layers learn very slowly, making training inefficient.

Solution: use RelU or Leaky Relu
Batch normalization
Use Residual Connections (Skip Connections) in deep networks.
### Optimization Techniques
### What is gradient descent?
Gradient Descent is an optimization algorithm used to minimize the loss function. It updated the weights in the direction of negative gradient
w new=w old−η ∂w/∂L
​, where 𝜂 η is the learning rate.
Types:
Batch Gradient Descent: Uses the entire dataset for each update.
Stochastic Gradient Descent (SGD): Updates weights for each sample (noisy but faster).
Mini-Batch Gradient Descent: Uses a small batch (most commonly used).

### What are the differences between Adam, RMSProp, and SGD?

### What is batch normalization?
Batch Normalization normalizes inputs to each layer, ensuring stable learning.

# Q1 Why do deep networks generalize better than shallow networks?
✔ Answer:
Deep networks learn hierarchical features—low-level features in early layers and high-level abstractions in later layers. This allows them to capture complex patterns that shallow networks cannot, leading to better generalization.

Q3. Why does Batch Normalization improve training stability?
✔ Answer:

Reduces internal covariate shift (normalizes feature distributions).
Acts as a form of regularization (reducing dependence on dropout).
Allows for larger learning rates, speeding up convergence.
🚀 Follow-up: Can BN be applied to small batch sizes?
✔ Use Group Normalization when batch size is small.

Q4. What are the pros and cons of using Adam vs. SGD?
✔ Answer:

Optimizer	Pros	Cons
Adam	Fast convergence, works well for sparse data	May not generalize well, sensitive to hyperparameters
SGD	Generalizes better, stable	Slower convergence, needs fine-tuned learning rate
🚀 Follow-up: How would you adapt Adam for better generalization?
✔ Use AdamW (Adam with weight decay) for better regularization.

3️⃣ Deep Learning Model Deployment & Scalability
Q5. How would you deploy a deep learning model for real-time inference?
✔ Answer:

Convert model to TensorRT, ONNX, or TorchScript for speed optimization.
Use model quantization (e.g., INT8 instead of FP32).
Deploy using FastAPI, Flask, or Triton Inference Server.
Optimize inference with batching, caching, and load balancing.
🚀 Follow-up: What are the latency vs. throughput trade-offs in deployment?

Q6. How do you handle model drift in a production deep learning system?
✔ Answer:

Monitor data drift using tools like EvidentlyAI.
Use active learning to retrain models on fresh data.
Implement shadow deployments before rolling out updates.
Compare feature distributions over time to detect changes.
4️⃣ Explainability & Interpretability in Deep Learning
Q7. How do you interpret a deep learning model's predictions?
✔ Answer:

SHAP (Shapley Additive Explanations) → Feature importance.
LIME (Local Interpretable Model-agnostic Explanations) → Local explanations.
Grad-CAM → For CNN-based models in computer vision.
Integrated Gradients → For NLP and structured data.
🚀 Follow-up: When would you use Grad-CAM over SHAP?
✔ Grad-CAM is for CNNs and image-based models, while SHAP is more general.

5️⃣ Transfer Learning & Model Fine-Tuning
Q8. How would you fine-tune a pre-trained deep learning model for a custom dataset?
✔ Answer:

Freeze early layers, retrain later layers.
Use discriminative learning rates (lower LR for early layers, higher for later layers).
Apply data augmentation to avoid overfitting.
Use mixed-precision training for speedup.
🚀 Follow-up: Why would you use learning rate schedulers like cosine annealing?
✔ To avoid sharp minima and improve convergence.

6️⃣ Advanced Deep Learning Topics
Q9. What is the role of Self-Attention in Transformer models?
✔ Answer:

Allows models to focus on relevant parts of the input sequence.
Eliminates the need for recurrence (faster than RNNs).
Used in BERT, GPT, ViTs for NLP and vision tasks.
🚀 Follow-up: Why does Transformer scaling lead to better performance?
✔ Larger models generalize better with sufficient data.

Q10. What is contrastive learning, and where is it used?
✔ Answer:

Self-supervised learning technique (e.g., SimCLR, MoCo).
Learns representations by maximizing similarity between positive pairs and minimizing similarity between negative pairs.
Used in unsupervised feature learning (e.g., vision and NLP).
🚀 Follow-up: How does contrastive learning compare to traditional supervised learning?

7️⃣ Case Study Questions
🔹 Q11. You are working on an autonomous vehicle system. How would you improve object detection accuracy?
✔ Answer:

Use YOLO/Faster R-CNN with data augmentation (brightness, rotation).
Apply multi-task learning (jointly predict bounding boxes + segmentation).
Use hard-negative mining to improve detection of rare classes.
🚀 Follow-up: How would you optimize inference speed?
✔ Convert to TensorRT or prune model layers.

🔹 Q12. How would you design a deep learning system for fraud detection in financial transactions?
✔ Answer:

Use Graph Neural Networks (GNNs) for transaction relationships.
Apply autoencoders for anomaly detection.
Use XGBoost on top of deep embeddings for tabular fraud detection.
🚀 Follow-up: How do you prevent adversarial attacks in fraud detection models?
✔ Use adversarial training and robust feature engineering.

8️⃣ Deep Learning in NLP & Vision
Q13. What is the difference between BERT and GPT models?
✔ Answer:

Model	Architecture	Training Objective	Use Case
BERT	Bidirectional	Masked Language Model (MLM)	Sentence classification, Q&A
GPT	Unidirectional	Causal Language Model (CLM)	Text generation
🚀 Follow-up: Why is GPT better for text generation?
✔ GPT is trained in a causal way, making it better at predicting the next word.

Q14. What are Vision Transformers (ViTs), and how do they compare to CNNs?
✔ Answer:

ViTs divide images into patches and use self-attention instead of convolution.
Better at long-range dependencies than CNNs.
Requires more data than CNNs to generalize well.
🚀 Follow-up: How would you fine-tune a ViT model for medical image classification?
