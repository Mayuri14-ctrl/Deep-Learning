# Deep-Learning

### What are the main components of a neural network?
It is a densely packed network of neurons where input is passed through input layer and hidden layer to predict outcome.
Weights and biases gets updated using backpropogation to minimize the loss function

- Input Layer
-Hidden Layer
-Output Layer
-Weights & biases
-Activation function

### What is the difference between a perceptron and a multilayer perceptron (MLP)?
A perceptron is single layer (input -> output). It is used for Linear classification as it can only learn linear separable problems
Multi layer perceptron consists of multiple perceptron in layers with ability to learn linear patterns

### What is the purpose of an activation function
It introduces non linearity to learn complex patterns
Sigmoid: Used for probabilities, but suffers from vanishing gradient
https://www.researchgate.net/publication/350103066/figure/fig1/AS:1002064838684673@1615922283491/Sigmoid-graph-Source.png
TanH: Used in hidden layers, zero-centered
Relu
Leaky Relu
Softmax: Used in multi-class classification


### Why do we prefer ReLU over Sigmoid and Tanh?
To avoid vanishing gradient problem

### What is backpropagation?
Backpropagation is used to train a neural network by minimizing the loss function. During forward propagation, data flows from the input to the output layer, making predictions using initial random weights and biases. The loss is then calculated by comparing predictions with actual values. Using the chain rule, the gradient of the loss is computed, and the error propagates backward layer by layer. Weights are updated using gradient descent to minimize the error. This process repeats until the model converges

### What is the vanishing gradient problem?
The vanishing gradient problem occurs when gradients become too small during backpropagation, especially in deep networks.This leads to extremely slow learning in the earlier layers of the network because the weights get very small updates.  This happens because activation functions like Sigmoid or Tanh squash values between a limited range, causing gradients to shrink as they pass through multiple layers. As a result, earlier layers learn very slowly, making training inefficient.

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

### Why do deep networks generalize better than shallow networks?
Deep networks learn hierarchical features—low-level features in early layers and high-level abstractions in later layers. This allows them to capture complex patterns that shallow networks cannot, leading to better generalization.

### Why does Batch Normalization improve training stability?

Reduces internal covariate shift (normalizes feature distributions).
Acts as a form of regularization (reducing dependence on dropout).
Allows for larger learning rates, speeding up convergence.
🚀 Follow-up: Can BN be applied to small batch sizes?
✔ Use Group Normalization when batch size is small.

### What are the pros and cons of using Adam vs. SGD?

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

## What is a Recurrent Neural Network (RNN)?
It is used for processing Sequential data, RNNs maintain a hidden state that acts as memory. 
Inputs : A sequence of data points is fed into the network.
Hidden States : The network maintains memory across all layers.
Output : The model produces an output at layers.
Weight Sharing: The same weights are used at each layer, making it effective for sequential tasks.
![image](https://github.com/user-attachments/assets/90229840-f62e-4e85-b6e8-deff6480a30d)
### What is the vanishing gradient problem in RNNs? How do you solve it?

During backpropagation, gradients become very small (vanish) as they move back through time.
This prevents RNNs from learning long-term dependencies.

Use LSTMs/GRUs: These architectures have gates that help retain long-term dependencies.
Use ReLU instead of tanh/sigmoid: ReLU activation mitigates vanishing gradients.
Gradient Clipping: Limits the gradient values to prevent vanishing/exploding.
Batch Normalization: Helps stabilize gradient updates.

### What are the different types of RNN architectures?
Many-to-One: One output per entire sequence (e.g., sentiment analysis).
One-to-Many: One input, multiple outputs (e.g., image captioning).
Many-to-Many: Multiple inputs and outputs (e.g., machine translation, video classification).
Bidirectional RNN (Bi-RNN): Uses both past and future information (e.g., Named Entity Recognition).
LSTMs & GRUs: Advanced RNNs that solve vanishing gradient issues.

### What is an LSTM? How does it solve RNN limitations?
It has three gates to regulate information flow:
Forget Gate – Decides what to forget from past memory.
Input Gate – Decides what new information to store.
Output Gate – Decides what part of memory to output.

### What is a GRU, and how is it different from LSTM?
It has only two:
Reset Gate: Decides how much past information to forget.
Update Gate: Controls how much new memory to store.

### What is the Transformer model? How does it work?
The Transformer architecture is a deep learning model that processes input sequences in parallel, unlike RNNs. It uses positional encoding to retain order information and self-attention mechanisms to weigh the importance of different words in a sequence. The model consists of multiple encoder-decoder layers and a feed-forward network to learn complex dependencies efficiently.
Parallel Processing – Unlike RNNs, Transformers do not rely on sequential data processing, making them faster.
Self-Attention Mechanism – Computes relationships between words in a sequence to understand context.
Multi-Head Attention – Uses multiple attention mechanisms to capture different contextual features.
Position Encoding – Since Transformers don’t have a fixed sequence order like RNNs, they use positional embeddings.
Feedforward Networks – Fully connected layers process the transformed representations.

### How does the self-attention mechanism work in Transformers?
Query, Key, and Value (Q, K, V) Matrices
Each input word is transformed into three vectors (Query, Key, and Value) using learned weight matrices.
Compute Attention Scores
The dot product between Query and Key vectors determines the relevance of each word.
This is divided by 𝑑𝑘(scaling factor) to stabilize gradients. ​
Apply Softmax: Converts attention scores into probabilities.
Multiply by Value (V) Matrix: Generates the new word representation.
\[
\text{Attention}(Q, K, V) = \text{Softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
\]

### What is the role of Self-Attention in Transformer models?
Allows models to focus on relevant parts of the input sequence.
Eliminates the need for recurrence (faster than RNNs).
Used in BERT, GPT, ViTs for NLP and vision tasks.
🚀 Follow-up: Why does Transformer scaling lead to better performance?
✔ Larger models generalize better with sufficient data.

### Difference Between Self-Attention and Multi-Head Attention
Self-Attention allows a model to focus on different words in a sentence while processing each word.
📌 It helps in capturing relationships between words, even if they are far apart in a sequence.

👉 Steps in Self-Attention:

Convert Input into Query (Q), Key (K), and Value (V) Matrices.
Compute Attention Scores:
Take the dot product of Query (Q) and Key (K).
Divide by 𝑑𝑘 (scaling factor) to stabilize gradients.
Apply Softmax to get attention weights.
Multiply with Value (V) to get the final output.

Multi-Head Attention (MHA) applies multiple self-attention mechanisms in parallel.
📌 Each attention head learns different types of relationships (e.g., syntax, meaning, long-term dependencies).
👉 Steps in Multi-Head Attention:
Create multiple sets of Query (Q), Key (K), and Value (V) matrices with different learned weights.
Apply self-attention to each set independently.
Concatenate outputs of all heads.
Pass the result through a linear transformation.

### What is BERT and how does it work?
Answer:Encoder-only
📌 BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained Transformer model developed by Google.
📌 Unlike traditional Transformers, BERT uses bidirectional attention, meaning it looks at both left and right context.
👉 Key Features:
Masked Language Model (MLM) – Some words in a sentence are randomly masked, and BERT learns to predict them.
Next Sentence Prediction (NSP) – Determines whether two sentences appear sequentially.

### What is GPT ?
GPT (Generative Pre-trained Transformer) is a decoder-only transformer model that generates text using next-word prediction. It is autoregressive, meaning it predicts each word based on previously generated words. 

### How do you fine-tune a Transformer model?
Choose a Pre-trained Model – Load models like BERT, GPT, or T5 from Hugging Face.
Prepare the Dataset – Tokenize input using Tokenizer.
Modify Model Layers – Add task-specific layers (classification, regression, QA).
Use a Suitable Loss Function – Cross-entropy for classification.
Train with GPU Optimization – Use AdamW optimizer and learning rate schedulin

### What is contrastive learning, and where is it used?
Self-supervised learning technique (e.g., SimCLR, MoCo).
Learns representations by maximizing similarity between positive pairs and minimizing similarity between negative pairs.
Used in unsupervised feature learning (e.g., vision and NLP).

### Follow-up: How does contrastive learning compare to traditional supervised learning?
Contrastive learning learns representations by bringing similar data points closer and pushing dissimilar ones apart, often without explicit labels. In contrast, traditional supervised learning relies on labeled data to minimize classification error.

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


Q14. What are Vision Transformers (ViTs), and how do they compare to CNNs?
✔ Answer:

ViTs divide images into patches and use self-attention instead of convolution.
Better at long-range dependencies than CNNs.
Requires more data than CNNs to generalize well.
🚀 Follow-up: How would you fine-tune a ViT model for medical image classification?

Q15. What is embedding?
An embedding is a way to represent high-dimensional data (like words, images, or categorical variables) in a lower-dimensional continuous vector space. It helps capture relationships and similarities between data points more efficiently.
Why Use Embeddings?
Reduce dimensionality while preserving meaningful relationships.
Improve computational efficiency.
Capture semantic meaning (e.g., similar words have similar vector representations in NLP).

Q16. Examples of Embeddings:
1. Word Embeddings (NLP)
In Natural Language Processing (NLP), embeddings convert words into numerical vectors. Examples:
Word2Vec (Google)
GloVe (Stanford)
FastText (Facebook)
Transformer-based embeddings (BERT, GPT, LLaMA)

2. Image Embeddings (Computer Vision)
Deep learning models (like CNNs) extract features from images and represent them as vectors.
Example: A face recognition system converts images of faces into embeddings and compares them for similarity.

3. Categorical Embeddings
In structured data, categorical variables (like "city" or "product type") are converted into numerical embeddings.

Q17 What is sentence embedding?
A sentence embedding is a numerical vector representation of an entire sentence. It captures the meaning of the sentence in a high-dimensional space, allowing models to understand semantic relationships between sentences.
1. Sentence-BERT (SBERT)
Fine-tuned BERT model to generate sentence-level embeddings.
Uses cosine similarity for comparing sentence meanings.
2.Universal Sentence Encoder (USE)
Developed by Google, optimized for semantic similarity and search.
Produces 512-dimensional embeddings.
3.OpenAI’s CLIP (for multimodal tasks)
Maps text and images into the same embedding space.
4. FastText and Doc2Vec

Older approaches, but still used in some NLP tasks.

## How will I deploy my model?
✅ Receives user input (natural language query)
✅ Calls OpenAI API to generate SQL queries
✅ Executes SQL queries on a database
✅ Returns results to the user


Create a FastAPI Server for Deployment
You can use FastAPI to serve the chatbot as an API.

✔ Reduce latency with caching, async queries, and optimized SQL
✔ Increase throughput with Nginx, Gunicorn, and Kubernetes
✔ Ensure scalability with Airflow-managed tasks
