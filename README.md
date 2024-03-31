# indieLLM
experimental small LLM trained on Indian scriptures

Model Specifications

| Specification       | Value            | Description                                                                 |
|---------------------|------------------|-----------------------------------------------------------------------------|
| **Model Name**      | IndieLLM         | A language model based on the GPT architecture tailored for text generation. |
| **Version**         | 1.0              | Initial version of the model.                                               |
| **Model Size**      | 35.15M parameters| Total number of trainable parameters in the model, indicating its complexity. |
| **Vocabulary Size** | 50,000 tokens    | The number of unique tokens that the model can recognize.                    |
| **Framework**       | PyTorch          | The deep learning framework used to implement and train the model.          |
| **Batch Size**      | 16               | Number of training samples processed before the model's internal parameters are updated. |
| **Block Size**      | 128 tokens       | The maximum length of the input sequence the model can handle.               |
| **Learning Rate**   | 0.001            | The step size at each iteration while moving toward a minimum of a loss function. |
| **Optimizer**       | AdamW            | Optimization algorithm used for minimizing the training loss.                |
| **Device**          | CUDA/CPU         | The computing device used for training and inference, GPU if available.      |
| **Embedding Size**  | 256              | Dimensionality of the token embeddings used in the model.                    |
| **Number of Heads** | 16               | The number of heads in the multi-head attention mechanism, affecting the model's ability to focus on different parts of the input sequence. |
| **Number of Layers**| 12               | The number of transformer blocks in the model, impacting its depth and potential for understanding complex dependencies. |
| **Dropout**         | 0.1              | Probability of dropping out a neuron, a regularization technique to prevent overfitting. |
| **Training Iterations** | 20,000        | Total number of iterations to train the model.                              |
| **Evaluation Interval** | 1,000 iterations | Frequency at which the model is evaluated on the validation set during training. |
