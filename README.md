# NeedleASR

NeedleASR is an enhanced version of Needle (Necessary element of deep learning), a deep learning framework developed as part of CMU's course 10-714 *Deep Learning Systems: Algorithms and Implementation* [[Link]](https://dlsyscourse.org). This framework is designed to support Automatic Speech Recognition (ASR) based on Connectionist Temporal Classification (CTC). 

[[Proposal]](./10714_Proposal.pdf) [[Report]](https://colab.research.google.com/drive/1lCXbd-8ypRbmNh6KKXV472l3L1J0hrAP?usp=sharing)

## Key Features

1. **NDArray Backend**:

   NeedleASR abstracts the array data in `NDArray` (N-Dimensional Array), which is similar to Numpy's `ndarray`. `NDArray` supports array operations on both CPU and CUDA backends. The `NDArray` structure supports operations such as dimension permutation, reshaping, and broadcasting. These operations are facilitated by its fields—`strides`, `shape`, and `offset`—which enable efficient manipulation of array without changing its compact layout. In addition, the `NDArray` backend also accelerates matrix multiplication through tiling and vectorization. 

2. **Automatic Differentiation**:

   NeedleASR includes support for the `Tensor` data structure, which extends `NDArray` by integrating operations and automatic differentiation capabilities. The `Tensor` in NeedleASR is designed to build and manage computational graphs dynamically, enabling the framework to automatically compute gradients along these graphs.

3. **Neural Network Modules**:

   <img src="assets/modules.png" alt="eend_der" style="zoom:50%;" />

   NeedleASR provides comprehensive support for modules commonly used in deep neural networks, including:

   - **Basic**: Linear, Flatten, Residual, Dropout.

   - **Activation**: ReLU, Tanh, Sigmoid.

   - **Normalization**: Batch Normalization, Layer Normalization. 

   - **Convolution**: 2D Convolution.

   - **Sequence**: Embedding, RNN Cell, RNN, LSTM Cell, LSTM.

   - **Transformer**: Transformer Encoder/Decoder, Multihead Attention Layer. 

   Beyond neural network layers, NeedleASR offers a range of features for model training:

   - **Initialization**: Xavier Uniform/Normal Initialization, Kaiming Uniform/Normal Initialization. 

   - **Optimization**: Stochastic Gradient Descent, Adam. 

   - **Data**: Dataset, Dataloader, Data Transforms.  

4. **CTC-Based ASR**:

   A key enhancement of NeedleASR over Needle is its support for CTC-based ASR with beam search decoding. NeedleASR supports CTC loss and its backpropagation, enabling align-free training between speech features and text transcripts. The CTC loss computation is implemented in the log domain to ensure numerical stability. Additionally, NeedleASR facilitates beam search decoding during inference stage. 

## Usage

We provide a detailed example in [Report](https://colab.research.google.com/drive/1lCXbd-8ypRbmNh6KKXV472l3L1J0hrAP?usp=sharing) for training a CTC-based Transformer ASR system. To train deep learning models with NeedleASR, follow these steps: 

1. **Installtion**: 

   Run the following commands to set up NeedleASR:

   ```bash
   git clone https://github.com/Kelton8Z/needleASR.git
   pip3 install pybind11 Levenshtein librosa soundfile
   cd /content/needleASR
   rm -rf build/
   make clean
   make
   ```

2. **Create a Training Script**:

   We provide an example training script for ASR systems in `apps/train_DEBUG.py`. In this script, users should:

   1. **Set the Device and Backend**:
      NeedleASR supports multiple backends, including `NDArray` on CPU or CUDA GPU, as well as `numpy.ndarray` on CPU. The device can be defined as:
      - `needle.cpu()` for `NDArray` on CPU,
      - `needle.cuda()` for `NDArray` on GPU, or
      - `needle.cpu_numpy()` for NumPy on CPU.
   2. **Define the Dataset, DataLoader, and Model**:
      Prepare the dataset and dataloader, and define the ASR model architecture.
   3. **Define the Training and Evaluation Processes**:
      Implement the training loop, evaluation logic, and necessary metrics for the model.

3. **Visualization with TensorBoard**: 

   The example script in `apps/train_DEBUG.py` also includes support for visualizing results using TensorBoard, making it easy to track training progress and performance metrics.

## Notes for Developers

If you want to add more features based on NeedleASR, there are a few notes you should remind: 

1. **Iterating Over `NDArray`**:
   
    Direct iteration over an `NDArray` is **not supported**: 
    
    ```python
    # This will not work:
    for a in A:  # A: NDArray
        ...
    ```
    Use iteration with an index range instead:
    ```python
    # Supported:
    for i in range(len(A)):  # A: NDArray
        ...
    ```
    
2. **Shape Requirements for New `NDArray`**:

    The `shape` must always be a **tuple**, even for a 1D vector.

    ```python
    # Correct:
    extended_symbols = array_api.full((2 * len(target) + 1,), self.blank, device=target.device)  # Note the comma
    
    # Incorrect:
    array_api.full(2 * len(target) + 1, self.blank, device=target.device)  # Missing tuple
    ```

3. **Behavior of `NDArray.__getitem__`**:

    The `__getitem__` operation creates an `NDArray` with the **same number of dimensions as the original**.

    ```python
    A = NDArray(shape=(3, 4))
    sliced = A[1, :3]  # Resulting shape: (1, 3), not (3,)
    ```

4. **Compact Operation Restrictions**:

    The `compact()` operation only supports **positive offsets**.

5. **Comparing `NDArray` with Scalars**:

    Direct comparison between an `NDArray` and a scalar is **not supported**:

    ```python
    a = NDArray([1.0])
    if a == 1.0:  # This will not work
        ...
    ```

    To compare, convert the `NDArray` to a scalar using `numpy()`:

    ```python
    if float(a.numpy()) == 1.0:  # Supported
        ...
    ```

## Contributors

The implementation of Needle was started from the assignments designed by Professor **Tianqi Chen** and **Zico Kolter** as a part of course 10-714 *Deep Learning Systems: Algorithms and Implementation* [[Link]](https://dlsyscourse.org). 

**Qingzheng Wang** provided the implementation of Needle and also collaborated with **Kelton Zhang** on the implementation of features for CTC-based ASR.

**Qingzheng Wang** is with CMU Information Networking Institute, and **Kelton Zhang** is with CMU Department of Electrical and Computer Engineering. 

[/10740_Proposal.pdf]: 
