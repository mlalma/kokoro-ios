//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

/// Text encoder that transforms tokenized phoneme sequences into contextual embeddings.
///
/// The encoder processes text through three stages:
/// 1. **Embedding layer**: Converts token IDs to dense vectors
/// 2. **CNN layers**: Extract local features with weight normalization and layer normalization
/// 3. **Bidirectional LSTM**: Captures long-range dependencies in both directions
///
/// The output embeddings are used by the decoder to generate speech aligned with the input text.
final class TextEncoder: Module {
  /// Embedding layer that converts token IDs to dense vectors
  @ModuleInfo var embedding: Embedding

  /// Stack of CNN blocks for local feature extraction
  /// Each block contains: [ConvWeighted, LayerNorm, Activation]
  @ModuleInfo var cnn: [[Module]]

  /// Bidirectional LSTM for capturing sequential dependencies
  @ModuleInfo var lstm: LSTM

  /// Initializes the text encoder with pretrained weights.
  /// - Parameters:
  ///   - weights: Dictionary of pretrained model weights
  ///   - channels: Number of channels in hidden layers
  ///   - kernelSize: Kernel size for convolutional layers
  ///   - depth: Number of CNN blocks to stack
  ///   - nSymbols: Size of the vocabulary (number of unique tokens)
  ///   - actv: Activation function (default: LeakyReLU with slope 0.2)
  init(weights: [String: MLXArray], channels: Int, kernelSize: Int, depth: Int, nSymbols _: Int, actv: Module = LeakyReLU(negativeSlope: 0.2)) {
    // Initialize embedding layer
    self._embedding.wrappedValue = Embedding(weight: weights["text_encoder.embedding.weight"]!)

    // Calculate padding to maintain sequence length
    let padding = (kernelSize - 1) / 2

    // Build CNN layers with weight normalization and layer normalization
    var cnnLayers: [[Module]] = []
    for i in 0 ..< depth {
      cnnLayers.append([
        // Weight-normalized convolution
        ConvWeighted(
          weightG: weights["text_encoder.cnn.\(i).0.weight_g"]!,
          weightV: weights["text_encoder.cnn.\(i).0.weight_v"]!,
          bias: weights["text_encoder.cnn.\(i).0.bias"]!,
          padding: padding
        ),
        // Layer normalization for stability
        LayerNormInference(
          weight: weights["text_encoder.cnn.\(i).1.gamma"]!,
          bias: weights["text_encoder.cnn.\(i).1.beta"]!
        ),
        // Activation function
        actv,
      ])
    }
    self._cnn.wrappedValue = cnnLayers

    // Initialize bidirectional LSTM
    self._lstm.wrappedValue = LSTM(
      inputSize: channels,
      hiddenSize: channels / 2,  // Half size because bidirectional (forward + backward)
      wxForward: weights["text_encoder.lstm.weight_ih_l0"]!,
      whForward: weights["text_encoder.lstm.weight_hh_l0"]!,
      biasIhForward: weights["text_encoder.lstm.bias_ih_l0"]!,
      biasHhForward: weights["text_encoder.lstm.bias_hh_l0"]!,
      wxBackward: weights["text_encoder.lstm.weight_ih_l0_reverse"]!,
      whBackward: weights["text_encoder.lstm.weight_hh_l0_reverse"]!,
      biasIhBackward: weights["text_encoder.lstm.bias_ih_l0_reverse"]!,
      biasHhBackward: weights["text_encoder.lstm.bias_hh_l0_reverse"]!
    )
  }

  /// Forward pass. Encodes input token sequences into contextual embeddings.
  ///
  /// The encoding pipeline:
  /// 1. Convert tokens to embeddings
  /// 2. Apply masking to ignore padding positions
  /// 3. Process through CNN blocks for local features
  /// 4. Process through bidirectional LSTM for sequential context
  /// 5. Apply final masking and return
  ///
  /// - Parameters:
  ///   - x: Input token IDs [batch_size, sequence_length]
  ///   - inputLengths: Length of each sequence (unused but kept for interface compatibility)
  ///   - m: Mask indicating padding positions [batch_size, sequence_length]
  /// - Returns: Encoded text features [batch_size, channels, sequence_length]
  public func callAsFunction(_ x: MLXArray, inputLengths _: MLXArray, m: MLXArray) -> MLXArray {
    // Step 1: Convert token IDs to embeddings [batch, seq_len, embed_dim]
    var x = embedding(x)

    // Transpose to [batch, embed_dim, seq_len] for CNN processing
    x = x.transposed(0, 2, 1)

    // Expand mask dimensions for broadcasting [batch, 1, seq_len]
    let mask = m.expandedDimensions(axis: 1)

    // Apply mask to zero out padding positions
    x = MLX.where(mask, 0.0, x)

    // Step 2: Process through CNN blocks
    for convBlock in cnn {
      for layer in convBlock {
        // Handle convolutional and normalization layers
        if layer is ConvWeighted || layer is LayerNormInference {
          // Swap axes to [batch, seq_len, channels] for processing
          x = MLX.swappedAxes(x, 2, 1)

          if let convWeighted = layer as? ConvWeighted {
            x = convWeighted(x, conv: MLX.conv1d)
          } else if let layer = layer as? LayerNormInference {
            x = layer(x)
          }

          // Swap back to [batch, channels, seq_len]
          x = MLX.swappedAxes(x, 2, 1)

        // Handle activation layers
        } else if let layer = layer as? LeakyReLU {
          x = layer(x)
        } else {
          fatalError("Unsupported layer type")
        }

        // Reapply mask after each layer to maintain padding
        x = MLX.where(mask, 0.0, x)
      }
    }

    // Step 3: Process through bidirectional LSTM
    // Transpose to [batch, seq_len, channels] for LSTM
    x = MLX.swappedAxes(x, 2, 1)
    let (lstmOutput, _) = lstm(x)
    // Transpose back to [batch, channels, seq_len]
    x = MLX.swappedAxes(lstmOutput, 2, 1)

    // Step 4: Ensure output has correct shape by padding if necessary
    let xPad = MLX.zeros([x.shape[0], x.shape[1], mask.shape[mask.shape.count - 1]])
    xPad._updateInternal(x)

    // Apply final mask and return
    return MLX.where(mask, 0.0, xPad)
  }
}
