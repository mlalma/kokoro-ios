//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

/// Duration encoder that processes text embeddings for phoneme duration prediction.
///
/// This encoder transforms BERT-encoded phoneme features into representations
/// suitable for predicting how long each phoneme should be pronounced.
///
/// Architecture:
/// - Alternating stack of bidirectional LSTM and AdaLayerNorm layers
/// - Style conditioning is applied throughout the network
/// - Each LSTM layer is followed by an adaptive layer normalization
///
/// The encoder outputs features that are then used by a duration predictor
/// to determine the temporal alignment of phonemes in the generated speech.
final class DurationEncoder: Module {
  /// Stack of LSTM and AdaLayerNorm layers that alternate
  /// Even indices: Bidirectional LSTM layers
  /// Odd indices: Adaptive Layer Normalization layers
  @ModuleInfo var lstms: [Module]

  /// Initializes the duration encoder with pretrained weights.
  ///
  /// Builds a stack of alternating LSTM and AdaLayerNorm layers.
  /// The structure is: LSTM → AdaLN → LSTM → AdaLN → ... (repeated nlayers times)
  ///
  /// - Parameters:
  ///   - weights: Dictionary of pretrained model weights
  ///   - dModel: Model dimension size (hidden size of features)
  ///   - styDim: Style embedding dimension (for speaker/prosody conditioning)
  ///   - nlayers: Number of LSTM/AdaLN layer pairs to create
  init(weights: [String: MLXArray], dModel: Int, styDim: Int, nlayers: Int) {
    var lstmsArr: [Module] = []

    // Build nlayers pairs of (LSTM, AdaLayerNorm)
    for i in 0 ..< nlayers * 2 {
      // Even indices: Create bidirectional LSTM layers
      if i % 2 == 0 {
        lstmsArr.append(
          LSTM(
            inputSize: dModel + styDim,  // Input includes both features and style
            hiddenSize: dModel / 2,       // Half size because bidirectional
            wxForward: weights["predictor.text_encoder.lstms.\(i).weight_ih_l0"]!,
            whForward: weights["predictor.text_encoder.lstms.\(i).weight_hh_l0"]!,
            biasIhForward: weights["predictor.text_encoder.lstms.\(i).bias_ih_l0"]!,
            biasHhForward: weights["predictor.text_encoder.lstms.\(i).bias_hh_l0"]!,
            wxBackward: weights["predictor.text_encoder.lstms.\(i).weight_ih_l0_reverse"]!,
            whBackward: weights["predictor.text_encoder.lstms.\(i).weight_hh_l0_reverse"]!,
            biasIhBackward: weights["predictor.text_encoder.lstms.\(i).bias_ih_l0_reverse"]!,
            biasHhBackward: weights["predictor.text_encoder.lstms.\(i).bias_hh_l0_reverse"]!
          )
        )
      // Odd indices: Create adaptive layer normalization
      } else {
        lstmsArr.append(
          AdaLayerNorm(
            weight: weights["predictor.text_encoder.lstms.\(i).fc.weight"]!,
            bias: weights["predictor.text_encoder.lstms.\(i).fc.bias"]!
          )
        )
      }
    }

    self._lstms.wrappedValue = lstmsArr
  }

  /// Forward pass. Encodes input features with style conditioning for duration prediction.
  ///
  /// Processing pipeline:
  /// 1. Concatenate input features with style embeddings
  /// 2. Apply masking to ignore padding positions
  /// 3. Process through alternating LSTM and AdaLayerNorm layers
  /// 4. Each layer maintains style conditioning
  ///
  /// - Parameters:
  ///   - x: Input features from BERT encoder [batch, seq_len, dModel]
  ///   - style: Style embeddings [batch, styDim]
  ///   - textLengths: Length of each sequence (unused but kept for interface)
  ///   - m: Mask indicating padding positions [batch, seq_len]
  /// - Returns: Duration-encoded features [batch, dModel, seq_len]
  func callAsFunction(_ x: MLXArray, style: MLXArray, textLengths _: MLXArray, m: MLXArray) -> MLXArray {
    // Step 1: Transpose to [seq_len, batch, dModel] for processing
    var x = x.transposed(2, 0, 1)

    // Step 2: Broadcast style to match sequence length [seq_len, batch, styDim]
    let s = MLX.broadcast(style, to: [x.shape[0], x.shape[1], style.shape[style.shape.count - 1]])

    // Step 3: Concatenate features with style [seq_len, batch, dModel + styDim]
    x = MLX.concatenated([x, s], axis: -1)

    // Step 4: Apply mask to zero out padding positions
    x = MLX.where(m.expandedDimensions(axes: [-1]).transposed(1, 0, 2), MLXArray.zeros(like: x), x)

    // Step 5: Transpose to [batch, dModel + styDim, seq_len] for layer processing
    x = x.transposed(1, 2, 0)

    // Step 6: Process through alternating LSTM and AdaLayerNorm layers
    for block in lstms {
      // Handle Adaptive Layer Normalization blocks
      if let adaLayerNorm = block as? AdaLayerNorm {
        // Apply AdaLN with style conditioning
        // Transpose to [batch, seq_len, features] for AdaLN
        x = adaLayerNorm(x.transposed(0, 2, 1), style).transposed(0, 2, 1)

        // Re-concatenate with style for next layer
        x = MLX.concatenated([x, s.transposed(1, 2, 0)], axis: 1)

        // Reapply mask to maintain padding
        x = MLX.where(m.expandedDimensions(axes: [-1]).transposed(0, 2, 1), MLXArray.zeros(like: x), x)

      // Handle LSTM blocks
      } else if let lstm = block as? LSTM {
        // Transpose to [batch, seq_len, features] for LSTM and extract first batch
        x = x.transposed(0, 2, 1)[0]

        // Process through bidirectional LSTM
        let (lstmOutput, _) = lstm(x)

        // Transpose back to [batch, features, seq_len]
        x = lstmOutput.transposed(0, 2, 1)

        // Pad output to match original sequence length if needed
        let xPad = MLXArray.zeros([x.shape[0], x.shape[1], m.shape[m.shape.count - 1]])
        xPad[0 ..< x.shape[0], 0 ..< x.shape[1], 0 ..< x.shape[2]] = x
        x = xPad
      }
    }

    // Step 7: Final transpose to [batch, seq_len, features] and return
    return x.transposed(0, 2, 1)
  }
}
