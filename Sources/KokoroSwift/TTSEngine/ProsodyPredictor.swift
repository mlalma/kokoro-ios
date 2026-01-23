//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

/// Prosody predictor that generates pitch and voicing parameters for natural speech.
///
/// Based on the StyleTTS2 architecture, this module predicts:
/// - **F0 (Fundamental Frequency)**: The pitch contour of the speech
/// - **N (Voicing)**: Characteristics related to voiced/unvoiced sounds
///
/// The predictor uses:
/// 1. A shared bidirectional LSTM to process input features
/// 2. Separate branches with AdaIN residual blocks for F0 and N prediction
/// 3. Projection layers to output final single-channel predictions
///
/// These prosodic features are essential for generating natural-sounding,
/// expressive speech with proper intonation and rhythm.
final class ProsodyPredictor: Module {
  /// Shared bidirectional LSTM for processing input features
  /// Captures temporal dependencies before branching into F0 and N predictions
  @ModuleInfo var shared: LSTM

  /// Stack of AdaIN residual blocks for F0 (pitch) prediction
  /// Includes upsampling to match the target temporal resolution
  @ModuleInfo var F0: [AdainResBlk1d]

  /// Stack of AdaIN residual blocks for N (voicing) prediction
  /// Parallel to F0 branch with similar architecture
  @ModuleInfo var N: [AdainResBlk1d]

  /// Projection layer to convert F0 features to single-channel output
  @ModuleInfo(key: "F0_proj") var F0Proj: Conv1dInference

  /// Projection layer to convert N features to single-channel output
  @ModuleInfo(key: "N_proj") var NProj: Conv1dInference

  /// Initializes the prosody predictor with pretrained weights.
  /// - Parameters:
  ///   - weights: Dictionary of pretrained model weights
  ///   - styleDim: Dimension of style embeddings for AdaIN conditioning
  ///   - dHid: Hidden dimension size for internal representations
  public init(weights: [String: MLXArray], styleDim: Int, dHid: Int) {
    // Initialize shared bidirectional LSTM
    // Processes concatenated hidden features and style embeddings
    self._shared.wrappedValue = LSTM(
      inputSize: dHid + styleDim,
      hiddenSize: dHid / 2,  // Half size because bidirectional (forward + backward)
      wxForward: weights["predictor.shared.weight_ih_l0"]!,
      whForward: weights["predictor.shared.weight_hh_l0"]!,
      biasIhForward: weights["predictor.shared.bias_ih_l0"]!,
      biasHhForward: weights["predictor.shared.bias_hh_l0"]!,
      wxBackward: weights["predictor.shared.weight_ih_l0_reverse"]!,
      whBackward: weights["predictor.shared.weight_hh_l0_reverse"]!,
      biasIhBackward: weights["predictor.shared.bias_ih_l0_reverse"]!,
      biasHhBackward: weights["predictor.shared.bias_hh_l0_reverse"]!
    )

    // Initialize F0 (pitch) prediction branch
    // Three residual blocks: maintain -> upsample -> refine
    self._F0.wrappedValue = [
      // Block 0: Process features at original resolution
      AdainResBlk1d(weights: weights, weightKeyPrefix: "predictor.F0.0", dimIn: dHid, dimOut: dHid, styleDim: styleDim),
      // Block 1: Upsample and reduce dimensions
      AdainResBlk1d(weights: weights, weightKeyPrefix: "predictor.F0.1", dimIn: dHid, dimOut: dHid / 2, styleDim: styleDim, upsample: "true"),
      // Block 2: Refine at higher temporal resolution
      AdainResBlk1d(weights: weights, weightKeyPrefix: "predictor.F0.2", dimIn: dHid / 2, dimOut: dHid / 2, styleDim: styleDim),
    ]

    // Initialize N (voicing) prediction branch
    // Parallel structure to F0 branch
    self._N.wrappedValue = [
      // Block 0: Process features at original resolution
      AdainResBlk1d(weights: weights, weightKeyPrefix: "predictor.N.0", dimIn: dHid, dimOut: dHid, styleDim: styleDim),
      // Block 1: Upsample and reduce dimensions
      AdainResBlk1d(weights: weights, weightKeyPrefix: "predictor.N.1", dimIn: dHid, dimOut: dHid / 2, styleDim: styleDim, upsample: "true"),
      // Block 2: Refine at higher temporal resolution
      AdainResBlk1d(weights: weights, weightKeyPrefix: "predictor.N.2", dimIn: dHid / 2, dimOut: dHid / 2, styleDim: styleDim),
    ]

    // Initialize F0 projection layer (multi-channel -> single channel)
    self._F0Proj.wrappedValue = Conv1dInference(
      inputChannels: dHid / 2,
      outputChannels: 1,
      kernelSize: 1,  // 1x1 convolution for channel reduction
      padding: 0,
      weight: weights["predictor.F0_proj.weight"]!,
      bias: weights["predictor.F0_proj.bias"]!
    )

    // Initialize N projection layer (multi-channel -> single channel)
    self._NProj.wrappedValue = Conv1dInference(
      inputChannels: dHid / 2,
      outputChannels: 1,
      kernelSize: 1,  // 1x1 convolution for channel reduction
      padding: 0,
      weight: weights["predictor.N_proj.weight"]!,
      bias: weights["predictor.N_proj.bias"]!
    )
  }

  /// Forward pass. Predicts F0 (pitch) and N (voicing) curves from input features.
  ///
  /// The prediction pipeline:
  /// 1. Process input through shared LSTM to capture temporal context
  /// 2. Branch into parallel F0 and N prediction paths
  /// 3. Each path processes through AdaIN residual blocks with style conditioning
  /// 4. Project to single-channel outputs
  ///
  /// - Parameters:
  ///   - x: Input features [batch, seq_len, channels]
  ///   - s: Style embeddings for AdaIN conditioning [batch, style_dim]
  /// - Returns: Tuple of (F0 curve, N curve), each with shape [batch, time_steps]
  func F0NTrain(x: MLXArray, s: MLXArray) -> (MLXArray, MLXArray) {
    // Step 1: Process through shared LSTM
    // Transpose to [batch, seq_len, channels] for LSTM
    let (x1, _) = shared(x.transposed(0, 2, 1))

    // Step 2: F0 (pitch) prediction branch
    // Transpose to [batch, channels, seq_len] for convolutions
    var F0Val = x1.transposed(0, 2, 1)

    // Process through AdaIN residual blocks with style conditioning
    for block in F0 {
      F0Val = block(x: F0Val, s: s)
    }

    // Swap axes for projection: [batch, seq_len, channels]
    F0Val = MLX.swappedAxes(F0Val, 2, 1)
    // Project to single channel
    F0Val = F0Proj(F0Val)
    // Swap back: [batch, channels, seq_len]
    F0Val = MLX.swappedAxes(F0Val, 2, 1)

    // Step 3: N (voicing) prediction branch
    // Transpose to [batch, channels, seq_len] for convolutions
    var NVal = x1.transposed(0, 2, 1)

    // Process through AdaIN residual blocks with style conditioning
    for block in N {
      NVal = block(x: NVal, s: s)
    }

    // Swap axes for projection: [batch, seq_len, channels]
    NVal = MLX.swappedAxes(NVal, 2, 1)
    // Project to single channel
    NVal = NProj(NVal)
    // Swap back: [batch, channels, seq_len]
    NVal = MLX.swappedAxes(NVal, 2, 1)

    // Remove the channel dimension (size 1) and return [batch, time_steps]
    return (F0Val.squeezed(axis: 1), NVal.squeezed(axis: 1))
  }
}
