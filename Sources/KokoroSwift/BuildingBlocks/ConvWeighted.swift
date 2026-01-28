//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

/// Conv1d with weight normalization
/// Optimized: pre-computes normalized weights at initialization for better performance
class ConvWeighted: Module {
  /// Pre-computed normalized weight (computed once at init)
  private let normalizedWeight: MLXArray
  /// Pre-computed transposed normalized weight for cases where transpose is needed
  private let normalizedWeightTransposed: MLXArray
  /// Reshaped bias for broadcasting
  private let reshapedBias: MLXArray?

  let stride: Int
  let padding: Int
  let dilation: Int
  let outputPadding: Int
  let groups: Int

  init(
    weightG: MLXArray,
    weightV: MLXArray,
    bias: MLXArray?,
    stride: Int = 1,
    padding: Int = 1,
    dilation: Int = 1,
    outputPadding: Int = 0,
    groups: Int = 1
  ) {
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.outputPadding = outputPadding
    self.groups = groups

    // Pre-compute normalized weight at initialization (instead of every forward pass)
    self.normalizedWeight = Self.computeWeightNorm(weightV: weightV, weightG: weightG)
    self.normalizedWeightTransposed = self.normalizedWeight.transposed()

    // Pre-reshape bias for broadcasting
    self.reshapedBias = bias?.reshaped([1, 1, -1])

    super.init()
  }

  /// Computes weight normalization: weight = g * (v / ||v||)
  /// This is mathematically equivalent to PyTorch's weight_norm
  private static func computeWeightNorm(
    weightV: MLXArray,
    weightG: MLXArray
  ) -> MLXArray {
    let rank = weightV.shape.count

    // Compute L2 norm over all axes except dim 0
    let axes = Array(1 ..< rank)
    let normV = MLX.sqrt(MLX.sum(weightV * weightV, axes: axes, keepDims: true))

    // Normalize: v / ||v|| with epsilon for numerical stability
    let normalizedWeight = weightV / (normV + 1e-7)

    // Scale by g
    return normalizedWeight * weightG
  }

  /// Forward pass for conv1d (6-parameter version)
  public func callAsFunction(_ x: MLXArray, conv: (MLXArray, MLXArray, Int, Int, Int, Int, StreamOrDevice) -> MLXArray) -> MLXArray {
    // Use pre-computed weights instead of computing on every call
    let weightToUse: MLXArray
    if x.shape.last == normalizedWeight.shape.last || groups > 1 {
      weightToUse = normalizedWeight
    } else {
      weightToUse = normalizedWeightTransposed
    }

    let result = conv(
      x,
      weightToUse,
      stride,
      padding,
      dilation,
      groups,
      .default
    )

    if let bias = reshapedBias {
      return result + bias
    }
    return result
  }

  /// Forward pass for convTransposed1d (7-parameter version with outputPadding)
  public func callAsFunction(_ x: MLXArray, conv: (MLXArray, MLXArray, Int, Int, Int, Int, Int, StreamOrDevice) -> MLXArray) -> MLXArray {
    // Use pre-computed weights instead of computing on every call
    let weightToUse: MLXArray
    if x.shape.last == normalizedWeight.shape.last || groups > 1 {
      weightToUse = normalizedWeight
    } else {
      weightToUse = normalizedWeightTransposed
    }

    let result = conv(
      x,
      weightToUse,
      stride,
      padding,
      dilation,
      outputPadding,
      groups,
      .default
    )

    if let bias = reshapedBias {
      return result + bias
    }
    return result
  }
}
