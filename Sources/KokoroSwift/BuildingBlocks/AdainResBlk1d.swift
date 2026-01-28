//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AdainResBlk1d: Module {
  let actv: LeakyReLU
  let dimIn: Int
  let upsampleType: String
  @ModuleInfo var upsample: UpSample1d
  let learned_sc: Bool
  @ModuleInfo var pool: Module

  @ModuleInfo var conv1: ConvWeighted
  @ModuleInfo var conv2: ConvWeighted
  @ModuleInfo var norm1: AdaIN1d
  @ModuleInfo var norm2: AdaIN1d
  @ModuleInfo var conv1x1: ConvWeighted?

  init(
    weights: [String: MLXArray],
    weightKeyPrefix: String,
    dimIn: Int,
    dimOut: Int,
    styleDim: Int = 64,
    actv: LeakyReLU = LeakyReLU(negativeSlope: 0.2),
    upsample: String = "none"
  ) {
    self.actv = actv
    self.dimIn = dimIn
    upsampleType = upsample
    self._upsample.wrappedValue = UpSample1d(layerType: upsample)
    learned_sc = dimIn != dimOut

    if upsample == "none" {
      self._pool.wrappedValue = Identity()
    } else {
      self._pool.wrappedValue = ConvWeighted(
        weightG: weights[weightKeyPrefix + ".pool.weight_g"]!,
        weightV: weights[weightKeyPrefix + ".pool.weight_v"]!,
        bias: weights[weightKeyPrefix + ".pool.bias"]!,
        stride: 2,
        padding: 1,
        groups: dimIn
      )
    }

    // Initialize all conv/norm layers inline (not in separate method)
    self._conv1.wrappedValue = ConvWeighted(
      weightG: weights[weightKeyPrefix + ".conv1.weight_g"]!,
      weightV: weights[weightKeyPrefix + ".conv1.weight_v"]!,
      bias: weights[weightKeyPrefix + ".conv1.bias"]!,
      stride: 1,
      padding: 1
    )

    self._conv2.wrappedValue = ConvWeighted(
      weightG: weights[weightKeyPrefix + ".conv2.weight_g"]!,
      weightV: weights[weightKeyPrefix + ".conv2.weight_v"]!,
      bias: weights[weightKeyPrefix + ".conv2.bias"]!,
      stride: 1,
      padding: 1
    )

    self._norm1.wrappedValue = AdaIN1d(
      styleDim: styleDim,
      numFeatures: dimIn,
      fcWeight: weights[weightKeyPrefix + ".norm1.fc.weight"]!,
      fcBias: weights[weightKeyPrefix + ".norm1.fc.bias"]!
    )

    self._norm2.wrappedValue = AdaIN1d(
      styleDim: styleDim,
      numFeatures: dimIn,
      fcWeight: weights[weightKeyPrefix + ".norm2.fc.weight"]!,
      fcBias: weights[weightKeyPrefix + ".norm2.fc.bias"]!
    )

    if learned_sc {
      self._conv1x1.wrappedValue = ConvWeighted(
        weightG: weights[weightKeyPrefix + ".conv1x1.weight_g"]!,
        weightV: weights[weightKeyPrefix + ".conv1x1.weight_v"]!,
        bias: nil,
        stride: 1,
        padding: 0
      )
    } else {
      self._conv1x1.wrappedValue = nil
    }
  }

  func shortcut(_ x: MLXArray) -> MLXArray {
    var x = MLX.swappedAxes(x, 2, 1)
    x = upsample(x)
    x = MLX.swappedAxes(x, 2, 1)

    if let conv1x1 = conv1x1 {
      x = MLX.swappedAxes(x, 2, 1)
      x = conv1x1(x, conv: MLX.conv1d)
      x = MLX.swappedAxes(x, 2, 1)
    }

    return x
  }

  func residual(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
    var x = norm1(x, s: s)
    x = actv(x)

    x = MLX.swappedAxes(x, 2, 1)
    if upsampleType != "none" {
      if let idPool = pool as? Identity {
        x = idPool(x)
      } else if let convPool = pool as? ConvWeighted {
        x = convPool(x, conv: MLX.convTransposed1d)
      }
      x = MLX.padded(x, widths: [IntOrPair([0, 0]), IntOrPair([1, 0]), IntOrPair([0, 0])])
    }
    x = MLX.swappedAxes(x, 2, 1)

    x = MLX.swappedAxes(x, 2, 1)
    x = conv1(x, conv: MLX.conv1d)
    x = MLX.swappedAxes(x, 2, 1)

    x = norm2(x, s: s)
    x = actv(x)

    x = MLX.swappedAxes(x, 2, 1)
    x = conv2(x, conv: MLX.conv1d)
    x = MLX.swappedAxes(x, 2, 1)

    return x
  }

  func callAsFunction(x: MLXArray, s: MLXArray) -> MLXArray {
    let out = residual(x, s)
    let result = (out + shortcut(x)) / sqrt(2.0)
    return result
  }
}
