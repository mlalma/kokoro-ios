//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AdaIN1d: Module {
  @ModuleInfo private var norm: InstanceNorm1d
  @ModuleInfo private var fc: Linear

  public init(styleDim _: Int, numFeatures: Int, fcWeight: MLXArray, fcBias: MLXArray) {
    self._norm.wrappedValue = InstanceNorm1d(numFeatures: numFeatures, affine: false)
    self._fc.wrappedValue = Linear(weight: fcWeight, bias: fcBias)
  }

  public func callAsFunction(_ x: MLXArray, s: MLXArray) -> MLXArray {
    let h = fc(s)
    let hExpanded = h.expandedDimensions(axes: [2])
    let split = hExpanded.split(parts: 2, axis: 1)
    let gamma = split[0]
    let beta = split[1]

    let normalized = norm(x)
    return (1 + gamma) * normalized + beta
  }
}
