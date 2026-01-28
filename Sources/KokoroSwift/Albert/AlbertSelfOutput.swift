//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AlbertSelfOutput: Module {
  @ModuleInfo var dense: Linear
  @ModuleInfo var layerNorm: LayerNorm

  init(config: AlbertModelArgs) {
    self._dense.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
    self._layerNorm.wrappedValue = LayerNorm(
      dimensions: config.hiddenSize,
      eps: config.layerNormEps
    )
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    inputTensor: MLXArray
  ) -> MLXArray {
    var output = dense(hiddenStates)
    output = layerNorm(output + inputTensor)
    return output
  }
}
