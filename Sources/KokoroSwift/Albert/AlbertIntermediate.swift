//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AlbertIntermediate: Module {
  @ModuleInfo var dense: Linear

  init(config: AlbertModelArgs) {
    self._dense.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
  }

  func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
    var output = dense(hiddenStates)
    output = MLXNN.gelu(output)
    return output
  }
}
