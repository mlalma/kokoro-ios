//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class UpSample1d: Module {
  private let layerType: String
  @ModuleInfo private var interpolate: Upsample

  init(layerType: String) {
    self.layerType = layerType
    self._interpolate.wrappedValue = Upsample(
      scaleFactor: 2.0,
      mode: .nearest
    )
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    if layerType == "none" {
      return x
    } else {
      return interpolate(x)
    }
  }
}
