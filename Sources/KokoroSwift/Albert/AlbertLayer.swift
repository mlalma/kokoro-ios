//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class AlbertLayer: Module {
  @ModuleInfo var attention: AlbertSelfAttention
  @ModuleInfo var fullLayerLayerNorm: LayerNormInference
  @ModuleInfo var ffn: Linear
  @ModuleInfo var ffnOutput: Linear

  init(weights: [String: MLXArray], config: AlbertModelArgs, layerNum: Int, innerGroupNum: Int) {
    self._attention.wrappedValue = AlbertSelfAttention(weights: weights, config: config, layerNum: layerNum, innerGroupNum: innerGroupNum)
    self._ffn.wrappedValue = Linear(weight: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).ffn.weight"]!,
                 bias: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).ffn.bias"]!)
    self._ffnOutput.wrappedValue = Linear(weight: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).ffn_output.weight"]!,
                       bias: weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).ffn_output.bias"]!)

    let fullLayerLayerNormWeights = weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).full_layer_layer_norm.weight"]!
    let fullLayerLayerNormBiases = weights["bert.encoder.albert_layer_groups.\(layerNum).albert_layers.\(innerGroupNum).full_layer_layer_norm.bias"]!

    guard fullLayerLayerNormWeights.count == config.hiddenSize, fullLayerLayerNormBiases.count == config.hiddenSize else {
      fatalError("Wrong shape for AlbertLayer FullLayerLayerNorm bias or weights!")
    }

    // Use LayerNormInference which accepts weights directly in init
    self._fullLayerLayerNorm.wrappedValue = LayerNormInference(weight: fullLayerLayerNormWeights, bias: fullLayerLayerNormBiases, eps: config.layerNormEps)
  }

  func ffChunk(_ attentionOutput: MLXArray) -> MLXArray {
    var ffnOutputArray = ffn(attentionOutput)
    ffnOutputArray = MLXNN.gelu(ffnOutputArray)
    ffnOutputArray = ffnOutput(ffnOutputArray)
    return ffnOutputArray
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    attentionMask: MLXArray? = nil
  ) -> MLXArray {
    let attentionOutput = attention(hiddenStates, attentionMask: attentionMask)
    let ffnOutput = ffChunk(attentionOutput)
    let output = fullLayerLayerNorm(ffnOutput + attentionOutput)
    return output
  }
}
