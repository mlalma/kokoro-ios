//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

class Decoder: Module {
  @ModuleInfo var encode: AdainResBlk1d
  @ModuleInfo var decode: [AdainResBlk1d]
  @ModuleInfo(key: "F0_conv") var F0Conv: ConvWeighted
  @ModuleInfo(key: "N_conv") var NConv: ConvWeighted
  @ModuleInfo(key: "asr_res") var asrRes: [ConvWeighted]
  @ModuleInfo var generator: Generator

  init(
    weights: [String: MLXArray],
    dimIn: Int,
    styleDim: Int,
    dimOut _: Int,
    resblockKernelSizes: [Int],
    upsampleRates: [Int],
    upsampleInitialChannel: Int,
    resblockDilationSizes: [[Int]],
    upsampleKernelSizes: [Int],
    genIstftNFft: Int,
    genIstftHopSize: Int
  ) {
    self._encode.wrappedValue = AdainResBlk1d(weights: weights, weightKeyPrefix: "decoder.encode", dimIn: dimIn + 2, dimOut: 1024, styleDim: styleDim)

    var decodeBlocks: [AdainResBlk1d] = []
    decodeBlocks.append(AdainResBlk1d(weights: weights, weightKeyPrefix: "decoder.decode.0", dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: styleDim))
    decodeBlocks.append(AdainResBlk1d(weights: weights, weightKeyPrefix: "decoder.decode.1", dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: styleDim))
    decodeBlocks.append(AdainResBlk1d(weights: weights, weightKeyPrefix: "decoder.decode.2", dimIn: 1024 + 2 + 64, dimOut: 1024, styleDim: styleDim))
    decodeBlocks.append(AdainResBlk1d(weights: weights, weightKeyPrefix: "decoder.decode.3", dimIn: 1024 + 2 + 64, dimOut: 512, styleDim: styleDim, upsample: "true"))
    self._decode.wrappedValue = decodeBlocks

    self._F0Conv.wrappedValue = ConvWeighted(
      weightG: weights["decoder.F0_conv.weight_g"]!,
      weightV: weights["decoder.F0_conv.weight_v"]!,
      bias: weights["decoder.F0_conv.bias"]!,
      stride: 2,
      padding: 1,
      groups: 1
    )
    self._NConv.wrappedValue = ConvWeighted(
      weightG: weights["decoder.N_conv.weight_g"]!,
      weightV: weights["decoder.N_conv.weight_v"]!,
      bias: weights["decoder.N_conv.bias"]!,
      stride: 2,
      padding: 1,
      groups: 1
    )

    self._asrRes.wrappedValue = [ConvWeighted(
      weightG: weights["decoder.asr_res.0.weight_g"]!,
      weightV: weights["decoder.asr_res.0.weight_v"]!,
      bias: weights["decoder.asr_res.0.bias"]!,
      padding: 0
    )]

    self._generator.wrappedValue = Generator(
      weights: weights,
      styleDim: styleDim,
      resblockKernelSizes: resblockKernelSizes,
      upsampleRates: upsampleRates,
      upsampleInitialChannel: upsampleInitialChannel,
      resblockDilationSizes: resblockDilationSizes,
      upsampleKernelSizes: upsampleKernelSizes,
      genIstftNFft: genIstftNFft,
      genIstftHopSize: genIstftHopSize
    )
  }

  func callAsFunction(asr: MLXArray, F0Curve: MLXArray, N: MLXArray, s: MLXArray) -> MLXArray {
    let F0CurveSwapped = MLX.swappedAxes(F0Curve.reshaped([F0Curve.shape[0], 1, F0Curve.shape[1]]), 2, 1)
    let F0 = MLX.swappedAxes(F0Conv(F0CurveSwapped, conv: MLX.conv1d), 2, 1)

    let NSwapped = MLX.swappedAxes(N.reshaped([N.shape[0], 1, N.shape[1]]), 2, 1)
    let NProcessed = MLX.swappedAxes(NConv(NSwapped, conv: MLX.conv1d), 2, 1)

    var x = MLX.concatenated([asr, F0, NProcessed], axis: 1)
    x = encode(x: x, s: s)

    let asrResidual = MLX.swappedAxes(asrRes[0](MLX.swappedAxes(asr, 2, 1), conv: MLX.conv1d), 2, 1)
    var res = true

    for block in decode {
      if res {
        x = MLX.concatenated([x, asrResidual, F0, NProcessed], axis: 1)
      }
      x = block(x: x, s: s)

      if block.upsampleType != "none" {
        res = false
      }
    }

    return generator(x, s, F0Curve)
  }
}
