//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN
import MLXUtilsLibrary

/// Main class that encapsulates the complete Kokoro text-to-speech pipeline.
///
/// KokoroTTS converts text input into audio output by:
/// 1. Processing text through grapheme-to-phoneme (G2P) conversion
/// 2. Encoding the phonemes using BERT-based embeddings
/// 3. Predicting duration and prosody for natural speech
/// 4. Generating audio through a decoder network
///
/// Example usage:
/// ```swift
/// let tts = KokoroTTS(modelPath: modelURL, g2p: .misaki)
/// let audioData = try tts.generateAudio(voice: voiceEmbedding,
///                                       language: .english,
///                                       text: "Hello world",
///                                       speed: 1.0)
/// ```
public final class KokoroTTS {
  /// Errors from the TTS side
  public enum KokoroTTSError: Error {
    /// Thrown when input text exceeds maximum token count
    case tooManyTokens
  }

  /// BERT model for encoding phoneme sequences
  private let bert: CustomAlbert!

  /// Linear layer to project BERT embeddings
  private let bertEncoder: Linear!

  /// Encoder for duration prediction features
  private let durationEncoder: DurationEncoder!

  /// Bidirectional LSTM for duration prediction
  private let predictorLSTM: LSTM!

  /// Projection layer for final duration values
  private let durationProj: Linear!

  /// Predictor for prosodic features (F0, pitch)
  private let prosodyPredictor: ProsodyPredictor!

  /// Text encoder that processes phoneme sequences
  private let textEncoder: TextEncoder!

  /// Decoder that generates audio from encoded features
  private let decoder: Decoder!

  /// Grapheme-to-phoneme processor for text conversion
  private let g2pProcessor: G2PProcessor?

  /// Currently active language (cached to avoid reinitializing G2P)
  private var chosenLanguage: Language = .none

  /// Initializes the Kokoro TTS engine with model weights and G2P processor.
  /// - Parameters:
  ///   - modelPath: URL to the directory containing model weights
  ///   - g2p: Grapheme-to-phoneme processor type (default: Misaki)
  public init(modelPath: URL, g2p: G2P = .misaki) {
    // Load and sanitize model weights
    let sanitizedWeights = WeightLoader.loadWeights(modelPath: modelPath)
    let config = KokoroConfig.loadConfig()

    // Initialize BERT model for phoneme encoding
    bert = CustomAlbert(
      weights: sanitizedWeights,
      config: AlbertModelArgs(
        numHiddenLayers: config.plbert.numHiddenLayers,
        numAttentionHeads: config.plbert.numAttentionHeads,
        hiddenSize: config.plbert.hiddenSize,
        intermediateSize: config.plbert.intermediateSize,
        vocabSize: config.nToken
      )
    )

    // Initialize BERT output encoder
    bertEncoder = Linear(
      weight: sanitizedWeights["bert_encoder.weight"]!,
      bias: sanitizedWeights["bert_encoder.bias"]!
    )

    // Initialize duration prediction components
    durationEncoder = DurationEncoder(
      weights: sanitizedWeights,
      dModel: config.hiddenDim,
      styDim: config.styleDim,
      nlayers: config.nLayer
    )

    // Initialize bidirectional LSTM for duration prediction
    predictorLSTM = LSTM(
      inputSize: config.hiddenDim + config.styleDim,
      hiddenSize: config.hiddenDim / 2,
      wxForward: sanitizedWeights["predictor.lstm.weight_ih_l0"]!,
      whForward: sanitizedWeights["predictor.lstm.weight_hh_l0"]!,
      biasIhForward: sanitizedWeights["predictor.lstm.bias_ih_l0"]!,
      biasHhForward: sanitizedWeights["predictor.lstm.bias_hh_l0"]!,
      wxBackward: sanitizedWeights["predictor.lstm.weight_ih_l0_reverse"]!,
      whBackward: sanitizedWeights["predictor.lstm.weight_hh_l0_reverse"]!,
      biasIhBackward: sanitizedWeights["predictor.lstm.bias_ih_l0_reverse"]!,
      biasHhBackward: sanitizedWeights["predictor.lstm.bias_hh_l0_reverse"]!
    )

    // Initialize duration projection layer
    durationProj = Linear(
      weight: sanitizedWeights["predictor.duration_proj.linear_layer.weight"]!,
      bias: sanitizedWeights["predictor.duration_proj.linear_layer.bias"]!
    )

    // Initialize prosody predictor (F0, pitch, etc.)
    prosodyPredictor = ProsodyPredictor(
      weights: sanitizedWeights,
      styleDim: config.styleDim,
      dHid: config.hiddenDim
    )

    // Initialize text encoder
    textEncoder = TextEncoder(
      weights: sanitizedWeights,
      channels: config.hiddenDim,
      kernelSize: config.textEncoderKernelSize,
      depth: config.nLayer,
      nSymbols: config.nToken
    )

    // Initialize audio decoder
    decoder = Decoder(
      weights: sanitizedWeights,
      dimIn: config.hiddenDim,
      styleDim: config.styleDim,
      dimOut: config.nMels,
      resblockKernelSizes: config.istftNet.resblockKernelSizes,
      upsampleRates: config.istftNet.upsampleRates,
      upsampleInitialChannel: config.istftNet.upsampleInitialChannel,
      resblockDilationSizes: config.istftNet.resblockDilationSizes,
      upsampleKernelSizes: config.istftNet.upsampleKernelSizes,
      genIstftNFft: config.istftNet.genIstftNFFT,
      genIstftHopSize: config.istftNet.genIstftHopSize
    )

    // Initialize G2P processor for text-to-phoneme conversion
    g2pProcessor = try? G2PFactory.createG2PProcessor(engine: g2p)
  }

  /// Quantizes the model to reduce memory usage and potentially speed up inference.
  ///
  /// This method converts all Linear layers to QuantizedLinear using the specified
  /// bit width and group size. Quantization reduces model size significantly:
  /// - 4-bit: ~4x memory reduction
  /// - 8-bit: ~2x memory reduction
  ///
  /// - Parameters:
  ///   - groupSize: Number of weights to group together for quantization (default: 64)
  ///   - bits: Bit width for quantized weights (4 or 8, default: 4)
  /// - Note: Call this after model initialization but before inference for best results.
  public func quantize(groupSize: Int = 64, bits: Int = 4) {
    // Quantize major model components
    // Note: Only quantize modules with simple structures (no heterogeneous arrays)
    // durationEncoder, prosodyPredictor, textEncoder have [Module] arrays with mixed types
    // which cause issues with Module.update()

    // Filter function: only quantize Linear layers with compatible dimensions
    let linearFilter: (String, Module) -> Bool = { [groupSize] _, module in
      guard let linear = module as? Linear else { return false }
      // Check that the last dimension of weight is divisible by groupSize
      // MLX quantization requires this for the matmul to work correctly
      let weightShape = linear.weight.shape
      guard weightShape.count >= 1 else { return false }
      let lastDim = weightShape[weightShape.count - 1]
      return lastDim % groupSize == 0
    }

    // BERT (CustomAlbert) - safe to quantize
    MLXNN.quantize(
      model: bert,
      groupSize: groupSize,
      bits: bits,
      filter: linearFilter
    )

    // Decoder - safe to quantize (complex but homogeneous structures)
    MLXNN.quantize(
      model: decoder,
      groupSize: groupSize,
      bits: bits,
      filter: linearFilter
    )

    // Skip: durationEncoder, prosodyPredictor, textEncoder
    // These contain [Module] arrays with mixed LSTM/AdaLayerNorm types
    // which cause Module.update() errors

    // Also quantize the standalone Linear layers
    MLXNN.quantize(
      model: bertEncoder,
      groupSize: groupSize,
      bits: bits,
      filter: linearFilter
    )

    MLXNN.quantize(
      model: durationProj,
      groupSize: groupSize,
      bits: bits,
      filter: linearFilter
    )

    // Force evaluation to apply quantization
    MLX.eval(bert, decoder, bertEncoder, durationProj)
  }

  /// Generates audio from text using the specified voice and parameters.
  ///
  /// This method performs the complete TTS pipeline:
  /// 1. Converts text to phonemes (G2P)
  /// 2. Tokenizes and encodes phonemes
  /// 3. Predicts duration and prosody
  /// 4. Generates audio waveform
  ///
  /// - Parameters:
  ///   - voice: Voice embedding array (contains speaker characteristics)
  ///   - language: Target language for pronunciation
  ///   - text: Input text to synthesize
  ///   - speed: Speech speed multiplier (1.0 = normal, >1.0 = faster, <1.0 = slower)
  /// - Returns: Array of audio samples as Float values
  /// - Throws: `KokoroTTSError.tooManyTokens` if text is too long,
  ///           or `G2PProcessorError` if G2P processing fails
  public func generateAudio(voice: MLXArray, language: Language, text: String, speed: Float = 1.0) throws -> ([Float], [MToken]?) {
    // Update language if it has changed
    try updateLanguageIfNeeded(language)

    // Start performance timing
    BenchmarkTimer.reset()
    BenchmarkTimer.startTimer(Constants.bm_TTS)

    // Step 1: Convert text to phonemes
    let (phonemizedText, tokenArray) = try phonemizeText(text)

    // Step 2: Tokenize and prepare input
    let (paddedInputIds, attentionMask, inputLengths, textMask, inputIds) = try prepareInputTensors(phonemizedText)

    // Step 3: Extract style embeddings from voice
    let (globalStyle, acousticStyle) = extractStyleEmbeddings(from: voice, tokenCount: inputIds.count)

    // Step 4-5: BERT encoding and duration prediction
    // Note: Removed intermediate .eval() calls to preserve lazy evaluation graph
    let durationFeatures = encodeBERTAndDuration(
      inputIds: paddedInputIds,
      attentionMask: attentionMask,
      inputLengths: inputLengths,
      textMask: textMask,
      style: globalStyle
    )

    let (predictedDurations, alignmentTarget) = predictDurations(
      features: durationFeatures,
      batchSize: paddedInputIds.shape[1],
      speed: speed
    )

    // Step 6-7: Prosody prediction
    let alignedEncoding = durationFeatures.transposed(0, 2, 1).matmul(alignmentTarget)
    let (f0Prediction, nPrediction) = prosodyPredictor.F0NTrain(x: alignedEncoding, s: globalStyle)

    // Step 8-9: Text encoding and audio generation
    let textEncoding = textEncoder(paddedInputIds, inputLengths: inputLengths, m: textMask)
    let asrFeatures = MLX.matmul(textEncoding, alignmentTarget)

    let audio = decoder(
      asr: asrFeatures,
      F0Curve: f0Prediction,
      N: nPrediction,
      s: acousticStyle
    )[0]

    // Single eval at the end triggers complete graph evaluation
    audio.eval()

    // Try to predict timestamp of each token if G2P processor returns tokens
    if let tokenArray {
      TimestampPredictor.preditTimestamps(tokens: tokenArray, predictionDuration: predictedDurations)
    }

    // Stop performance timing
    BenchmarkTimer.stopTimer(Constants.bm_TTS)

    // Extract result before clearing cache
    let result = audio[0].asArray(Float.self)

    // Clear GPU cache to reduce memory accumulation
    Memory.clearCache()

    return (result, tokenArray)
  }

  /// Updates the G2P language if it differs from the current language.
  private func updateLanguageIfNeeded(_ language: Language) throws {
    guard chosenLanguage != language else { return }

    guard let g2pProcessor else {
      throw G2PProcessorError.processorNotInitialized
    }

    try g2pProcessor.setLanguage(language)
    chosenLanguage = language
  }

  /// Converts input text to phonemes using the G2P processor.
  private func phonemizeText(_ text: String) throws -> (String, [MToken]?) {
    let phonemizedOutput = try g2pProcessor?.process(input: text)
    guard let phonemizedOutput else {
      throw G2PProcessorError.processorNotInitialized
    }
    return phonemizedOutput
  }

  /// Prepares input tensors for the model from phonemized text.
  /// - Returns: Tuple containing:
  ///   - paddedInputIds: Tokenized and padded input sequence
  ///   - attentionMask: Mask for attention mechanism
  ///   - inputLengths: Length of input sequence
  ///   - textMask: Mask for text padding
  ///   - inputIds: Original token IDs before padding
  private func prepareInputTensors(_ phonemizedText: String) throws -> (MLXArray, MLXArray, MLXArray, MLXArray, [Int]) {
    // Tokenize phonemized text
    let inputIds = Tokenizer.tokenize(phonemizedText: phonemizedText)

    // Check token count limit
    guard inputIds.count <= Constants.maxTokenCount else {
      throw KokoroTTSError.tooManyTokens
    }

    // Add padding tokens at start and end
    let paddedInputIdsArray = [0] + inputIds + [0]
    let paddedInputIds = MLXArray(paddedInputIdsArray).expandedDimensions(axes: [0])

    // Create input length tensor
    let inputLengths = MLXArray(paddedInputIds.dim(-1))
    let inputLengthMax: Int = inputLengths.max().item()

    // Create text mask for padding positions
    var textMask = MLXArray(0 ..< inputLengthMax)
    textMask = textMask + 1 .> inputLengths
    textMask = textMask.expandedDimensions(axes: [0])

    // Create attention mask using pure MLX operations (1 for valid positions, 0 for padding)
    // textMask is true for padding positions, so we negate it
    let attentionMask = MLX.where(textMask, MLXArray(0), MLXArray(1)).reshaped(textMask.shape)

    return (paddedInputIds, attentionMask, inputLengths, textMask, inputIds)
  }

  /// Extracts style embeddings from the voice array.
  /// - Parameters:
  ///   - voice: Voice embedding array
  ///   - tokenCount: Number of tokens in the input
  /// - Returns: Tuple of (globalStyle, acousticStyle)
  ///   - globalStyle: Style embedding for prosody/duration (indices 128+)
  ///   - acousticStyle: Style embedding for acoustic features (indices 0-127)
  private func extractStyleEmbeddings(from voice: MLXArray, tokenCount: Int) -> (MLXArray, MLXArray) {
    // Extract reference style from voice embedding
    let referenceStyle = voice[tokenCount - 1, 0 ... 1, 0...]

    // Split into global style (for prosody/duration) and acoustic style
    let globalStyle = referenceStyle[0 ... 1, 128...]
    let acousticStyle = referenceStyle[0 ... 1, 0 ... 127]

    return (globalStyle, acousticStyle)
  }

  /// Encodes text with BERT and generates duration prediction features.
  private func encodeBERTAndDuration(
    inputIds: MLXArray,
    attentionMask: MLXArray,
    inputLengths: MLXArray,
    textMask: MLXArray,
    style: MLXArray
  ) -> MLXArray {
    // Pass through BERT model
    let (bertOutput, _) = bert(inputIds, attentionMask: attentionMask)

    // Project BERT output and transpose for duration encoder
    let bertEncoded = bertEncoder(bertOutput).transposed(0, 2, 1)

    // Generate duration features with style conditioning
    let durationFeatures = durationEncoder(
      bertEncoded,
      style: style,
      textLengths: inputLengths,
      m: textMask
    )

    return durationFeatures
  }

  /// Predicts phoneme durations and creates alignment target matrix.
  /// - Parameters:
  ///   - features: Duration prediction features from encoder
  ///   - batchSize: Size of the input batch
  ///   - speed: Speech speed multiplier
  /// - Returns: Predicted durations and alignment target matrix for duration expansion
  private func predictDurations(features: MLXArray, batchSize: Int, speed: Float) -> (MLXArray, MLXArray) {
    // Pass through LSTM
    let (lstmOutput, _) = predictorLSTM(features)

    // Project to duration values
    let durationLogits = durationProj(lstmOutput)

    // Convert to actual durations (clamped to minimum of 1 frame)
    let durationSigmoid = MLX.sigmoid(durationLogits).sum(axis: -1) / speed
    let predictedDurations = MLX.clip(durationSigmoid.round(), min: 1).asType(.int32)[0]

    // Create alignment matrix
    return (predictedDurations, createAlignmentTarget(durations: predictedDurations, batchSize: batchSize))
  }

  /// Creates an alignment target matrix from predicted durations.
  /// Maps each phoneme to multiple frames based on duration.
  /// - Parameters:
  ///   - durations: Predicted duration for each phoneme [numPhonemes]
  ///   - batchSize: Size of the input batch (number of phonemes)
  /// - Returns: Alignment matrix [1 × batchSize × totalFrames]
  private func createAlignmentTarget(durations: MLXArray, batchSize: Int) -> MLXArray {
    // Get all durations in a single batch transfer (1 GPU sync instead of N)
    durations.eval()
    let durationArray: [Int32] = durations.asArray(Int32.self)
    let totalFrames = durationArray.reduce(0) { $0 + Int($1) }

    // Build indices array - each phoneme index repeated by its duration
    // This happens on CPU but uses batch-transferred data (no GPU sync per phoneme)
    var indices: [Int32] = []
    indices.reserveCapacity(totalFrames)
    for (phonemeIdx, duration) in durationArray.enumerated() {
      for _ in 0 ..< Int(duration) {
        indices.append(Int32(phonemeIdx))
      }
    }

    // Create one-hot alignment using MLX scatter
    let indicesMLX = MLXArray(indices)
    let frameIndices = MLXArray(0 ..< totalFrames)

    // Create zeros and scatter 1.0 values at (frame, phoneme) positions
    let oneHot = MLXArray.zeros([totalFrames, batchSize])
    oneHot[frameIndices, indicesMLX] = MLXArray(1.0)

    // Transpose to [batchSize, totalFrames] and add batch dimension
    let alignmentTarget = oneHot.transposed(1, 0)
    return alignmentTarget.expandedDimensions(axis: 0)
  }

  /// Constants used throughout the TTS engine.
  public struct Constants {
    /// Maximum number of tokens allowed in input
    public static let maxTokenCount = 510

    /// Audio sampling rate in Hz
    public static let samplingRate = 24000

    // Benchmark timer identifiers
    public static let bm_TTS = "TTSAudio"
    static let bm_Phonemize = "Phonemize"
    static let bm_bert = "BERT"
    static let bm_duration = "Duration"
    static let bm_prosody = "Prosody"
    static let bm_decoder = "Decoder"
  }
}
