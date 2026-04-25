//
//  Kokoro-tts-lib
//
#if canImport(MisakiSwift)

import Foundation
import MisakiSwift
import MLXUtilsLibrary

#if canImport(MisakiZH)
import MisakiZH
#endif

/// A G2P processor that uses the MisakiSwift library for English phonemization
/// and MisakiZH for Chinese (Mandarin) phonemization.
/// Requires the MisakiSwift framework to be available at compile time.
/// Chinese support requires the MisakiZH framework.
final class MisakiG2PProcessor : G2PProcessor {
  /// The underlying MisakiSwift English G2P engine instance.
  /// This property is initialized when `setLanguage(_:)` is called with an English language.
  private var englishG2P: EnglishG2P?

  #if canImport(MisakiZH)
  /// The underlying MisakiZH Chinese G2P engine instance.
  /// This property is initialized when `setLanguage(_:)` is called with `.zhCN`.
  private var chineseG2P: ZHG2P?
  #endif

  /// The currently active language.
  private var currentLanguage: Language = .none

  /// Configures the processor for the specified language.
  /// - Parameter language: The target language for phonemization.
  /// - Throws: `G2PProcessorError.unsupportedLanguage` if the language is not supported.
  func setLanguage(_ language: Language) throws {
    currentLanguage = language
    switch language {
    case .enUS:
      englishG2P = EnglishG2P(british: false)
    case .enGB:
      englishG2P = EnglishG2P(british: true)
    case .zhCN:
      #if canImport(MisakiZH)
      chineseG2P = try ZHG2P()
      #else
      throw G2PProcessorError.unsupportedLanguage
      #endif
    default:
      throw G2PProcessorError.unsupportedLanguage
    }
  }
  
  /// Converts input text to phonetic representation.
  /// - Parameter input: The text string to be converted to phonemes.
  /// - Returns: A phonetic string representation of the input text and optionally arrays of tokens.
  /// - Throws: `G2PProcessorError.processorNotInitialized` if `setLanguage(_:)` has not been called.
  func process(input: String) throws -> (String, [MToken]?) {
    switch currentLanguage {
    case .enUS, .enGB:
      guard let englishG2P else { throw G2PProcessorError.processorNotInitialized }
      return englishG2P.phonemize(text: input)
    case .zhCN:
      #if canImport(MisakiZH)
      guard let chineseG2P else { throw G2PProcessorError.processorNotInitialized }
      let (phonemes, tokens) = chineseG2P.phonemize(input)
      return (phonemes, tokens.isEmpty ? nil : tokens)
      #else
      throw G2PProcessorError.unsupportedLanguage
      #endif
    default:
      throw G2PProcessorError.processorNotInitialized
    }
  }
}

#endif
