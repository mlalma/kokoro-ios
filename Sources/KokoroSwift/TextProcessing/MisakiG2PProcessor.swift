//
//  Kokoro-tts-lib
//
#if canImport(MisakiSwift)

import Foundation
import MisakiSwift
import MLXUtilsLibrary

/// A G2P processor that uses the MisakiSwift library for phonemization.
/// Supports English and other languages that use the English G2P engine (Spanish, French, Hindi, Italian, Portuguese).
/// Requires the MisakiSwift framework to be available at compile time.
final class MisakiG2PProcessor : G2PProcessor {
  /// The underlying MisakiSwift English G2P engine instance.
  /// This property is initialized when `setLanguage(_:)` is called and remains
  /// `nil` until the processor is properly configured.
  var misaki: EnglishG2P?

  /// Configures the processor for the specified language.
  /// - Parameter language: The target language for phonemization.
  /// - Throws: `G2PProcessorError.unsupportedLanguage` for Japanese and Chinese (require dedicated G2P).
  func setLanguage(_ language: Language) throws {
    switch language {
    case .enUS, .spanish, .french, .hindi, .italian, .portuguese:
      // These languages all use misaki[en] with American English base
      misaki = EnglishG2P(british: false)
    case .enGB:
      // British English
      misaki = EnglishG2P(british: true)
    case .japanese, .chinese:
      // These require dedicated misaki[ja] / misaki[zh] - not yet ported to Swift
      throw G2PProcessorError.unsupportedLanguage
    case .none:
      // Default to American English
      misaki = EnglishG2P(british: false)
    }
  }

  /// Converts input text to phonetic representation.
  /// - Parameter input: The text string to be converted to phonemes.
  /// - Returns: A phonetic string representation of the input text and arrays of tokens.
  /// - Throws: `G2PProcessorError.processorNotInitialized` if `setLanguage(_:)` has not been called.
  func process(input: String) throws -> (String, [MToken]?) {
    guard let misaki else { throw G2PProcessorError.processorNotInitialized }
    return misaki.phonemize(text: input)
  }
}

#endif
