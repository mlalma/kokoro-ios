//
//  Kokoro-tts-lib
//
import Foundation

/// Supported languages for text-to-speech synthesis.
/// This enum defines the available language variants that can be used with the Kokoro TTS engine.
public enum Language: String, CaseIterable {
  /// No language specified or language-independent processing.
  case none = ""
  /// US English (American English) - lang_code 'a'
  case enUS = "en-us"
  /// GB English (British English) - lang_code 'b'
  case enGB = "en-gb"
  /// Japanese - lang_code 'j' (requires misaki[ja], not yet supported)
  case japanese = "ja"
  /// Mandarin Chinese - lang_code 'z' (requires misaki[zh], not yet supported)
  case chinese = "zh"
  /// Spanish - lang_code 'e' (uses misaki[en])
  case spanish = "es"
  /// French - lang_code 'f' (uses misaki[en])
  case french = "fr"
  /// Hindi - lang_code 'h' (uses misaki[en])
  case hindi = "hi"
  /// Italian - lang_code 'i' (uses misaki[en])
  case italian = "it"
  /// Brazilian Portuguese - lang_code 'p' (uses misaki[en])
  case portuguese = "pt-br"
}
