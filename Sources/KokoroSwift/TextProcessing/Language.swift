//
//  Kokoro-tts-lib
//
import Foundation
import MisakiSwift

/// Supported languages for text-to-speech synthesis.
/// This enum defines the available language variants that can be used with the Kokoro TTS engine.
public enum Language: String, CaseIterable {
  /// No language specified or language-independent processing.
  case none = ""
  /// US English (American English).
  case enUS = "en-us"
  /// GB English (British English).
  case enGB = "en-gb"
  /// Chinese (Mandarin, Simplified).
  case zhCN = "zh-cn"
}
