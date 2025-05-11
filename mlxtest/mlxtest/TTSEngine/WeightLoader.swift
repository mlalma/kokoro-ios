//
//  Kokoro-tts-lib
//

import Foundation
import Hub
import MLX
import MLXNN

// Utility class for loading and preprocessing the weights for the model
class WeightLoader {
  private init() {}

  static let resourceName = "kokoro-v1_0"
  static let resourceType = "safetensors"
  static let hubRepoId = "prince-canuma/Kokoro-82M"

  static func loadWeights() -> [String: MLXArray] {
    var weightsFileURL: URL?
    var operationError: Error?
    let semaphore = DispatchSemaphore(value: 0)

    Task {
      defer { semaphore.signal() }
      do {
        weightsFileURL = try await getWeightsFileURL()
      } catch {
        operationError = error
      }
    }

    semaphore.wait()  // Block until the async task completes

    if let error = operationError {
      fatalError("Failed to obtain weights file URL: \(error.localizedDescription)")
    }

    guard let finalURL = weightsFileURL else {
      fatalError(
        "Weight file URL could not be determined."
      )
    }

    logPrint("Proceeding to load and process weights from URL: \(finalURL.path)")
    do {
      return try self.loadAndProcessWeights(from: finalURL)
    } catch {
      fatalError(
        "Failed to load and process weights from \(finalURL.path): \(error.localizedDescription)")
    }
  }

  private static func getWeightsFileURL() async throws -> URL {
    let fileName = "\(resourceName).\(resourceType)"

    if let localPath = Bundle.main.path(forResource: resourceName, ofType: resourceType) {
      logPrint("Found local weights at: \(localPath)")
      return URL(fileURLWithPath: localPath)
    } else {
      logPrint(
        "Local weights file '\(fileName)' not found in bundle. Attempting to download from Hub...")

      let repo = Hub.Repo(id: hubRepoId)

      logPrint("Starting download of \(fileName) from repo \(repo.id)")

      let modelDirectory: URL = try await Hub.snapshot(
        from: repo,
        matching: [fileName],
        progressHandler: { progress in
          let percentage = (progress.fractionCompleted) * 100
          logPrint(String(format: "Download progress for \(fileName): %.2f%%", percentage))
        }
      )

      let downloadedFileUrl = modelDirectory.appendingPathComponent(fileName)

      logPrint("Weights downloaded to directory: \(modelDirectory.path). Expected file at: \(downloadedFileUrl.path)")

      if FileManager.default.fileExists(atPath: downloadedFileUrl.path) {
        logPrint("Successfully verified downloaded file exists at: \(downloadedFileUrl.path)")
        return downloadedFileUrl
      } else {
        var directoryContentsMessage =
          "Contents of download directory '\(modelDirectory.path)' could not be determined or directory is empty."
        if let contents = try? FileManager.default.contentsOfDirectory(atPath: modelDirectory.path),
          !contents.isEmpty
        {
          directoryContentsMessage =
            "Contents of download directory '\(modelDirectory.path)\': \(contents.joined(separator: ", "))"
        } else if let contents = try? FileManager.default.contentsOfDirectory(
          atPath: modelDirectory.path), contents.isEmpty
        {
          directoryContentsMessage = "Download directory '\(modelDirectory.path)' is empty."
        }

        throw NSError(
          domain: "WeightLoaderError", code: 1001,
          userInfo: [
            NSLocalizedDescriptionKey:
              "Downloaded weight file '\(fileName)' not found at expected location: \(downloadedFileUrl.path). \(directoryContentsMessage)"
          ])
      }
    }
  }

  private static func loadAndProcessWeights(from url: URL) throws -> [String: MLXArray] {
    let weights = try MLX.loadArrays(url: url)
    var sanitizedWeights: [String: MLXArray] = [:]

    for (key, value) in weights {
      if key.hasPrefix("bert") {
        if key.contains("position_ids") {
          continue
        }
        sanitizedWeights[key] = value
      } else if key.hasPrefix("predictor") {
        if key.contains("F0_proj.weight") {
          sanitizedWeights[key] = value.transposed(0, 2, 1)
        } else if key.contains("N_proj.weight") {
          sanitizedWeights[key] = value.transposed(0, 2, 1)
        } else if key.contains("weight_v") {
          if checkArrayShape(arr: value) {
            sanitizedWeights[key] = value
          } else {
            sanitizedWeights[key] = value.transposed(0, 2, 1)
          }
        } else {
          sanitizedWeights[key] = value
        }
      } else if key.hasPrefix("text_encoder") {
        if key.contains("weight_v") {
          if checkArrayShape(arr: value) {
            sanitizedWeights[key] = value
          } else {
            sanitizedWeights[key] = value.transposed(0, 2, 1)
          }
        } else {
          sanitizedWeights[key] = value
        }
      } else if key.hasPrefix("decoder") {
        if key.contains("noise_convs"), key.hasSuffix(".weight") {
          sanitizedWeights[key] = value.transposed(0, 2, 1)
        } else if key.contains("weight_v") {
          if checkArrayShape(arr: value) {
            sanitizedWeights[key] = value
          } else {
            sanitizedWeights[key] = value.transposed(0, 2, 1)
          }
        } else {
          sanitizedWeights[key] = value
        }
      }
    }
    return sanitizedWeights
  }

  private static func checkArrayShape(arr: MLXArray) -> Bool {
    guard arr.shape.count != 3 else { return false }

    let outChannels = arr.shape[0]
    let kH = arr.shape[1]
    let kW = arr.shape[2]

    return (outChannels >= kH) && (outChannels >= kW) && (kH == kW)
  }
}
