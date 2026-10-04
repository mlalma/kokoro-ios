import MLX
import Testing
@testable import KokoroSwift

/// Row p is 1 for the frames that phoneme p lasts: the reference the model expects.
private func referenceAlignment(_ durations: [Int32]) -> [[Float]] {
  let frames = durations.reduce(0) { $0 + Int($1) }
  var rows = Array(repeating: Array(repeating: Float(0), count: frames), count: durations.count)
  var frame = 0
  for (phoneme, duration) in durations.enumerated() {
    for _ in 0 ..< Int(duration) {
      rows[phoneme][frame] = 1
      frame += 1
    }
  }
  return rows
}

private func alignmentRows(_ durations: [Int32]) -> [[Float]] {
  let target = KokoroTTS.createAlignmentTarget(durations: MLXArray(durations), batchSize: durations.count)
  #expect(target.shape == [1, durations.count, durations.reduce(0) { $0 + Int($1) }])
  let values = target.asArray(Float.self)
  let frames = target.shape[2]
  return (0 ..< durations.count).map { Array(values[$0 * frames ..< ($0 + 1) * frames]) }
}

@Test func alignmentTargetIsOneHotPerFrame() {
  #expect(alignmentRows([1, 2, 3]) == [
    [1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1],
  ])
}

@Test func alignmentTargetMatchesReferenceForSentenceSizedInput() {
  var generator = SystemRandomNumberGenerator()
  let durations = (0 ..< 120).map { _ in Int32.random(in: 1 ... 12, using: &generator) }
  #expect(alignmentRows(durations) == referenceAlignment(durations))
}
