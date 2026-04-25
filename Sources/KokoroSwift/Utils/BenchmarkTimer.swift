//
//  MLXUtilsLibrary
//
import Foundation
import MLX

/// A singleton benchmarking utility for measuring and logging task execution times.
/// `BenchmarkTimer` provides a hierarchical timing system that can measure nested tasks
/// and print formatted results. It's designed to be used only in DEBUG builds and
/// automatically becomes a no-op in RELEASE builds for zero runtime overhead.
///
/// **Thread Safety:** This class is thread-safe and can be used across multiple threads.
///
/// Usage Example:
/// ```swift
/// // Start timing the main task
/// BenchmarkTimer.startTimer("mainTask")
///
/// // Start timing a subtask within the main task
/// BenchmarkTimer.startTimer("subtask", "mainTask")
/// // Perform subtask work...
/// BenchmarkTimer.stopTimer("subtask")
///
/// // Finish main task
/// BenchmarkTimer.stopTimer("mainTask")
///
/// // Print all timing results
/// BenchmarkTimer.print()
///
/// // Clean up timers
/// BenchmarkTimer.reset()
/// ```
public final class BenchmarkTimer {
  /// Internal class representing a single timing measurement.
  ///
  /// Each `Timing` instance tracks the start/stop times of a task and maintains
  /// references to child tasks for hierarchical timing.
  private class Timing {
    /// Unique identifier for this timing measurement
    let id: String
    
    /// Start time of the current timing interval
    private var start: DispatchTime
    
    /// Finish time of the timing measurement (nil if not yet stopped)
    private var finish: DispatchTime?
    
    /// Child timing tasks nested within this task
    private var childTasks: [Timing] = []
    
    /// Parent timing task (nil for root-level tasks)
    internal let parent: Timing?
    
    /// Accumulated time delta in nanoseconds (supports multiple start/stop cycles)
    private var delta: UInt64 = 0
    
    /// Lock for thread-safe access to mutable properties
    private let lock = NSLock()
    
    /// Initializes a new timing measurement.
    /// - Parameters:
    ///   - id: Unique identifier for this timing
    ///   - parent: Optional parent timing for hierarchical measurements
    init(id: String, parent: Timing?) {
      start = DispatchTime.now()
      self.id = id
      self.parent = parent
      
      // Add this timing to parent's child list (thread-safe)
      if let parent {
        parent.lock.lock()
        parent.childTasks.append(self)
        parent.lock.unlock()
      }
    }
    
    /// Starts or restarts the timer for this timing measurement.
    func startTimer() {
      lock.lock()
      start = DispatchTime.now()
      lock.unlock()
    }
    
    /// Stops the timer and accumulates the elapsed time.
    func stop() {
      lock.lock()
      finish = DispatchTime.now()
      delta += finish!.uptimeNanoseconds - start.uptimeNanoseconds
      lock.unlock()
    }
    
    /// Logs this timing and all child timings in a hierarchical format.
    /// - Parameter spaces: Number of leading spaces for indentation (used for hierarchy)
    func log(spaces: Int = 0) {
      lock.lock()
      guard let _ = finish else {
        lock.unlock()
        return
      }
      
      let spaceString = String(repeating: " ", count: spaces)
      let message = spaceString + id + ": " + deltaInSec + " sec"
      let children = childTasks
      lock.unlock()
      
      Swift.print(message)
      for childTask in children {
        childTask.log(spaces: spaces + 2)
      }
    }
    
    /// Returns the accumulated time in seconds
    var deltaTime: Double {
      lock.lock()
      let time = Double(delta) / 1_000_000_000
      lock.unlock()
      return time
    }
    
    /// Returns the accumulated time formatted as a string with 4 decimal places
    var deltaInSec: String {
      lock.lock()
      let formatted = String(format: "%.4f", Double(delta) / 1_000_000_000)
      lock.unlock()
      return formatted
    }
  }
  
  /// Shared singleton instance
  nonisolated(unsafe) public static let shared = BenchmarkTimer()
  
  /// Private initializer to enforce singleton pattern
  private init() {}
  
  /// Dictionary storing all active timers by their ID
  private var timers: [String: Timing] = [:]
  
  /// Lock for thread-safe access to the timers dictionary
  private let timersLock = NSLock()
  
  /// Creates a new timer with the given ID and optional parent.
  /// - Parameters:
  ///   - id: Unique identifier for the timer
  ///   - parentId: Optional parent timer ID for hierarchical timing
  /// - Returns: The created or existing Timing instance, or nil if parent not found
  @discardableResult
  private func create(id: String, parent parentId: String? = nil) -> Timing? {
    timersLock.lock()
    defer { timersLock.unlock() }
    
    // Return existing timer if already created
    guard timers[id] == nil else { return timers[id] }
    
    var parentTiming: Timing?
    if let parentId {
      parentTiming = timers[parentId]
      guard parentTiming != nil else { return nil }
    }
    
    let timing = Timing(id: id, parent: parentTiming)
    timers[id] = timing
    return timing
  }
  
  /// Stops the timer with the given ID.
  /// - Parameter id: The timer ID to stop
  private func stop(id: String) {
    timersLock.lock()
    let timing = timers[id]
    timersLock.unlock()
    
    timing?.stop()
  }
  
  /// Prints the log for the timer with the given ID.
  /// - Parameter id: The timer ID to print
  private func printLogs(id: String) {
    timersLock.lock()
    let timing = timers[id]
    timersLock.unlock()
    
    timing?.log()
  }
  
  /// Resets all timers, clearing the internal state.
  private func reset() {
    timersLock.lock()
    timers = [:]
    timersLock.unlock()
  }
  
  /// Checks if a timer with the given ID exists.
  /// - Parameter id: The timer ID to check
  /// - Returns: The Timing instance if it exists, nil otherwise
  private func exists(id: String) -> Timing? {
    timersLock.lock()
    defer { timersLock.unlock() }
    return timers[id]
  }
  
  /// Retrieves the elapsed time for a timer.
  /// - Parameter id: The timer ID
  /// - Returns: The elapsed time in seconds, or nil if timer doesn't exist
  private func getTime(id: String) -> Double? {
    timersLock.lock()
    let timing = timers[id]
    timersLock.unlock()
    return timing?.deltaTime
  }
    
  #if DEBUG
  
  /// Prints all root-level timers and their hierarchical children.
  /// This method outputs timing results in a tree format, showing parent and child
  /// task relationships with proper indentation.
  ///
  /// **Note:** Only active in DEBUG builds.
  static public func print() {
    BenchmarkTimer.shared.timersLock.lock()
    let timersSnapshot = BenchmarkTimer.shared.timers
    BenchmarkTimer.shared.timersLock.unlock()
    
    for (key, timing) in timersSnapshot {
      if timing.parent == nil {
        BenchmarkTimer.shared.printLogs(id: key)
      }
    }
  }
  
  /// Resets all timers, clearing all timing data.
  /// **Note:** Only active in DEBUG builds.
  static public func reset() {
    BenchmarkTimer.shared.reset()
  }
  
  /// Starts or creates a timer with the given ID.
  /// If the timer already exists, it will be restarted. If it doesn't exist,
  /// a new timer will be created and started.
  /// - Parameters:
  ///   - id: Unique identifier for the timer
  ///   - parent: Optional parent timer ID for nested timing measurements
  /// **Note:** Only active in DEBUG builds. In RELEASE builds, this is a no-op.
  @inline(__always) public static func startTimer(_ id: String, _ parent: String? = nil) {
    if let timer = BenchmarkTimer.shared.create(id: id, parent: parent) {
      timer.startTimer()
    }
  }
  
  /// Stops the timer with the given ID and accumulates the elapsed time.
  /// - Parameter id: The timer ID to stop
  /// **Note:** Only active in DEBUG builds. In RELEASE builds, this is a no-op.
  @inline(__always) public static func stopTimer(_ id: String) {
    BenchmarkTimer.shared.stop(id: id)
  }
  
  /// Retrieves the accumulated time for a timer in seconds.
  /// - Parameter id: The timer ID
  /// - Returns: The elapsed time in seconds, or nil if timer doesn't exist
  /// **Note:** Only active in DEBUG builds. In RELEASE builds, returns 0.0.
  @inline(__always) public static func getTimeInSec(_ id: String) -> Double? {
    BenchmarkTimer.shared.getTime(id: id)
  }
  
  #else
    
  /// No-op in RELEASE builds
  static public func print() {}
  
  /// No-op in RELEASE builds
  static public func reset() {}
  
  /// No-op in RELEASE builds
  @inline(__always) public static func startTimer(_ id: String, _ parent: String? = nil) {}
  
  /// No-op in RELEASE builds
  @inline(__always) public static func stopTimer(_ id: String) {}
  
  /// No-op in RELEASE builds - returns 0.0
  @inline(__always) public static func getTimeInSec(_ id: String) -> Double? { 0.0 }
  
  #endif
}
