//
//  LSTMKernels.swift
//  Kokoro-tts-lib
//
//  Custom Metal kernels for optimized LSTM operations
//
import Foundation
import MLX

/// Provides optimized Metal kernels for LSTM operations
public enum LSTMKernels {
  
  /// Cached kernel for fused LSTM gate computation
  nonisolated(unsafe) private static var fusedGatesKernel: MLXFast.MLXFastKernel?
  
  /// Creates the fused LSTM gates kernel if not already cached
  private static func getFusedGatesKernel() -> MLXFast.MLXFastKernel {
    if let kernel = fusedGatesKernel {
      return kernel
    }
    
    // Create kernel that computes:
    // i = sigmoid(gates[0:H])
    // f = sigmoid(gates[H:2H])
    // g = tanh(gates[2H:3H])
    // o = sigmoid(gates[3H:4H])
    // new_cell = f * cell + i * g
    // new_hidden = o * tanh(new_cell)
    let source = """
      uint tid = thread_position_in_grid.x;
      uint batch = thread_position_in_grid.y;
      
      // Input shape: [batch, 4*hidden_size] for ifgo
      // Cell shape: [batch, hidden_size]
      uint H = cell_shape[1];
      uint idx = batch * H + tid;
      
      if (tid < H) {
        // Read gate values
        T i_val = ifgo[batch * 4 * H + tid];
        T f_val = ifgo[batch * 4 * H + H + tid];
        T g_val = ifgo[batch * 4 * H + 2 * H + tid];
        T o_val = ifgo[batch * 4 * H + 3 * H + tid];
        T c_val = cell[idx];
        
        // Apply activations
        T i_gate = T(1) / (T(1) + metal::exp(-i_val));  // sigmoid
        T f_gate = T(1) / (T(1) + metal::exp(-f_val));  // sigmoid
        T g_gate = metal::tanh(g_val);                   // tanh
        T o_gate = T(1) / (T(1) + metal::exp(-o_val));  // sigmoid
        
        // Compute new cell state
        T new_c = f_gate * c_val + i_gate * g_gate;
        
        // Compute new hidden state
        T new_h = o_gate * metal::tanh(new_c);
        
        // Write outputs
        new_cell[idx] = new_c;
        new_hidden[idx] = new_h;
      }
    """
    
    let kernel = MLXFast.metalKernel(
      name: "fused_lstm_step",
      inputNames: ["ifgo", "cell"],
      outputNames: ["new_cell", "new_hidden"],
      source: source,
      ensureRowContiguous: true
    )
    
    fusedGatesKernel = kernel
    return kernel
  }
  
  /// Computes a single LSTM step with fused gate operations
  /// - Parameters:
  ///   - ifgo: Combined gate values [batch, 4*hidden_size]
  ///   - cell: Current cell state [batch, hidden_size]
  /// - Returns: Tuple of (new_cell, new_hidden)
  public static func fusedLSTMStep(ifgo: MLXArray, cell: MLXArray) -> (MLXArray, MLXArray)? {
    let kernel = getFusedGatesKernel()
    
    let batchSize = cell.shape[0]
    let hiddenSize = cell.shape[1]
    
    // Grid: one thread per hidden unit, one block per batch
    let grid = (hiddenSize, batchSize, 1)
    let threadGroup = (min(256, hiddenSize), 1, 1)
    
    do {
      let outputs = kernel(
        [ifgo, cell],
        template: [("T", DType.float32)],
        grid: grid,
        threadGroup: threadGroup,
        outputShapes: [[batchSize, hiddenSize], [batchSize, hiddenSize]],
        outputDTypes: [.float32, .float32]
      )
      
      return (outputs[0], outputs[1])
    } catch {
      return nil
    }
  }
}
