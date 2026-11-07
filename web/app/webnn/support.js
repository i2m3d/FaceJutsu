// WebNN (Web Neural Network API) feature detection, usable from the page and from workers.
//
// Chrome ships WebNN behind a flag / origin trial: chrome://flags/#web-machine-learning-neural-network
// (or --enable-features=WebMachineLearningNeuralNetwork).

export const WEBNN_FLAG = "chrome://flags/#web-machine-learning-neural-network";

export function hasWebNN() {
  return typeof navigator !== "undefined" && !!navigator.ml && typeof MLGraphBuilder !== "undefined";
}

// device: a GPUDevice to share with WebGPU (tensors can then be lent to it, see gpuInterop), or
// null for a context of its own
export async function createWebNNContext(device = null) {
  if (!hasWebNN()) throw new Error(`WebNN is not enabled in this browser (turn on ${WEBNN_FLAG})`);
  if (device) {
    try {
      return await navigator.ml.createContext(device);
    } catch (e) {
      console.warn("WebNN can't share the WebGPU device, using a context of its own:", e.message);
    }
  }
  // deviceType is ignored by current Chrome builds (powerPreference picks the GPU) but older ones need it
  return navigator.ml.createContext({ powerPreference: "high-performance", deviceType: "gpu" });
}

// Whether tensors of a context made from a GPUDevice can be lent to WebGPU: createExportableTensor
// makes a tensor that exportToGPU turns into a GPUBuffer of that device; WebNN can use the tensor
// again once the buffer is destroyed. Lets WebGPU write a graph's inputs and read its outputs
// without a copy through the CPU.
export const gpuInterop = (context) => typeof context.createExportableTensor === "function" && typeof context.exportToGPU === "function";

const types = (limits, op, arg) => limits?.[op]?.[arg]?.dataTypes || [];

// The compute type for the convolutions: "float16" when supported (and wanted), else "float32".
export function computeType(context, prefer = "float16") {
  const L = context.opSupportLimits?.() || {};
  const f16 = types(L, "conv2d", "input").includes("float16") && types(L, "conv2d", "filter").includes("float16");
  return prefer === "float16" && f16 ? "float16" : "float32";
}

