# Workload FSM Documentation: Transformer Training with Gradient Checkpointing

## Function Reference

**Function:** `Workload::iterate_hybrid_parallel_Transformer_fwd_in_bckwd()`  
**File:** `Workload.cc`  
**Purpose:** Orchestrates the execution of Transformer model training with hybrid parallelism and gradient checkpointing

---

## Overview

This function implements a **Finite State Machine (FSM)** that manages the complete training loop for Transformer models. It coordinates:

- **Forward propagation** through neural network layers
- **Backward propagation** (gradient computation)
- **Gradient checkpointing** with activation recomputation
- **Communication collectives** (AllReduce, AllGather, ReduceScatter)
- **Compute simulation** with accurate timing delays
- **Hybrid parallelism** (Tensor, Data, Expert parallelism support)

The function is **event-driven**: it returns control to the simulator after scheduling each operation, enabling concurrent execution of multiple workloads, network communications, and memory operations.

---

## Finite State Machine

### States

The FSM operates through **4 primary states** defined by the `LoopState` enum:

| State | Purpose | Layer Traversal | Communication Type |
|-------|---------|----------------|-------------------|
| **Forward_Pass** | Compute activations | 0 → SIZE-1 | AllGather, AllReduce (TP) |
| **Input_Gradient** | Compute ∂Loss/∂input | SIZE-1 → 0 | AllReduce, ReduceScatter (TP) |
| **Weight_Gradient** | Compute ∂Loss/∂weights | SIZE-1 → 0 | AllReduce (DP) |
| **Forward_In_BackPass** | Recompute activations | checkpoint → target | Same as Forward_Pass |

### State Transition Flow

```
Start
  ↓
┌─────────────────────────────────────────────────┐
│           FORWARD_PASS (index: 0→SIZE)          │
│  ┌─────────────────────────────────────────┐   │
│  │ 1. Wait for prev WG sync                │   │
│  │ 2. Load & apply compute delay           │   │
│  │ 3. Issue forward comm (Blocking)        │   │
│  │ 4. Advance to next layer                │   │
│  └─────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────┘
                     │ index >= SIZE
                     ↓
┌─────────────────────────────────────────────────┐
│       INPUT_GRADIENT (index: SIZE-1→0)          │
│  ┌─────────────────────────────────────────┐   │
│  │ Check: needs checkpoint recomputation?  │   │
│  │  ├─Yes→ Rewind to checkpoint layer      │   │
│  │  │      Transition to Forward_In_BackPass│   │
│  │  └─No → Continue normal backprop         │   │
│  │ 1. Load & apply compute delay           │   │
│  │ 2. Issue input grad comm (Blocking)     │   │
│  │ 3. Transition to Weight_Gradient        │   │
│  └─────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────┐
│      WEIGHT_GRADIENT (index: SIZE-1→0)          │
│  ┌─────────────────────────────────────────┐   │
│  │ 1. Load & apply compute delay           │   │
│  │ 2. Issue WG AllReduce (Non-Blocking)    │   │
│  │ 3. Wait for IG comm completion          │   │
│  │ 4. Decrement index                      │   │
│  │ 5. If index=-1: Next epoch              │   │
│  │    Else: Back to Input_Gradient         │   │
│  └─────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────┘
                     │ index == -1
                     ↓
               Next Epoch Start
```

---

## State 1: Forward Pass

### Entry Point

```cpp
if (current_state == LoopState::Forward_Pass)
```

### Execution Sequence

#### Step 1: Synchronization Barrier

```cpp
if (!layers[index]->is_weight_grad_comm_finished_blocking()) {
    return;
}
```

**Purpose:** Ensures weight gradients from the **previous training iteration** have been synchronized via AllReduce before using those weights for forward computation.

**Rationale:** In data parallelism, weights are updated after averaging gradients across all replicas. Layer `i` at step `N` requires weights that were updated by the weight gradient AllReduce from step `N-1`.

**Example Timeline:**
```
Step N-1: WG AllReduce for Layer 5 (in-flight)
Step N:   FP compute for Layer 5 (blocked)
          ↓ (WG completes)
          FP compute for Layer 5 (proceeds)
```

#### Step 2: Compute Simulation

```cpp
if (delay_loaded == false) {
    counter = layers[index]->get_fwd_pass_compute();
    delay_loaded = true;
}
if (counter > 0) {
    generator->try_register_event(this, EventType::Workload_Wait, NULL, counter);
    return;
}
```

**Mechanism:**
1. **Load compute time:** `get_fwd_pass_compute()` returns cycle count for operations like:
   - Matrix multiplications (QKV projections in attention)
   - Softmax, LayerNorm
   - Feed-forward network (FFN) computations
2. **Schedule delayed callback:** `try_register_event()` registers this workload to be called again after `counter` cycles
3. **Return immediately:** Simulator continues processing other events
4. **Resume after delay:** On next invocation, `counter = 0`, proceeds to next step

**Purpose:** Models GPU compute time without blocking the entire simulation, enabling accurate timing of overlapped operations.

#### Step 3: Forward Pass Communication

```cpp
if (!collective_issued) {
    collective_issued = true;
    if (layers[index]->fwd_pass_comm_size < 4096 && 
        layers[index]->fwd_pass_comm_size > 0) {
        layers[index]->fwd_pass_comm_size = 4096;
    }
    layers[index]->issue_forward_pass_comm(
        SchedulingPolicy::None, 
        CollectiveBarrier::Blocking);
    return;
}
```

**Operations:**

1. **Minimum message size enforcement:**
   - NCCL has significant protocol overhead for messages < 4KB
   - Padding small messages to 4KB models this overhead accurately
   - Reflects real-world behavior where tiny messages are inefficient

2. **Collective issuance:**
   - `issue_forward_pass_comm()` determines collective type from layer configuration
   - For Tensor Parallelism: AllGather to collect sharded activations
   - For Expert Parallelism: AllToAll for expert routing
   - Generates flow models via `MockNcclGroup::genFlowModels()`
   - Emits network packets through NS-3 backend

3. **Blocking barrier:**
   - `CollectiveBarrier::Blocking` means function returns but tracks completion
   - Next invocation checks if collective finished before proceeding
   - Necessary for Tensor Parallelism: next layer requires complete activations

**Communication Patterns:**
- **TP AllGather:** Gather tensor-parallel shards before next layer
- **TP AllReduce:** Reduce across TP dimension (e.g., attention output projection)
- **None:** Some layers have no FP communication (e.g., LayerNorm, activation functions)

#### Step 4: Layer Advancement

```cpp
index++;
delay_loaded = false;
collective_issued = false;
if (index >= SIZE) {
    current_state = LoopState::Input_Gradient;
    index--;
}
generator->register_event(this, EventType::General, NULL, 1);
return;
```

**Logic:**
1. Increment `index` to next layer
2. Reset state flags for next iteration
3. **Termination check:** If processed all SIZE layers:
   - Transition to `Input_Gradient` (start backward pass)
   - Set `index = SIZE-1` (last layer)
4. Schedule immediate callback (1 cycle) to continue FSM

---

## State 2: Weight Gradient

### Entry Point

```cpp
else if (current_state == LoopState::Weight_Gradient)
```

### Purpose

Compute **weight gradients** (∂Loss/∂W) and synchronize them across data parallel replicas using AllReduce.

### Execution Sequence

#### Step 1: Weight Gradient Computation

```cpp
if (delay_loaded == false) {
    counter = layers[index]->get_weight_grad_compute();
    delay_loaded = true;
}
if (counter > 0) {
    generator->try_register_event(this, EventType::Workload_Wait, NULL, counter);
    return;
}
```

**Computation includes:**
- Gradient of loss w.r.t. weight matrices (∂L/∂W = ∂L/∂out ⊗ input)
- For attention: Q, K, V projection weight gradients
- For FFN: Linear layer weight gradients
- Bias gradients

#### Step 2: Weight Gradient AllReduce

```cpp
if (!collective_issued) {
    collective_issued = true;
    layers[index]->issue_weight_grad_comm(
        SchedulingPolicy::FIFO, 
        CollectiveBarrier::Non_Blocking);
}
```

**Key Differences from Forward Pass:**

1. **FIFO Scheduling:**
   - Weight gradient AllReduces issued in **forward layer order** (0→SIZE)
   - Enables pipelining: While layer `i` reduces, layer `i+1` computes
   - Optimizes for memory pressure (earlier layers can free gradient buffers sooner)

2. **Non-Blocking Barrier:**
   - Doesn't wait for completion immediately
   - Allows overlapping WG AllReduce of layer `i` with WG compute of layer `i-1`
   - Improves performance through compute-communication overlap

**Communication:** Data Parallel AllReduce across all DP replicas to average weight gradients.

#### Step 3: Input Gradient Dependency Wait

```cpp
if (!layers[index]->is_input_grad_comm_finished_blocking()) {
    return;
}
```

**Dependency Chain:**
```
Input Gradient (layer i) → Weight Gradient (layer i)
```

**Why necessary:** Weight gradient computation requires:
- **Input activations** (from forward pass)
- **Output gradients** (∂Loss/∂output, from input gradient of layer i+1)

Formula: `∂L/∂W_i = activations_i^T × ∂L/∂output_i`

If input gradient communication hasn't completed, output gradients may not be available (in TP scenarios).

#### Step 4: Epoch Completion or Continue Backward

```cpp
index--;
if (index == -1) {
    index = 0;
    if (generator->id == 0) {
        std::cout << "pass: " << pass_counter 
                  << " finished at time: " << Sys::boostedTick() << std::endl;
    }
    pass_counter++;
    current_state = LoopState::Forward_Pass;
} else {
    current_state = LoopState::Input_Gradient;
}
```

**Logic:**
1. Decrement `index` (moving backward through layers)
2. **If index reaches -1:** All layers processed
   - Print epoch completion timestamp (only rank 0)
   - Increment epoch counter (`pass_counter`)
   - Reset to `Forward_Pass` state with `index = 0`
   - **Next iteration starts new epoch**
3. **Otherwise:** Continue to `Input_Gradient` for layer `index`

---

## State 3: Input Gradient

### Entry Point

```cpp
else if (current_state == LoopState::Input_Gradient)
```

### Purpose

Compute **input gradients** (∂Loss/∂input) during backward propagation, with support for gradient checkpointing and activation recomputation.

### Execution Sequence

#### Step 1: Checkpoint Recomputation Trigger

```cpp
if (layers[index]->needs_fwd_in_bckwd_initiation && !checkpoint_initiated) {
    int tmp = index;
    while (!layers[index--]->is_checkpoint);
    index++;
    current_state = LoopState::Forward_In_BackPass;
    checkpoint_initiated = true;
    generator->register_event(this, EventType::General, NULL, 1);
    if (generator->id == 0) {
        std::cout << "***** info, initiating fwd_in_bkwd starting from layer:"
                  << index << " to layer: " << tmp
                  << " ,at time: " << Sys::boostedTick() << std::endl;
    }
    return;
}
```

#### Gradient Checkpointing Deep Dive

**Problem:** Storing all intermediate activations for backward pass requires memory proportional to model depth.

**Solution:** **Gradient Checkpointing** (also called activation checkpointing or rematerialization):
1. **Forward Pass:** Only save activations at designated **checkpoint layers** (e.g., every 4th layer)
2. **Backward Pass:** When reaching a non-checkpointed layer:
   - **Recompute forward pass** from last checkpoint to current layer
   - Use recomputed activations for gradient computation
   - Discard recomputed activations after use

**Memory-Compute Tradeoff:**
- **Memory saved:** ~75% (only 1/4 of activations stored)
- **Compute overhead:** ~33% (recompute 3/4 of forward passes once)
- **Net benefit:** Enables training much larger models on same hardware

**Code Flow:**

1. **Trigger condition:** `layers[index]->needs_fwd_in_bckwd_initiation == true`
   - Set in workload file via `checkpoint_initiates:` parameter
   - Marks layers that require recomputation

2. **Find checkpoint:** Scan backward (`index--`) until finding `layers[index]->is_checkpoint == true`
   - Checkpoints marked in workload file via `checkpoints:` parameter

3. **Transition:** Switch to `Forward_In_BackPass` state to recompute forward

4. **Flag:** Set `checkpoint_initiated = true` to prevent re-triggering

**Example Scenario:**
```
Layers: [0][1][2][3][4][5][6][7][8][9][10][11]
Checkpoints:    ^       ^       ^         ^     (layers 2, 5, 8, 11)
Recompute:          ^               ^           (initiate at 3, 7)

Backward at layer 7:
1. Detects needs_fwd_in_bckwd_initiation = true
2. Rewinds to layer 5 (last checkpoint)
3. Enters Forward_In_BackPass state
4. Recomputes FP: layer 5 → 6 → 7
5. Returns to Input_Gradient at layer 7 with fresh activations
6. Computes gradients for layer 7
7. Continues backward to layer 6, 5, ...
```

#### Step 2: Input Gradient Computation

```cpp
if (delay_loaded == false) {
    counter = layers[index]->get_input_grad_compute();
    delay_loaded = true;
}
if (counter > 0) {
    generator->try_register_event(this, EventType::Workload_Wait, NULL, counter);
    return;
}
```

**Computations:**
- Backpropagation through layer operations
- For linear layers: `∂L/∂input = ∂L/∂output × W^T`
- For attention: Backprop through softmax, QKV projections
- For activation functions: Element-wise gradient multiplication

#### Step 3: Input Gradient Communication

```cpp
if (!collective_issued) {
    collective_issued = true;
    layers[index]->issue_input_grad_comm(
        SchedulingPolicy::LIFO, 
        CollectiveBarrier::Blocking);
    return;
}
```

**Characteristics:**

1. **LIFO Scheduling (Last-In-First-Out):**
   - Input gradients issued in **reverse layer order** (SIZE→0)
   - Matches backward propagation order
   - Optimizes dependency chain: layer `i-1` waits for layer `i`'s gradient

2. **Blocking Barrier:**
   - Must complete before proceeding
   - Next layer (i-1) requires this layer's gradient as input

**Communication Patterns:**
- **TP ReduceScatter:** Reduce gradients and scatter across TP ranks
- **TP AllReduce:** For certain layer types requiring full gradient sync
- **EP AllToAll:** Expert gradient routing in MoE models

#### Step 4: Transition to Weight Gradient

```cpp
checkpoint_initiated = false;
collective_issued = false;
delay_loaded = false;
current_state = LoopState::Weight_Gradient;
generator->register_event(this, EventType::General, NULL, 1);
```

**Logic:**
1. Reset all state flags
2. Transition to `Weight_Gradient` for **same layer** `index`
3. Note: `index` is **not decremented** here (happens in Weight_Gradient state)
4. Schedule immediate callback to continue

**State Flow:**
```
Input_Gradient (layer i) → Weight_Gradient (layer i) → Input_Gradient (layer i-1)
```

---

## State 4: Forward In BackPass

### Entry Point

```cpp
else if (current_state == LoopState::Forward_In_BackPass)
```

### Purpose

**Recompute forward pass** for checkpointed segments during backward propagation to regenerate activations needed for gradient computation.

### Implementation

The code is **nearly identical** to `Forward_Pass` state:

```cpp
if (!layers[index]->is_weight_grad_comm_finished_blocking()) {
    return;
}
if (delay_loaded == false) {
    counter = layers[index]->get_fwd_pass_compute();
    delay_loaded = true;
}
if (counter > 0) {
    generator->try_register_event(this, EventType::Workload_Wait, NULL, counter);
    return;
}
if (!collective_issued) {
    collective_issued = true;
    layers[index]->issue_forward_pass_comm(
        SchedulingPolicy::None, 
        CollectiveBarrier::Blocking);
    return;
}

index++;
delay_loaded = false;
collective_issued = false;

// [KEY DIFFERENCE] Exit condition
if (layers[index]->needs_fwd_in_bckwd_initiation) {
    current_state = LoopState::Input_Gradient;
}
generator->register_event(this, EventType::General, NULL, 1);
return;
```

### Key Differences from Normal Forward Pass

| Aspect | Forward_Pass | Forward_In_BackPass |
|--------|--------------|---------------------|
| **Entry point** | Layer 0 | Last checkpoint layer |
| **Exit condition** | `index >= SIZE` | `layers[index]->needs_fwd_in_bckwd_initiation` |
| **Next state** | Input_Gradient (start BP) | Input_Gradient (resume BP) |
| **Purpose** | Initial forward propagation | Activation regeneration |
| **Activation usage** | Stored for BP | Used immediately, then discarded |

### Execution Flow Example

```
Configuration:
- Checkpoint layers: [2, 5, 8]
- Recomputation needed at layer 7

Step-by-step execution:
1. Backward pass reaches layer 7
2. Input_Gradient detects needs_fwd_in_bckwd_initiation
3. Rewinds to layer 5 (checkpoint)
4. [ENTERS Forward_In_BackPass STATE]
5. Executes FP for layer 5 (compute + comm)
6. index++ → layer 6
7. Executes FP for layer 6
8. index++ → layer 7
9. Executes FP for layer 7
10. Detects layers[7]->needs_fwd_in_bckwd_initiation == true
11. [EXITS to Input_Gradient STATE]
12. Now at layer 7 with fresh activations
13. Computes input gradients for layer 7 using recomputed activations
14. Continues normal backward pass
```

### Memory Pattern

```
Normal Forward Pass (without checkpointing):
Memory: [Act0][Act1][Act2][Act3][Act4][Act5][Act6][Act7][Act8]...

With Checkpointing:
After FP:  [    ][    ][Act2][    ][    ][Act5][    ][    ][Act8]
           └─────┘      └────────────┘      └────────────┘
           Discarded    Only checkpoints    Discarded

During BP at layer 7:
1. Load Act5 (checkpoint)
2. Recompute → [Act5][Act6][Act7]
3. Use Act7 for gradients
4. Discard Act6, Act7
5. Keep Act5 (still needed for layer 6)
```

---

## Key Design Patterns

### 1. Event-Driven Simulation Architecture

```cpp
generator->register_event(this, EventType::General, NULL, 1);
generator->try_register_event(this, EventType::Workload_Wait, NULL, counter);
```

**Pattern:** Cooperative multitasking via event scheduling

**Benefits:**
- **Concurrency:** Multiple workloads, network ops, memory transfers run simultaneously
- **Accuracy:** Models true asynchronous behavior of GPUs and network
- **Scalability:** Simulator can handle thousands of concurrent events
- **No busy-waiting:** CPU-efficient simulation

**Event Types:**
- `EventType::General`: Continue FSM immediately (next simulation cycle)
- `EventType::Workload_Wait`: Resume after compute delay (models GPU busy time)

### 2. Flag-Based State Persistence

```cpp
bool delay_loaded;        // Has compute time been loaded for current step?
bool collective_issued;   // Has communication been initiated?
bool checkpoint_initiated; // Has recomputation started (prevents re-trigger)?
```

**Purpose:** Track sub-state within each FSM state across multiple function invocations

**Why necessary:** Function returns frequently (after each event schedule), but execution context must persist

**Pattern:**
```cpp
// First call
if (!delay_loaded) {
    counter = get_compute_time();
    delay_loaded = true;  // Mark as loaded
}
// ... schedule event, return ...

// Second call (after delay)
if (counter > 0) {
    // ... counter is now 0 ...
}
// Proceed to next step

// Reset at end
delay_loaded = false;  // Ready for next layer
```

### 3. Blocking vs Non-Blocking Communication

| Barrier Type | Used For | Scheduling | Overlap Possible |
|--------------|----------|------------|------------------|
| **Blocking** | FP, IG comm | None, LIFO | No - strict dependencies |
| **Non-Blocking** | WG comm | FIFO | Yes - with next layer compute |

**Blocking Example (FP):**
```
Layer 5 FP compute → Layer 5 FP comm (blocking)
                     ↓ (must complete)
                     Layer 6 FP compute
```

**Non-Blocking Example (WG):**
```
Layer 5 WG compute → Layer 5 WG comm (non-blocking, starts)
                     ↓ (continues immediately)
Layer 4 WG compute   (overlaps with Layer 5 WG comm)
```

### 4. Scheduling Policies

| Policy | Usage | Order | Rationale |
|--------|-------|-------|-----------|
| **None** | FP comm | Natural order | No optimization needed |
| **FIFO** | WG comm | Forward order (0→SIZE) | Enables pipelining |
| **LIFO** | IG comm | Reverse order (SIZE→0) | Matches BP dependencies |

**FIFO Benefit:**
```
Issue WG AllReduce for layers in order:
Layer 0 WG (start) → Layer 1 WG (start) → Layer 2 WG (start)
     ↓ (completes)        ↓ (completes)        ↓ (completes)
Free gradient buffer  Free gradient buffer  Free gradient buffer
```
Memory pressure reduced as earlier layers finish first.

### 5. Index Navigation Patterns

| State | Direction | Start | End | Increment |
|-------|-----------|-------|-----|-----------|
| Forward_Pass | Forward | 0 | SIZE-1 | `index++` |
| Input_Gradient | Backward | SIZE-1 | 0 | *(stays same)* |
| Weight_Gradient | Backward | SIZE-1 | -1 | `index--` |
| Forward_In_BackPass | Forward | checkpoint | target | `index++` |

**Index Flow Across States:**
```
FP:  0 → 1 → 2 → ... → SIZE-1
     ↓
IG:  SIZE-1 (no change)
     ↓
WG:  SIZE-1 → SIZE-2 → ... → 0 → -1 (next epoch)
```

---

## Synchronization Dependencies

### Cross-Layer Dependencies

```
Forward Pass:
Layer i-1 output → Layer i input (data dependency)

Backward Pass:
Layer i+1 gradient → Layer i gradient computation (chain rule)

Weight Update:
Layer i input gradient → Layer i weight gradient (requires both input and output gradients)
```

### Cross-Iteration Dependencies

```
Iteration N:
  Layer i WG AllReduce (started)
  ...

Iteration N+1:
  Layer i FP compute (blocked on WG AllReduce from iter N)
  ↓ (WG completes)
  Layer i FP compute (proceeds with updated weights)
```

### Checkpoint Recomputation Dependencies

```
Backward at Layer 7:
1. Requires activations from Layer 6
2. Layer 6 not checkpointed → Recompute needed
3. Find last checkpoint (Layer 5)
4. Recompute FP: Layer 5 → 6 → 7
5. Now Layer 7 BP can proceed
```

---

## Performance Optimizations

### 1. Compute-Communication Overlap

**Non-Blocking WG AllReduce** enables:
```
Timeline:
|--- Layer 5 WG compute ---|
                            |--- Layer 5 WG AllReduce ---|
                      |--- Layer 4 WG compute ---|
                                                  |--- Layer 4 WG AllReduce ---|
```

**Benefit:** Hides communication latency behind compute, reducing total training time.

### 2. Gradient Checkpointing

**Memory-Compute Tradeoff:**
- **Without checkpointing:** Store N layer activations = O(N) memory
- **With checkpointing (every K layers):** Store N/K activations, recompute (K-1)/K of FP once = O(N/K) memory, 1.33× compute
- **Example:** 48-layer Transformer, checkpoints every 4 layers:
  - Memory: 12 checkpoints instead of 48 activations (75% reduction)
  - Compute: Recompute 36 layers once (33% increase)
  - **Net:** Can fit 4× larger model in same memory

### 3. Message Size Padding

```cpp
if (layers[index]->fwd_pass_comm_size < 4096 && 
    layers[index]->fwd_pass_comm_size > 0) {
    layers[index]->fwd_pass_comm_size = 4096;
}
```

**Rationale:** NCCL protocol overhead dominates for tiny messages. Padding to 4KB:
- Models real-world inefficiency
- Accurately reflects that small messages don't save proportional time
- Prevents unrealistic speedup predictions for fine-grained communication

### 4. FIFO WG Scheduling

**Memory Pressure Reduction:**
```
FIFO Order:
Layer 0 WG → Layer 1 WG → Layer 2 WG → ...
    ↓ (free)     ↓ (free)     ↓ (free)
Gradient buffers released in order, reducing peak memory
```

**Contrast with LIFO:**
```
LIFO Order:
Layer N WG → Layer N-1 WG → Layer N-2 WG → ...
    ↓ (free)     ↓ (free)      ↓ (free)
All gradients held until last layer completes (higher peak memory)
```

---

## Error Handling and Assertions

### Entry Validation

```cpp
assert(index >= 0);
assert(index < SIZE);
```

**Prevents:** Index out-of-bounds errors that could cause segfaults or undefined behavior

**When would this fail:**
- Logic error in state transitions
- Corruption of `index` variable
- Incorrect termination condition

### Simulation Termination Check

```cpp
check_for_sim_end();
```

**Purpose:** Every iteration checks if training complete (`pass_counter == TOTAL_PASS`)

**Actions if complete:**
- Waits for all in-flight communications to finish
- Calls `report()` to output statistics
- Notifies `Sys` that workload finished
- Triggers simulator cleanup

---

## Debugging and Logging

### Debug Log Example

```cpp
NcclLog->writeLog(NcclLogLevel::DEBUG, 
    "workload::call fwd_pass register_event EventType::General");
```

**Purpose:** Trace execution flow for debugging FSM transitions

### Epoch Completion Log

```cpp
if (generator->id == 0) {
    std::cout << "pass: " << pass_counter 
              << " finished at time: " << Sys::boostedTick() << std::endl;
}
```

**Output Example:**
```
pass: 0 finished at time: 52847291
pass: 1 finished at time: 105694582
pass: 2 finished at time: 158541873
```

**Analysis:** Measure per-epoch time, detect performance anomalies

### Checkpoint Recomputation Log

```cpp
if (generator->id == 0) {
    std::cout << "***** info, initiating fwd_in_bckwd starting from layer:"
              << index << " to layer: " << tmp
              << " ,at time: " << Sys::boostedTick() << std::endl;
}
```

**Output Example:**
```
***** info, initiating fwd_in_bckwd starting from layer:5 to layer:7 ,at time: 3784293
```

**Debugging Use:** Verify checkpoint recomputation triggers correctly

---

## State Transition Summary Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         EPOCH LOOP                              │
│                                                                 │
│  ┌────────────────┐                                            │
│  │  Forward_Pass  │ index: 0 → SIZE-1                          │
│  │  (compute FP)  │ comm: Blocking                             │
│  └───────┬────────┘                                            │
│          │ index >= SIZE                                        │
│          ↓                                                      │
│  ┌────────────────┐                                            │
│  │ Input_Gradient │ index: SIZE-1 (no change)                 │
│  │  (compute IG)  │ comm: Blocking (LIFO)                     │
│  └───────┬────────┘                                            │
│          │ checkpoint needed?                                   │
│          ├─Yes─▶┌──────────────────────┐                      │
│          │      │ Forward_In_BackPass  │                      │
│          │      │  (recompute FP)      │                      │
│          │      └──────────┬───────────┘                      │
│          │                 │ target reached                    │
│          ◀─────────────────┘                                  │
│          │                                                      │
│          ↓                                                      │
│  ┌────────────────┐                                            │
│  │Weight_Gradient │ index: SIZE-1 → -1                        │
│  │  (compute WG)  │ comm: Non-Blocking (FIFO)                │
│  └───────┬────────┘                                            │
│          │ index > 0                                            │
│          ├─────────────────▶ Input_Gradient (loop back)       │
│          │ index == -1                                          │
│          └─────────────────▶ Forward_Pass (next epoch)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Code Structure Overview

```cpp
void Workload::iterate_hybrid_parallel_Transformer_fwd_in_bckwd() {
    // Entry validation
    assert(index >= 0 && index < SIZE);
    check_for_sim_end();

    // State machine dispatcher
    if (current_state == LoopState::Forward_Pass) {
        // 1. Wait for prev WG sync
        // 2. Load & apply compute delay
        // 3. Issue FP comm (Blocking)
        // 4. Advance index
        // 5. Transition to IG if index >= SIZE
    }
    else if (current_state == LoopState::Weight_Gradient) {
        // 1. Load & apply compute delay
        // 2. Issue WG comm (Non-Blocking, FIFO)
        // 3. Wait for IG comm completion
        // 4. Decrement index
        // 5. Epoch complete or continue IG
    }
    else if (current_state == LoopState::Input_Gradient) {
        // 1. Check checkpoint recomputation trigger
        // 2. Load & apply compute delay
        // 3. Issue IG comm (Blocking, LIFO)
        // 4. Transition to WG
    }
    else if (current_state == LoopState::Forward_In_BackPass) {
        // 1. Recompute FP (same as Forward_Pass)
        // 2. Exit when reaching target layer
        // 3. Return to Input_Gradient
    }
}
```

---

## Integration with Simulator Components

### Workload → Layer

```cpp
layers[index]->get_fwd_pass_compute()         // Query compute time
layers[index]->issue_forward_pass_comm(...)   // Trigger collective
layers[index]->is_weight_grad_comm_finished_blocking()  // Check completion
```

### Workload → Sys (Generator)

```cpp
generator->register_event(this, EventType::General, NULL, 1)  // Schedule callback
generator->try_register_event(...)            // Schedule with delay
generator->id                                 // Rank ID
Sys::boostedTick()                            // Current simulation time
```

### Layer → Sys → MockNccl → NS-3

```
Layer::issue_forward_pass_comm()
  ↓
Sys::generate_collective()
  ↓
MockNcclComm::getFlowModels()
  ↓
MockNcclGroup::genFlowModels()
  ↓
Sys::sim_send() / sim_recv()
  ↓
NS-3 RdmaDriver (network simulation)
```

---

## Usage Example

### Workload Configuration

```
HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: 8 ep: 1 pp: 1 vpp: 1 ga: 1 all_gpus: 64 checkpoints: 4 12 20 28 checkpoint_initiates: 4 13 21 29
32
embedding      -1  50000   NONE         0  1        NONE         0  50000   NONE         0  100
attention_0    -1  120000  ALLGATHER    2M 120000   REDUCESCATTER 2M 120000  ALLREDUCE    8M 100
mlp_0          -1  180000  NONE         0  180000   NONE         0  180000  ALLREDUCE    16M 100
...
```

### Execution Trace

```
[Forward_Pass] Layer 0 (embedding): compute=50000 cycles, comm=NONE
[Forward_Pass] Layer 1 (attention_0): compute=120000 cycles, comm=ALLGATHER 2MB
[Forward_Pass] Layer 2 (mlp_0): compute=180000 cycles, comm=NONE
...
[Forward_Pass] Layer 31: Complete, transition to Input_Gradient

[Input_Gradient] Layer 31: compute=180000 cycles, comm=REDUCESCATTER 2MB
...
[Input_Gradient] Layer 13: Checkpoint recomputation needed
***** info, initiating fwd_in_bckwd starting from layer:12 to layer:13 ,at time: 9483748

[Forward_In_BackPass] Layer 12: compute=180000 cycles, comm=ALLGATHER 2MB
[Forward_In_BackPass] Layer 13: Complete, return to Input_Gradient

[Input_Gradient] Layer 13: compute=120000 cycles, comm=REDUCESCATTER 2MB
...
[Weight_Gradient] Layer 31: compute=180000 cycles, comm=ALLREDUCE (FIFO, Non-Blocking)
...
[Weight_Gradient] Layer 0: Complete
pass: 0 finished at time: 52847291

[Forward_Pass] Layer 0: Start epoch 1
...
```

---

## Performance Considerations

### Critical Paths

1. **Forward pass latency:** Sum of all layer compute times + blocking communication
2. **Backward pass latency:** Similar to forward, plus checkpoint recomputation overhead
3. **Synchronization barriers:** WG AllReduce completion blocks next epoch start

### Optimization Opportunities

1. **Overlapping WG AllReduce with compute:** Non-blocking WG communication
2. **Pipelining layers:** FIFO scheduling enables pipeline parallelism
3. **Reducing checkpoint recomputation:** Strategic checkpoint placement minimizes recompute
4. **Message coalescing:** Batching small messages to 4KB+ reduces protocol overhead

### Scaling Behavior

- **Strong scaling (fixed model, more GPUs):** Communication becomes bottleneck
- **Weak scaling (proportional model/GPUs):** Compute-communication ratio maintained
- **Checkpoint scaling:** Recompute overhead increases with model depth

---

## Common Pitfalls and Debugging

### Infinite Loops

**Symptom:** FSM never advances  
**Cause:** Blocking communication never completes  
**Debug:** Check MockNcclLog for flow model generation errors

### Index Out of Bounds

**Symptom:** Assertion failure on `assert(index < SIZE)`  
**Cause:** Incorrect state transition logic  
**Debug:** Trace index modifications in each state

### Missed Synchronization

**Symptom:** Numerical errors or silent corruption  
**Cause:** Reading stale data (e.g., using pre-AllReduce weights)  
**Debug:** Verify all `is_*_comm_finished_blocking()` checks present

### Checkpoint Configuration Errors

**Symptom:** Recomputation never triggers or triggers incorrectly  
**Cause:** Mismatched `checkpoints:` and `checkpoint_initiates:` in workload file  
**Debug:** Check logs for "initiating fwd_in_bckwd" messages

---

## Future Extensions for Pipeline Parallelism

To support Pipeline Parallelism (PP), this function needs modifications:

1. **Layer ownership check:** Skip layers not owned by this PP stage
2. **PP send/recv operations:** Issue point-to-point activations/gradients between stages
3. **Microbatch scheduling:** Interleave forward/backward for multiple microbatches
4. **Bubble time modeling:** Account for pipeline bubbles at start/end

**Proposed modifications:**

```cpp
if (current_state == LoopState::Forward_Pass) {
    // [ADD] Skip layers not owned by this stage
    if (pp_enabled && layers[index]->pipeline_stage != my_stage) {
        index++;
        generator->register_event(this, EventType::General, NULL, 1);
        return;
    }

    // [ADD] PP receive from previous stage (before compute)
    if (pp_enabled && layers[index]->needs_pp_recv) {
        layers[index]->issue_pp_recv_forward(current_microbatch);
    }

    // ... existing compute logic ...

    // [ADD] PP send to next stage (after compute)
    if (pp_enabled && layers[index]->needs_pp_send) {
        layers[index]->issue_pp_send_forward(current_microbatch);
    }

    // ... existing advancement logic ...
}
```

---

*Documentation generated for SimAI/ASTRA-SIM Workload FSM - Version: Transformer with Gradient Checkpointing*
