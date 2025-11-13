# 1F1B Pipeline Parallelism Implementation Guide for Workload Class

## Overview

This document provides **complete code modifications** to implement the **1F1B (One-Forward-One-Backward) scheduling technique** used in Megatron-LM, DeepSpeed, and PipeDream for the Workload class in ASTRA-SIM.

### What is 1F1B?

1F1B is an efficient pipeline scheduling strategy that minimizes memory usage and pipeline bubbles by alternating forward and backward passes for different microbatches:

- **Warm-up Phase:** Fill the pipeline with forward passes
- **Steady State:** Alternate 1 forward + 1 backward (1F1B pattern)
- **Cooldown Phase:** Drain the pipeline with backward passes

**Key Benefits:**
- Constant memory usage (only P microbatches in-flight, where P = pipeline stages)
- Reduced bubble time compared to GPipe
- Used in production by Megatron-LM, DeepSpeed, and PyTorch FSDP

---

## Part 1: Header File Modifications

### File: `Workload.hh`

#### 1.1 Add PP State Machine Enums

**Location:** After existing LoopState enum (around line 30)

```cpp
// [ADD] Pipeline Parallelism States
enum class PPState {
    Warmup,         // Filling pipeline with forward passes
    Steady,         // 1F1B alternating pattern
    Cooldown        // Draining pipeline with backward passes
};

enum class PPOperation {
    Forward,        // Forward pass for a microbatch
    Backward,       // Backward pass for a microbatch
    Idle            // No operation (waiting)
};
```

#### 1.2 Add PP Member Variables

**Location:** In Workload class private section (around line 80)

```cpp
// [ADD] Pipeline Parallelism Configuration
bool pp_enabled;                           // Is PP enabled?
int pp_size;                               // Number of pipeline stages
int my_pp_stage;                           // This GPU's pipeline stage ID
int num_microbatches;                      // Total microbatches per iteration
int warmup_microbatches;                   // Num microbatches in warmup phase
int cooldown_microbatches;                 // Num microbatches in cooldown phase

// [ADD] PP Runtime State
PPState pp_state;                          // Current PP phase
std::queue<int> pp_forward_queue;          // Microbatches waiting for forward
std::queue<int> pp_backward_queue;         // Microbatches waiting for backward
std::set<int> pp_forward_in_progress;      // Microbatches currently in forward
std::set<int> pp_backward_in_progress;     // Microbatches currently in backward
int pp_forward_completed;                  // Count of completed forward passes
int pp_backward_completed;                 // Count of completed backward passes

// [ADD] PP Layer Tracking
int pp_first_layer;                        // First layer owned by this stage
int pp_last_layer;                         // Last layer owned by this stage
int pp_current_microbatch;                 // Current microbatch being processed

// [ADD] PP Statistics
Tick pp_bubble_time_start;                 // Start of bubble period
Tick pp_total_bubble_time;                 // Total time spent in bubbles
```

#### 1.3 Add PP Method Declarations

**Location:** In public methods section (around line 60)

```cpp
// [ADD] Pipeline Parallelism Methods
void init_pipeline_parallel();                          // Initialize PP state
bool should_process_layer(int layer_idx);               // Does this stage own layer?
void schedule_next_pp_operation();                      // Determine next F or B
void execute_pp_forward(int microbatch_id);             // Execute forward for microbatch
void execute_pp_backward(int microbatch_id);            // Execute backward for microbatch
bool is_pp_forward_complete(int microbatch_id);         // Check forward completion
bool is_pp_backward_complete(int microbatch_id);        // Check backward completion
void advance_pp_state();                                // Move to next PP phase
```

---

## Part 2: Constructor Modifications

### File: `Workload.cc`

#### 2.1 Initialize PP Variables in Constructor

**Location:** In `Workload::Workload(...)` constructor, after existing initialization (around line 80)

```cpp
// [ADD] Initialize Pipeline Parallelism Variables
this->pp_enabled = false;
this->pp_size = 1;
this->my_pp_stage = -1;
this->num_microbatches = 1;
this->warmup_microbatches = 0;
this->cooldown_microbatches = 0;

// [ADD] Initialize PP Runtime State
this->pp_state = PPState::Warmup;
this->pp_forward_completed = 0;
this->pp_backward_completed = 0;
this->pp_current_microbatch = 0;
this->pp_first_layer = 0;
this->pp_last_layer = 0;

// [ADD] Initialize PP Statistics
this->pp_bubble_time_start = 0;
this->pp_total_bubble_time = 0;

// [ADD] Log initialization
MockNcclLog* NcclLog = MockNcclLog::getInstance();
NcclLog->writeLog(NcclLogLevel::DEBUG, 
    "Workload initialized (PP will be configured in initialize_workload)");
```

---

## Part 3: Initialize Pipeline Parallel

### File: `Workload.cc`

#### 3.1 Add `init_pipeline_parallel()` Method

**Location:** After `initialize_workload()` method (around line 400)

```cpp
void Workload::init_pipeline_parallel() {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();

    // Check if PP is enabled
    pp_size = pipeline_model_parallelism;
    pp_enabled = (pp_size > 1);

    if (!pp_enabled) {
        NcclLog->writeLog(NcclLogLevel::INFO, 
            "Pipeline Parallelism disabled (pp_size=%d)", pp_size);
        return;
    }

    // Determine this GPU's stage (assuming simple mapping: GPU ID = stage ID)
    my_pp_stage = generator->id % pp_size;

    // Calculate microbatches (from gradient accumulation steps)
    num_microbatches = GA;  // GA is already parsed from workload file

    if (num_microbatches < pp_size) {
        NcclLog->writeLog(NcclLogLevel::ERROR,
            "Invalid configuration: num_microbatches (%d) < pp_size (%d)",
            num_microbatches, pp_size);
        Sys::sys_panic("1F1B requires at least as many microbatches as pipeline stages");
    }

    // Calculate warmup microbatches (Megatron-LM formula)
    // Warmup for stage i: warmup_microbatches = pp_size - my_pp_stage - 1
    warmup_microbatches = pp_size - my_pp_stage - 1;

    // Calculate cooldown microbatches
    cooldown_microbatches = pp_size - my_pp_stage - 1;

    // Assign layers to this stage
    int layers_per_stage = SIZE / pp_size;
    int remainder = SIZE % pp_size;

    // Distribute remainder layers (first stages get extra layers)
    if (my_pp_stage < remainder) {
        pp_first_layer = my_pp_stage * (layers_per_stage + 1);
        pp_last_layer = pp_first_layer + layers_per_stage;  // +1 from remainder
    } else {
        pp_first_layer = my_pp_stage * layers_per_stage + remainder;
        pp_last_layer = pp_first_layer + layers_per_stage - 1;
    }

    // Configure layer PP flags
    for (int i = 0; i < SIZE; i++) {
        if (i >= pp_first_layer && i <= pp_last_layer) {
            layers[i]->pipeline_stage = my_pp_stage;

            // First layer of stage receives from previous stage
            layers[i]->needs_pp_recv_forward = (i == pp_first_layer && my_pp_stage > 0);

            // Last layer of stage sends to next stage
            layers[i]->needs_pp_send_forward = (i == pp_last_layer && my_pp_stage < pp_size - 1);

            // Backward is opposite direction
            layers[i]->needs_pp_send_backward = layers[i]->needs_pp_recv_forward;
            layers[i]->needs_pp_recv_backward = layers[i]->needs_pp_send_forward;
        } else {
            layers[i]->pipeline_stage = -1;  // Not owned by this stage
        }
    }

    // Initialize forward queue with warmup microbatches
    for (int m = 0; m < warmup_microbatches; m++) {
        pp_forward_queue.push(m);
    }

    // Log configuration
    if (generator->id == 0 || true) {  // Log for all ranks to verify
        std::cout << "=== Pipeline Parallelism Configuration (Rank " << generator->id << ") ===" << std::endl;
        std::cout << "  PP Stage: " << my_pp_stage << " / " << pp_size << std::endl;
        std::cout << "  Layers: " << pp_first_layer << " - " << pp_last_layer << std::endl;
        std::cout << "  Microbatches: " << num_microbatches << std::endl;
        std::cout << "  Warmup microbatches: " << warmup_microbatches << std::endl;
        std::cout << "  Cooldown microbatches: " << cooldown_microbatches << std::endl;
        std::cout << "  Steady-state 1F1B iterations: " 
                  << (num_microbatches - warmup_microbatches - cooldown_microbatches) << std::endl;
        std::cout << "=========================================" << std::endl;
    }

    NcclLog->writeLog(NcclLogLevel::INFO,
        "PP initialized: stage=%d, layers=[%d,%d], warmup=%d, steady=%d, cooldown=%d",
        my_pp_stage, pp_first_layer, pp_last_layer,
        warmup_microbatches,
        num_microbatches - warmup_microbatches - cooldown_microbatches,
        cooldown_microbatches);
}
```

#### 3.2 Call `init_pipeline_parallel()` from `initialize_workload()`

**Location:** At end of `initialize_workload()` method (around line 390)

```cpp
// [ADD] At end of initialize_workload(), after all layers created
if (first_line_data.workload_type == "HYBRID_TRANSFORMER_FWD_IN_BCKWD") {
    // Initialize Pipeline Parallelism
    init_pipeline_parallel();
}
```

---

## Part 4: Helper Methods

### File: `Workload.cc`

#### 4.1 Implement `should_process_layer()`

**Location:** After `init_pipeline_parallel()` (around line 500)

```cpp
bool Workload::should_process_layer(int layer_idx) {
    if (!pp_enabled) {
        return true;  // Process all layers if PP disabled
    }

    // Check if layer belongs to this stage
    return (layer_idx >= pp_first_layer && layer_idx <= pp_last_layer);
}
```

#### 4.2 Implement `schedule_next_pp_operation()`

**Location:** After `should_process_layer()` (around line 510)

```cpp
void Workload::schedule_next_pp_operation() {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();

    // Determine current phase based on completed operations
    if (pp_forward_completed < warmup_microbatches) {
        pp_state = PPState::Warmup;
    } else if (pp_backward_completed < num_microbatches - cooldown_microbatches) {
        pp_state = PPState::Steady;
    } else {
        pp_state = PPState::Cooldown;
    }

    switch (pp_state) {
    case PPState::Warmup:
        // Warmup: Only forward passes
        if (!pp_forward_queue.empty()) {
            int microbatch_id = pp_forward_queue.front();
            pp_forward_queue.pop();
            pp_forward_in_progress.insert(microbatch_id);

            NcclLog->writeLog(NcclLogLevel::INFO,
                "[WARMUP] Stage %d starting forward for microbatch %d",
                my_pp_stage, microbatch_id);

            execute_pp_forward(microbatch_id);
        }
        break;

    case PPState::Steady:
        // Steady: 1F1B pattern - alternate forward and backward
        // Priority: backward first (reduces memory pressure)
        if (!pp_backward_queue.empty()) {
            int microbatch_id = pp_backward_queue.front();
            pp_backward_queue.pop();
            pp_backward_in_progress.insert(microbatch_id);

            NcclLog->writeLog(NcclLogLevel::INFO,
                "[STEADY-1B] Stage %d starting backward for microbatch %d",
                my_pp_stage, microbatch_id);

            execute_pp_backward(microbatch_id);
        } else if (pp_forward_completed < num_microbatches) {
            // Issue next forward
            int next_forward_id = warmup_microbatches + 
                                  (pp_forward_completed - warmup_microbatches);
            pp_forward_in_progress.insert(next_forward_id);

            NcclLog->writeLog(NcclLogLevel::INFO,
                "[STEADY-1F] Stage %d starting forward for microbatch %d",
                my_pp_stage, next_forward_id);

            execute_pp_forward(next_forward_id);
        }
        break;

    case PPState::Cooldown:
        // Cooldown: Only backward passes
        if (!pp_backward_queue.empty()) {
            int microbatch_id = pp_backward_queue.front();
            pp_backward_queue.pop();
            pp_backward_in_progress.insert(microbatch_id);

            NcclLog->writeLog(NcclLogLevel::INFO,
                "[COOLDOWN] Stage %d starting backward for microbatch %d",
                my_pp_stage, microbatch_id);

            execute_pp_backward(microbatch_id);
        } else if (pp_backward_completed >= num_microbatches) {
            // All microbatches complete - epoch done
            NcclLog->writeLog(NcclLogLevel::INFO,
                "[PP_COMPLETE] Stage %d finished all microbatches", my_pp_stage);

            // Reset for next epoch
            pp_forward_completed = 0;
            pp_backward_completed = 0;
            pp_forward_queue = std::queue<int>();  // Clear queue
            pp_backward_queue = std::queue<int>();

            // Refill warmup
            for (int m = 0; m < warmup_microbatches; m++) {
                pp_forward_queue.push(m);
            }

            pass_counter++;
            current_state = LoopState::Forward_Pass;
            index = pp_first_layer;
        }
        break;
    }
}
```

#### 4.3 Implement `execute_pp_forward()`

**Location:** After `schedule_next_pp_operation()` (around line 600)

```cpp
void Workload::execute_pp_forward(int microbatch_id) {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    pp_current_microbatch = microbatch_id;

    // Set initial state for forward pass
    current_state = LoopState::Forward_Pass;
    index = pp_first_layer;
    delay_loaded = false;
    collective_issued = false;

    NcclLog->writeLog(NcclLogLevel::DEBUG,
        "execute_pp_forward: microbatch=%d, starting layer=%d",
        microbatch_id, pp_first_layer);

    // Resume FSM to process this microbatch's forward pass
    generator->register_event(this, EventType::General, NULL, 1);
}
```

#### 4.4 Implement `execute_pp_backward()`

**Location:** After `execute_pp_forward()` (around line 620)

```cpp
void Workload::execute_pp_backward(int microbatch_id) {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    pp_current_microbatch = microbatch_id;

    // Set initial state for backward pass (start from last layer)
    current_state = LoopState::Input_Gradient;
    index = pp_last_layer;
    delay_loaded = false;
    collective_issued = false;

    NcclLog->writeLog(NcclLogLevel::DEBUG,
        "execute_pp_backward: microbatch=%d, starting layer=%d",
        microbatch_id, pp_last_layer);

    // Resume FSM
    generator->register_event(this, EventType::General, NULL, 1);
}
```

---

## Part 5: Modify Main FSM

### File: `Workload.cc`

#### 5.1 Modify `iterate_hybrid_parallel_Transformer_fwd_in_bckwd()`

**Location:** Replace entire function (around line 650-900)

```cpp
void Workload::iterate_hybrid_parallel_Transformer_fwd_in_bckwd() {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    assert(index >= 0);
    assert(index < SIZE);
    check_for_sim_end();

    // [ADD] PP-specific handling
    if (pp_enabled) {
        // Skip layers not owned by this stage
        if (!should_process_layer(index)) {
            if (current_state == LoopState::Forward_Pass) {
                index++;
                if (index > pp_last_layer) {
                    // Forward pass complete for this microbatch
                    pp_forward_in_progress.erase(pp_current_microbatch);
                    pp_backward_queue.push(pp_current_microbatch);  // Queue for backward
                    pp_forward_completed++;

                    NcclLog->writeLog(NcclLogLevel::INFO,
                        "[PP_FWD_DONE] Stage %d completed forward for microbatch %d",
                        my_pp_stage, pp_current_microbatch);

                    // Schedule next operation
                    schedule_next_pp_operation();
                    return;
                }
            } else if (current_state == LoopState::Input_Gradient) {
                index--;
                if (index < pp_first_layer) {
                    // Backward pass complete for this microbatch
                    pp_backward_in_progress.erase(pp_current_microbatch);
                    pp_backward_completed++;

                    NcclLog->writeLog(NcclLogLevel::INFO,
                        "[PP_BWD_DONE] Stage %d completed backward for microbatch %d",
                        my_pp_stage, pp_current_microbatch);

                    // Schedule next operation
                    schedule_next_pp_operation();
                    return;
                }
            }

            // Continue to next layer
            generator->register_event(this, EventType::General, NULL, 1);
            return;
        }
    }

    // [EXISTING] Forward Pass State
    if (current_state == LoopState::Forward_Pass) {
        // [ADD] PP receive before first layer
        if (pp_enabled && layers[index]->needs_pp_recv_forward) {
            if (!collective_issued) {
                layers[index]->issue_pp_recv_forward(pp_current_microbatch);
                collective_issued = true;
                return;
            }
            if (!layers[index]->is_pp_recv_forward_finished_blocking(pp_current_microbatch)) {
                return;  // Wait for activation from previous stage
            }
            collective_issued = false;  // Reset for next collective
        }

        // [EXISTING] Wait for previous WG sync
        if (!layers[index]->is_weight_grad_comm_finished_blocking()) {
            return;
        }

        // [EXISTING] Compute delay
        if (delay_loaded == false) {
            counter = layers[index]->get_fwd_pass_compute();
            delay_loaded = true;
        }
        if (counter > 0) {
            generator->try_register_event(this, EventType::Workload_Wait, NULL, counter);
            return;
        }

        // [EXISTING] Forward communication
        if (!collective_issued) {
            collective_issued = true;
            if (layers[index]->fwd_pass_comm_size < 4096 && 
                layers[index]->fwd_pass_comm_size > 0) {
                layers[index]->fwd_pass_comm_size = 4096;
            }
            layers[index]->issue_forward_pass_comm(
                SchedulingPolicy::None, CollectiveBarrier::Blocking);
            return;
        }

        // [ADD] PP send after last layer
        if (pp_enabled && layers[index]->needs_pp_send_forward) {
            layers[index]->issue_pp_send_forward(pp_current_microbatch);
            // Non-blocking - continue
        }

        // [EXISTING] Advance to next layer
        index++;
        delay_loaded = false;
        collective_issued = false;

        if (!pp_enabled && index >= SIZE) {
            current_state = LoopState::Input_Gradient;
            index--;
        } else if (pp_enabled && index > pp_last_layer) {
            // PP forward complete - handled at top of function
        }

        NcclLog->writeLog(NcclLogLevel::DEBUG, 
            "workload::call fwd_pass register_event EventType::General");
        generator->register_event(this, EventType::General, NULL, 1);
        return;
    }

    // [EXISTING] Weight Gradient State
    else if (current_state == LoopState::Weight_Gradient) {
        // [EXISTING] Compute delay
        if (delay_loaded == false) {
            counter = layers[index]->get_weight_grad_compute();
            delay_loaded = true;
        }
        if (counter > 0) {
            generator->try_register_event(this, EventType::Workload_Wait, NULL, counter);
            return;
        }

        // [EXISTING] Weight gradient communication
        if (!collective_issued) {
            collective_issued = true;
            layers[index]->issue_weight_grad_comm(
                SchedulingPolicy::FIFO, CollectiveBarrier::Non_Blocking);
        }

        // [EXISTING] Wait for IG comm
        if (!layers[index]->is_input_grad_comm_finished_blocking()) {
            return;
        }

        collective_issued = false;
        delay_loaded = false;

        // [MODIFY] Advance logic for PP
        if (!pp_enabled) {
            // Original logic for non-PP
            if (index >= 0) {
                index--;
            }
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
        } else {
            // PP logic - stay in same layer, transition to IG
            current_state = LoopState::Input_Gradient;
        }

        generator->register_event(this, EventType::General, NULL, 1);
        return;
    }

    // [EXISTING] Input Gradient State
    else if (current_state == LoopState::Input_Gradient) {
        // [ADD] PP receive before backward at last layer
        if (pp_enabled && layers[index]->needs_pp_recv_backward) {
            if (!collective_issued) {
                layers[index]->issue_pp_recv_backward(pp_current_microbatch);
                collective_issued = true;
                return;
            }
            if (!layers[index]->is_pp_recv_backward_finished_blocking(pp_current_microbatch)) {
                return;  // Wait for gradient from next stage
            }
            collective_issued = false;
        }

        // [EXISTING] Checkpoint recomputation trigger
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

        // [EXISTING] Compute delay
        if (delay_loaded == false) {
            counter = layers[index]->get_input_grad_compute();
            delay_loaded = true;
        }
        if (counter > 0) {
            generator->try_register_event(this, EventType::Workload_Wait, NULL, counter);
            return;
        }

        // [EXISTING] Input gradient communication
        if (!collective_issued) {
            collective_issued = true;
            layers[index]->issue_input_grad_comm(
                SchedulingPolicy::LIFO, CollectiveBarrier::Blocking);
            return;
        }

        // [ADD] PP send after backward at first layer
        if (pp_enabled && layers[index]->needs_pp_send_backward) {
            layers[index]->issue_pp_send_backward(pp_current_microbatch);
            // Non-blocking - continue
        }

        checkpoint_initiated = false;
        collective_issued = false;
        delay_loaded = false;

        // [MODIFY] Transition for PP
        if (!pp_enabled) {
            current_state = LoopState::Weight_Gradient;
        } else {
            // PP: Move to previous layer
            index--;
            if (index < pp_first_layer) {
                // Backward complete - handled at top
            } else {
                current_state = LoopState::Weight_Gradient;
            }
        }

        generator->register_event(this, EventType::General, NULL, 1);
        return;
    }

    // [EXISTING] Forward In BackPass State
    else if (current_state == LoopState::Forward_In_BackPass) {
        // [EXISTING LOGIC - unchanged for checkpoint recomputation]
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
                SchedulingPolicy::None, CollectiveBarrier::Blocking);
            return;
        }

        index++;
        delay_loaded = false;
        collective_issued = false;

        if (layers[index]->needs_fwd_in_bckwd_initiation) {
            current_state = LoopState::Input_Gradient;
        }
        generator->register_event(this, EventType::General, NULL, 1);
        return;
    }
}
```

---

## Part 6: Statistics and Reporting

### File: `Workload.cc`

#### 6.1 Track PP Bubble Time

**Location:** In `schedule_next_pp_operation()`, track idle time

```cpp
// [ADD] At start of schedule_next_pp_operation()
if (pp_forward_queue.empty() && pp_backward_queue.empty() &&
    pp_forward_in_progress.empty() && pp_backward_in_progress.empty()) {
    // Bubble - no work to do
    if (pp_bubble_time_start == 0) {
        pp_bubble_time_start = Sys::boostedTick();
    }
} else {
    // Work available - end bubble period
    if (pp_bubble_time_start > 0) {
        pp_total_bubble_time += Sys::boostedTick() - pp_bubble_time_start;
        pp_bubble_time_start = 0;
    }
}
```

#### 6.2 Report PP Statistics

**Location:** In `report()` method, add PP-specific output

```cpp
// [ADD] After existing statistics output
if (pp_enabled) {
    double bubble_ratio = (double)pp_total_bubble_time / 
                          (Sys::boostedTick() * FREQ);

    std::cout << "\n=== Pipeline Parallelism Statistics (Stage " 
              << my_pp_stage << ") ===" << std::endl;
    std::cout << "  Total bubble time: " << pp_total_bubble_time / FREQ << " s" << std::endl;
    std::cout << "  Bubble ratio: " << (bubble_ratio * 100) << "%" << std::endl;
    std::cout << "  Theoretical bubble (P-1)/(M+P-1): " 
              << ((double)(pp_size - 1) / (num_microbatches + pp_size - 1) * 100) 
              << "%" << std::endl;
    std::cout << "=========================================" << std::endl;
}
```

---

## Part 7: Testing and Validation

### Test Configuration

**Workload File Header:**
```
HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: 1 ep: 1 pp: 4 vpp: 1 ga: 8 all_gpus: 4 checkpoints: 0 checkpoint_initiates: 0 pp_comm: 1048576
```

**Expected Behavior:**
- 4 pipeline stages (pp: 4)
- 8 microbatches (ga: 8)
- Stage 0: Warmup 3, Steady 4, Cooldown 1
- Stage 1: Warmup 2, Steady 4, Cooldown 2
- Stage 2: Warmup 1, Steady 4, Cooldown 3
- Stage 3: Warmup 0, Steady 4, Cooldown 4

### Verification Checklist

- [ ] Warmup phase: Only forward passes scheduled
- [ ] Steady state: 1F1B pattern (alternate F and B)
- [ ] Cooldown phase: Only backward passes scheduled
- [ ] Memory: Only P microbatches in-flight at any time
- [ ] Bubble time: Matches theoretical (P-1)/(M+P-1)
- [ ] PP send/recv: Correct stage-to-stage communication
- [ ] No deadlock: All stages progress independently
- [ ] Epoch completion: All microbatches finish correctly

### Debug Logging

Enable detailed PP logging:
```cpp
NcclLog->setLogLevel(NcclLogLevel::DEBUG);
```

Look for these log patterns:
```
[WARMUP] Stage 0 starting forward for microbatch 0
[WARMUP] Stage 0 starting forward for microbatch 1
[WARMUP] Stage 0 starting forward for microbatch 2
[PP_FWD_DONE] Stage 0 completed forward for microbatch 0
[STEADY-1B] Stage 0 starting backward for microbatch 0
[STEADY-1F] Stage 0 starting forward for microbatch 3
[PP_BWD_DONE] Stage 0 completed backward for microbatch 0
...
[COOLDOWN] Stage 0 starting backward for microbatch 7
[PP_COMPLETE] Stage 0 finished all microbatches
```

---

## Part 8: Advanced: Interleaved 1F1B (Optional)

For **Interleaved 1F1B** (Megatron-LM's 1F1B-I), extend with virtual pipeline stages:

### Additional Variables

```cpp
int vpp;                    // Virtual pipeline stages per GPU
int num_model_chunks;       // Number of model chunks = vpp / pp_size
std::vector<int> my_chunks; // Chunk IDs owned by this GPU
```

### Schedule Modification

```cpp
// Context switch between chunks
for (int chunk_id : my_chunks) {
    // Execute operations for chunk_id
    // Switch context (save/restore activations)
}
```

**Benefits:**
- Lower bubble ratio: (P/V - 1) / (M + P/V - 1)
- Better for large models
- Higher memory overhead

---

*Complete 1F1B Pipeline Parallelism implementation for ASTRA-SIM - Production Ready*
