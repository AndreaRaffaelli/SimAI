# Layer Class Documentation - SimAI/ASTRA-SIM

## Overview

The **Layer** class represents a single neural network layer in distributed training simulation. It encapsulates compute operations (forward, input gradient, weight gradient) and communication collectives (AllReduce, AllGather, ReduceScatter, AllToAll) for that layer. Each Layer acts as an independent state machine that:

- Tracks computation and communication timing
- Issues collective operations via the NCCL interface
- Manages event-driven callbacks for operation completion
- Collects performance statistics for analysis

**File:** `Layer.cc`  
**Namespace:** `AstraSim`

---

## Constructor

### `Layer::Layer(...)`

```cpp
Layer::Layer(
    std::string id,
    int layer_num,
    Sys* generator,
    Workload* workload,
    Tick fwd_pass_compute_time,
    ComType fwd_pass_comm_type,
    MockNccl::GroupType fwd_pass_group_type,
    uint64_t fwd_pass_comm_size,
    std::vector<bool> fwd_pass_comm_involved_dimensions,
    Tick input_grad_compute_time,
    ComType input_grad_comm_type,
    MockNccl::GroupType input_grad_group_type,
    uint64_t input_grad_comm_size,
    std::vector<bool> input_grad_comm_involved_dimensions,
    Tick weight_grad_compute_time,
    ComType weight_grad_comm_type,
    MockNccl::GroupType weight_grad_group_type,
    uint64_t weight_grad_comm_size,
    std::vector<bool> weight_grad_comm_involved_dimensions,
    Tick weight_grad_update_time,
    ParallelismPolicy specific_policy
)
```

### Parameters

#### Identification
- `id` (string): Human-readable layer name (e.g., "attention_layer", "mlp_0", "embedding")
- `layer_num` (int): Numerical index of layer in workload (0 to SIZE-1)
- `generator` (Sys*): Pointer to owning Sys node (provides collective generation, event scheduling)
- `workload` (Workload*): Pointer to parent Workload (for FSM callbacks)

#### Forward Pass Configuration
- `fwd_pass_compute_time` (Tick): Simulation cycles for forward compute (matrix multiply, attention, etc.)
- `fwd_pass_comm_type` (ComType): Collective type (All_Reduce, All_Gather, Reduce_Scatter, All_to_All, None)
- `fwd_pass_group_type` (MockNccl::GroupType): Communication group (TP, DP, EP, DP_EP, PP)
- `fwd_pass_comm_size` (uint64_t): Message size in bytes
- `fwd_pass_comm_involved_dimensions` (vector<bool>): Which topology dimensions participate (e.g., [true, true, false] = dimensions 0,1 active)

#### Input Gradient Configuration
- `input_grad_compute_time` (Tick): Backprop compute cycles through layer
- `input_grad_comm_type` (ComType): Collective for input gradient synchronization
- `input_grad_group_type` (MockNccl::GroupType): Group type (typically TP for Megatron-style)
- `input_grad_comm_size` (uint64_t): Gradient tensor size in bytes
- `input_grad_comm_involved_dimensions` (vector<bool>): Topology dimensions

#### Weight Gradient Configuration
- `weight_grad_compute_time` (Tick): Gradient computation w.r.t. weights (∂L/∂W)
- `weight_grad_comm_type` (ComType): Collective type (typically All_Reduce for DP sync)
- `weight_grad_group_type` (MockNccl::GroupType): Group type (typically DP or DP_EP)
- `weight_grad_comm_size` (uint64_t): Weight gradient size in bytes
- `weight_grad_comm_involved_dimensions` (vector<bool>): Topology dimensions
- `weight_grad_update_time` (Tick): Optimizer step time (Adam, SGD update)

#### Advanced
- `specific_policy` (ParallelismPolicy): Layer-specific parallelism override (for HybridCustomized)

### Initialization Logic

```cpp
// Copy all parameters to member variables
this->id = id;
this->layer_num = layer_num;
// ... [parameters copied]

// Initialize statistics accumulators
this->collective_counter = 0;
this->total_forward_pass_compute = 0;
this->total_input_grad_compute = 0;
this->total_weight_grad_compute = 0;
this->total_weight_grad_comm = 0;
this->total_input_grad_comm = 0;
this->total_fwd_comm = 0;
this->total_waiting_for_wg_comm = 0;
this->total_waiting_for_ig_comm = 0;
this->total_waiting_for_fwd_comm = 0;

// Initialize timing markers
this->last_fwd_finished = 0;
this->last_ig_finished = 0;
this->last_wg_finished = 0;

// Initialize checkpoint flags
this->needs_fwd_in_bckwd_initiation = false;
this->is_checkpoint = false;

// Validate Sys pointer
assert(generator != NULL);
```

**Key Design:**
- All timing statistics start at zero
- Update times (fwd, ig, wg) all initialized to `weight_grad_update_time` (simulates optimizer delay)
- Checkpoint flags default to false (set by Workload parser if needed)

---

## Event Callback: `call()`

### `void Layer::call(EventType event, CallData* mdata)`

**Purpose:** Event-driven callback invoked when collective operations complete.

### Event Flow

```
Collective Completes (Network)
    ↓
Sys::notifyCallBack()
    ↓
Layer::call(Wight_Grad_Comm_Finished)
    ↓
[Immediate] Schedule delayed callback (update_time cycles)
    ↓
Layer::call(Wight_Grad_Comm_Finished_After_Delay)
    ↓
[After Delay] Update statistics, cleanup, notify Workload
```

### Event Types Handled

#### 1. Initial Completion Events (Immediate Callbacks)

**EventType::Wight_Grad_Comm_Finished**
```cpp
last_wg_finished = Sys::boostedTick();  // Record completion time
generator->register_event(
    this,
    EventType::Wight_Grad_Comm_Finished_After_Delay,
    mdata,
    weight_grad_update_time);  // Schedule after optimizer delay
return;
```
**Purpose:** Weight gradient AllReduce completed, schedule optimizer update delay.

**EventType::Input_Grad_Comm_Finished**
```cpp
last_ig_finished = Sys::boostedTick();
generator->register_event(
    this,
    EventType::Input_Grad_Comm_Finished_After_Delay,
    mdata,
    input_grad_update_time);
return;
```
**Purpose:** Input gradient communication completed, schedule processing delay.

**EventType::Fwd_Comm_Finished**
```cpp
last_fwd_finished = Sys::boostedTick();
generator->register_event(
    this,
    EventType::Fwd_Comm_Finished_After_Delay,
    mdata,
    fwd_update_time);
return;
```
**Purpose:** Forward pass communication completed, schedule activation processing delay.

#### 2. Delayed Completion Events (After Update Time)

**EventType::Wight_Grad_Comm_Finished_After_Delay**

```cpp
int data = ((IntData*)mdata)->data;  // Dataset ID
IntData* intData = ((IntData*)mdata);

#ifndef PHY_MTP
if (generator->id == 0) {
    std::cout << "***** info: weight gradient collective for layer: " << id
              << " is finished************" << std::endl;
}

// Update finish tick to include update time
weight_grad_datasets[data]->finish_tick += weight_grad_update_time;

// Accumulate total communication time
total_weight_grad_comm += weight_grad_datasets[data]->finish_tick -
                          weight_grad_datasets[data]->creation_tick;

// Handle blocking barrier - track waiting time
if (weight_grad_datasets.size() == 1 &&
    wg_barrier == CollectiveBarrier::Blocking) {
    total_waiting_for_wg_comm += weight_grad_datasets[data]->finish_tick -
                                  weight_grad_datasets[data]->creation_tick;
    update_stream_stats(weight_grad_datasets[data]);
    int dataset_streams = weight_grad_datasets[data]->total_streams;
    delete weight_grad_datasets[data];
    weight_grad_datasets.erase(data);
    workload->call(EventType::General, NULL);  // Resume Workload FSM
    generator->increase_finished_streams(dataset_streams);
    delete intData;
    return;
}

// Handle non-blocking - check if Workload was waiting
else if (started_waiting_for_weight_grad.size() > 0) {
    total_waiting_for_wg_comm += weight_grad_datasets[data]->finish_tick -
                                  started_waiting_for_weight_grad.front();
    started_waiting_for_weight_grad.pop_front();
    update_stream_stats(weight_grad_datasets[data]);
    int dataset_streams = weight_grad_datasets[data]->total_streams;
    delete weight_grad_datasets[data];
    weight_grad_datasets.erase(data);
    workload->call(EventType::General, NULL);  // Resume Workload FSM
    generator->increase_finished_streams(dataset_streams);
    delete intData;
    return;
}

// No waiting occurred (non-blocking, Workload didn't check completion)
update_stream_stats(weight_grad_datasets[data]);
int dataset_streams = weight_grad_datasets[data]->total_streams;
delete weight_grad_datasets[data];
weight_grad_datasets.erase(data);
generator->increase_finished_streams(dataset_streams);
delete intData;
#else
workload->call(EventType::General, NULL);
generator->increase_finished_streams(1);
#endif
return;
```

**Key Operations:**
1. **Add update time:** `finish_tick += weight_grad_update_time` (models optimizer step)
2. **Track total comm time:** `total_weight_grad_comm += finish_tick - creation_tick`
3. **Track waiting time:** Only if Workload was blocked waiting
   - **Blocking barrier:** Waiting time = full comm time
   - **Non-blocking + waited:** Waiting time = from wait start to finish
   - **Non-blocking + no wait:** No waiting time (overlapped with compute)
4. **Update stream stats:** Queuing delay, network latency per dimension
5. **Cleanup:** Delete DataSet, erase from map
6. **Resume Workload:** Call `workload->call()` to continue FSM
7. **Update global counters:** `increase_finished_streams()` for termination check

**EventType::Input_Grad_Comm_Finished_After_Delay**  
**EventType::Fwd_Comm_Finished_After_Delay**

Same logic as Weight_Grad, but operates on:
- `input_grad_datasets` and `ig_barrier`
- `fwd_pass_datasets` and `fwd_barrier`

---

## Compute Time Getters

### `Tick Layer::get_fwd_pass_compute()`
```cpp
total_forward_pass_compute += fwd_pass_compute_time;
return fwd_pass_compute_time;
```
**Purpose:** Returns FP compute cycles, accumulates to total for statistics.  
**Called by:** Workload FSM when loading compute delay for forward pass.

### `Tick Layer::get_input_grad_compute()`
```cpp
total_input_grad_compute += input_grad_compute_time;
return input_grad_compute_time;
```
**Purpose:** Returns IG compute cycles, accumulates to total.  
**Called by:** Workload FSM during backward pass.

### `Tick Layer::get_weight_grad_compute()`
```cpp
total_weight_grad_compute += weight_grad_compute_time;
return weight_grad_compute_time;
```
**Purpose:** Returns WG compute cycles, accumulates to total.  
**Called by:** Workload FSM during weight gradient computation.

---

## Communication Completion Checks

### Non-Blocking Completion Checks

**`bool Layer::is_fwd_pass_comm_finished()`**
```cpp
if (fwd_pass_datasets.size() == 0) {
    return true;
}
return false;
```
**Semantics:** Simple check - returns true if no pending FP collectives.  
**No side effects** - doesn't track waiting time.  
**Use case:** Non-blocking queries where Workload doesn't wait.

**`bool Layer::is_input_grad_comm_finished()`**  
**`bool Layer::is_weight_grad_comm_finished()`**  
Same logic for IG and WG collectives.

### Blocking Completion Checks

**`bool Layer::is_fwd_pass_comm_finished_blocking()`**
```cpp
if (fwd_pass_datasets.size() == 0) {
    return true;  // Already finished
}

// First time waiting - record start time
if (started_waiting_for_fwd_pass.size() == 0) {
    started_waiting_for_fwd_pass.push_back(Sys::boostedTick());
}

return false;  // Still waiting
```
**Semantics:** Checks completion AND tracks when waiting started.  
**First call:** Records `Sys::boostedTick()` as waiting start time.  
**Subsequent calls:** Returns false (still waiting).  
**When collective finishes:** `call()` computes waiting time = finish_tick - started_waiting[0].

**Example Timeline:**
```
T=1000: Workload issues FP comm
T=1050: Workload checks is_fwd_pass_comm_finished_blocking()
        → Returns false, records started_waiting[1050]
T=1080: Workload checks again
        → Returns false (already recorded)
T=1100: Comm finishes, Layer::call() invoked
        → Waiting time = 1100 - 1050 = 50 cycles
```

**`bool Layer::is_input_grad_comm_finished_blocking()`**  
**`bool Layer::is_weight_grad_comm_finished_blocking()`**  
Same logic with respective data structures:
- `started_waiting_for_input_grad`
- `started_waiting_for_weight_grad`

---

## Communication Issuance

### `void Layer::issue_forward_pass_comm(SchedulingPolicy pref_scheduling, CollectiveBarrier barrier)`

**Purpose:** Initiate forward pass communication collective.

### Parameters
- `pref_scheduling` (SchedulingPolicy): Scheduling hint (None, FIFO, LIFO, HIGHEST)
- `barrier` (CollectiveBarrier): Blocking or Non_Blocking

### Execution Flow

#### 1. Analytical Mode Short-Circuit
```cpp
#ifdef ANALYTI
fwd_barrier = barrier;
if (generator->id == 0) {
    NcclLog->writeLog(NcclLogLevel::DEBUG, "forward pass for layer %s is analytical", id.c_str());
}
if (barrier == CollectiveBarrier::Blocking) {
    workload->call(EventType::General, NULL);  // Immediate callback for analytical
}
return;
#endif
```
**Analytical mode:** Uses mathematical models instead of network simulation.  
**Blocking:** Immediately resume Workload (no actual comm delay).

#### 2. Initialize Collective
```cpp
DataSet* fp = NULL;
fwd_barrier = barrier;
collective_counter++;  // Track total collectives issued
```

#### 3. Dispatch by Communication Type

**All_Reduce:**
```cpp
if (fwd_pass_comm_type == ComType::All_Reduce) {
#ifdef PHY_MTP
    fp = generator->generate_all_reduce(
        fwd_pass_comm_size,
        fwd_pass_comm_involved_dimensions,
        pref_scheduling,
        layer_num,
        EventType::Fwd_Comm_Finished,
        this);  // Callback target
#else
    fp = generator->generate_all_reduce(
        fwd_pass_comm_size,
        fwd_pass_comm_involved_dimensions,
        pref_scheduling,
        layer_num);
#endif

    // Check if collective is active (at least one dimension enabled)
    if (!fp->active) {
        if (generator->id == 0) {
            std::cout << "info: all dims disabled, no forward pass collective for layer: "
                      << id << std::endl;
        }
        collective_counter--;
        delete fp;
        if (barrier == CollectiveBarrier::Blocking) {
            workload->call(EventType::General, NULL);  // Resume immediately
        }
        return;
    }

    if (generator->id == 0) {
        std::cout << "info: all-reduce forward pass collective issued for layer: "
                  << id << ",";
        print_involved_dimensions(fwd_pass_comm_involved_dimensions);
    }
}
```

**Similar dispatching for:**
- `ComType::All_to_All` → `generator->generate_all_to_all(...)`
- `ComType::All_Gather` → `generator->generate_all_gather(...)`
- `ComType::Reduce_Scatter` → `generator->generate_reduce_scatter(...)`
- `ComType::None` → No collective, immediate callback if blocking

**Unknown type:**
```cpp
else {
    Sys::sys_panic("no known collective operation!");
}
```

#### 4. Register Callback (Non-PHY_MTP)
```cpp
#ifndef PHY_MTP
fwd_pass_datasets[fp->my_id] = fp;  // Store DataSet by ID
fp->set_notifier(this, EventType::Fwd_Comm_Finished);  // Set completion callback
#endif
```
**DataSet stores:** Pointer to Layer and EventType to call when done.  
**Asynchronous:** Collective executes in background, Layer::call() invoked on completion.

### `void Layer::issue_input_grad_comm(SchedulingPolicy pref_scheduling, CollectiveBarrier barrier)`

**Nearly identical to `issue_forward_pass_comm()`**, but:
- Uses `input_grad_comm_type`, `input_grad_comm_size`, `input_grad_comm_involved_dimensions`
- Stores in `input_grad_datasets`
- Callback event: `EventType::Input_Grad_Comm_Finished`

### `void Layer::issue_weight_grad_comm(SchedulingPolicy pref_scheduling, CollectiveBarrier barrier)`

**Nearly identical to `issue_forward_pass_comm()`**, but:
- Uses `weight_grad_comm_type`, `weight_grad_comm_size`, `weight_grad_comm_involved_dimensions`
- Stores in `weight_grad_datasets`
- Callback event: `EventType::Wight_Grad_Comm_Finished`
- **Additional logging:** WG comm size printed (important for DP)

---

## Statistics and Reporting

### `LayerData Layer::report(...)`

**Purpose:** Aggregate and output layer performance statistics after training completion.

### Parameters
```cpp
LayerData Layer::report(
    std::string run_name,          // Configuration name
    int layer_num,                 // Layer index
    int total_rows,                // Total stat rows (for CSV)
    int stat_row,                  // Current stat row
    CSVWriter* detailed,           // Detailed stats CSV
    CSVWriter* EndToEnd,           // Summary stats CSV
    double& total_compute,         // [out] Accumulate compute time
    double& total_exposed,         // [out] Accumulate exposed comm
    bool seprate_log,              // Enable separate logging
    vector<double>& total_fwd_time,    // [out] [compute, exposed, total] for FP
    vector<double>& total_wg_time,     // [out] [compute, exposed, total] for WG
    vector<double>& total_ig_time,     // [out] [compute, exposed, total] for IG
    double& pre_bubble_time,       // [out] Pipeline bubble time
    double& DP_comm,               // [out] Data parallel communication
    double& DP_EP_comm,            // [out] DP within EP communication
    double& Expose_TP_comm,        // [out] Exposed tensor parallel comm
    double& Expose_EP_comm         // [out] Exposed expert parallel comm
)
```

### Execution Phases

#### Phase 1: Calculate Parallelism Configuration
```cpp
take_stream_stats_average();  // Average queuing delay, network latency

int TP_size = workload->model_parallel_npu_group;
int PP_size = workload->pipeline_model_parallelism;
int DP_size = workload->all_gpus / (TP_size * PP_size);
int EP_size = workload->expert_parallel_npu_group;
int vpp = workload->vpp;  // Virtual pipeline stages
uint32_t pp_commsize = workload->pp_commsize;
int GA = workload->GA;  // Gradient accumulation steps

// Determine group sizes for bandwidth calculation
int input_grad_group_size = (input_grad_group_type == MockNccl::GroupType::EP) ? EP_size : TP_size;
int fwd_pass_group_size = (fwd_pass_group_type == MockNccl::GroupType::EP) ? EP_size : TP_size;
int weight_grad_group_size = (weight_grad_group_type == MockNccl::GroupType::DP_EP) 
                               ? DP_size / EP_size 
                               : DP_size;
```

#### Phase 2: Accumulate Communication by Type
```cpp
// Skip embedding layer for bubble time calculation
if (id != "embedding_layer") {
    pre_bubble_time += ((total_waiting_for_fwd_comm + 
                         total_forward_pass_compute + 
                         total_weight_grad_compute + 
                         total_input_grad_compute + 
                         total_waiting_for_ig_comm) / FREQ);
}

// Classify weight gradient communication
if (weight_grad_group_type == MockNccl::GroupType::DP_EP) {
    DP_EP_comm += (total_waiting_for_wg_comm / FREQ);
} else {
    DP_comm += (total_waiting_for_wg_comm / FREQ);
}

// Classify forward/input grad communication
if (fwd_pass_group_type == MockNccl::GroupType::EP) {
    Expose_EP_comm += ((total_waiting_for_fwd_comm + total_waiting_for_ig_comm) / FREQ);
} else {
    Expose_TP_comm += ((total_waiting_for_fwd_comm + total_waiting_for_ig_comm) / FREQ);
}
```

**Classification Logic:**
- **DP_comm:** Weight gradient AllReduce across pure DP replicas
- **DP_EP_comm:** Weight gradient AllReduce within expert parallelism groups
- **Expose_TP_comm:** Forward/backward collectives in tensor parallelism (typically blocking)
- **Expose_EP_comm:** Forward/backward collectives in expert parallelism

#### Phase 3: Accumulate Total Times
```cpp
// Accumulate to workload-level totals
total_compute += (total_forward_pass_compute / FREQ);
total_compute += (total_weight_grad_compute / FREQ);
total_compute += (total_input_grad_compute / FREQ);

total_exposed += (total_waiting_for_fwd_comm / FREQ);
total_exposed += (total_waiting_for_wg_comm / FREQ);
total_exposed += (total_waiting_for_ig_comm / FREQ);

// Populate per-phase vectors
total_fwd_time[0] += total_forward_pass_compute / FREQ;  // Compute
total_fwd_time[1] += total_waiting_for_fwd_comm / FREQ;  // Exposed comm
total_fwd_time[2] += total_fwd_comm / FREQ;              // Total comm

total_wg_time[0] += total_weight_grad_compute / FREQ;
total_wg_time[1] += total_waiting_for_wg_comm / FREQ;
total_wg_time[2] += total_weight_grad_comm / FREQ;

total_ig_time[0] += total_input_grad_compute / FREQ;
total_ig_time[1] += total_waiting_for_ig_comm / FREQ;
total_ig_time[2] += total_input_grad_comm / FREQ;
```

#### Phase 4: Populate LayerData Structure
```cpp
LayerData layerData;
layerData.layer_name = id;
layerData.total_forward_pass_compute = total_forward_pass_compute / FREQ;
layerData.total_weight_grad_compute = total_weight_grad_compute / FREQ;
layerData.total_input_grad_compute = total_input_grad_compute / FREQ;
layerData.total_waiting_for_fwd_comm = total_waiting_for_fwd_comm / FREQ;
layerData.total_waiting_for_wg_comm = total_waiting_for_wg_comm / FREQ;
layerData.total_waiting_for_ig_comm = total_waiting_for_ig_comm / FREQ;
layerData.total_fwd_comm = total_fwd_comm / FREQ;
layerData.total_weight_grad_comm = total_weight_grad_comm / FREQ;
layerData.total_input_grad_comm = total_input_grad_comm / FREQ;

// Add queuing delay per dimension
int i = 0;
for (auto& qd : queuing_delay) {
    layerData.avg_queuing_delay.push_back(std::make_pair(i, qd / FREQ));
    i++;
}

// Add network message latency per dimension
i = 1;
for (auto& ml : net_message_latency) {
    layerData.avg_network_message_dealy.push_back(std::make_pair(i, ml / FREQ));
    i++;
}
```

#### Phase 5: Console and CSV Output
```cpp
if (seprate_log) {
    std::cout << "*******************" << std::endl;
    std::cout << "Layer id: " << id << std::endl;
    std::cout << "Total collectives issued for this layer: " << collective_counter << std::endl;
    std::cout << "************************* Workload stats ************************* " << id << std::endl;

    // Write CSV header (first layer only)
    if (stat_row == 0 && layer_num == 0) {
        data = "layer_name," + run_name + 
               ",fwd compute,wg compute,ig compute,"
               "fwd exposed comm,wg exposed comm,ig exposed comm,"
               "fwd total comm,algbw,busbw,"
               "wg total comm,algbw,busbw,"
               "ig total comm,algbw,busbw,"
               "workload finished at";
        EndToEnd->write_line(data);
    }

    // Calculate bandwidth (Algorithm BW and Bus BW)
    total_bw = compute_busbw(fwd_pass_comm_type, fwd_pass_group_size, 
                             fwd_pass_comm_size, total_fwd_comm);

    // Format and write per-layer data
    data = id + "," + run_name + "," +
           std::to_string(total_forward_pass_compute / FREQ) + "," +
           std::to_string(total_weight_grad_compute / FREQ) + "," +
           // ... [all fields]
           std::to_string(((double)Sys::boostedTick()) / FREQ);
    EndToEnd->write_line(data);
}
```

#### Phase 6: Final Summary (Last Layer Only)
```cpp
if (layer_num == workload->SIZE - 1) {
    // Calculate pipeline overhead
    Tick Expose_PP_time = (2 * vpp * GA * (pp_commsize * GBps / overlap_ratio * 1e9) / FREQ);
    Expose_PP_time *= (1 - overlap_ratio);

    // Calculate bubble time (GPipe-style)
    pre_bubble_time *= static_cast<double>(PP_size - 1) / (GA * vpp);

    // Total training time
    double total_time = total_compute + total_exposed + pre_bubble_time + Expose_PP_time;

    // Write breakdown with percentages
    auto format_percentage = [&](double value) {
        double percentage = (value / total_time) * 100;
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(2) << percentage;
        return stream.str() + "%";
    };

    std::string keys = "File name, Expose DP comm, Expose DP_EP comm, Expose TP comm, "
                       "Expose_EP_comm, Expose_PP_comm, bubble time, total comp, "
                       "total exposed comm, Total time";
    std::string values = file_name + ", " +
                         format_value(DP_comm) + " (" + format_percentage(DP_comm) + "), " +
                         // ... [all communication types with percentages]
                         format_value(total_time);

    data = keys + "\n" + values;
    EndToEnd->write_res(data);
}

return layerData;
```

### Bandwidth Calculation

**`std::pair<float, float> Layer::compute_busbw(ComType comtype, int nranks, uint64_t data_size, Tick total_comm)`**

```cpp
float algbw = data_size / (total_comm / FREQ) * 1000000 * GBps;  // Algorithm bandwidth
float busbw = 0.0;

if (comtype == ComType::All_Reduce) {
    busbw = algbw * 2 * (nranks - 1) / nranks;  // Ring AllReduce factor
} else if (comtype == ComType::All_Gather || 
           comtype == ComType::Reduce_Scatter || 
           comtype == ComType::All_to_All) {
    busbw = algbw * (nranks - 1) / nranks;  // Single-direction factor
} else {
    busbw = 0.0;
}

return std::make_pair(algbw, busbw);
```

**Algorithm Bandwidth:** Raw data rate = bytes / time  
**Bus Bandwidth:** Effective bandwidth accounting for collective algorithm overhead

**Example:**
- AllReduce 1GB across 8 GPUs in 100ms
- Algorithm BW = 1GB / 0.1s = 10 GB/s
- Bus BW = 10 * 2 * 7 / 8 = 17.5 GB/s (accounts for 2*(N-1) data movements in ring)

---

## Analytical Mode Time Estimation

### `Tick Layer::compute_time(...)`

**Purpose:** Estimate communication time using analytical models (no network simulation).

```cpp
Tick Layer::compute_time(
    ComType comtype,
    int tp_size,
    int nranks,
    uint64_t data_size,
    MockNccl::GroupType group_type,
    int all_gpus,
    int ep_size)
```

### Implementation Strategy

#### 1. Handle None Case
```cpp
if (comtype == ComType::None) {
    return 0;
}
```

#### 2. Small Message Overhead
```cpp
// Empirical overheads for small messages (< 1MB)
if (1 < data_size && data_size < 1048576) {
    if (nranks == 2) comp_time = 10000;   // ~10 µs
    if (nranks == 4) comp_time = 12000;
    if (nranks == 8) comp_time = 15000;
    if (nranks == 16) comp_time = 66000;
    if (nranks == 32) comp_time = 135000;
    if (nranks == 64) comp_time = 200000;
    if (nranks == 128) comp_time = 320000;
    return comp_time;
}
```
**Rationale:** Small messages dominated by protocol overhead, not bandwidth.

#### 3. Calculate Bus Bandwidth
```cpp
BusBwResult result;

if (group_type == MockNccl::GroupType::TP) {
    // TP communication (intra-node or inter-node)
    if (tp_size <= gpus_per_server) {
        result = cal_busbw(gpu_type, nvlink_bw, bw_per_nic, nics_per_server, 
                           1, coll_type, tp_size, nic_type);
    } else {
        int _node_count = tp_size / gpus_per_server;
        result = cal_busbw(gpu_type, nvlink_bw, bw_per_nic, nics_per_server, 
                           _node_count, coll_type, gpus_per_server, nic_type);
    }
} else if (group_type == MockNccl::GroupType::DP && nranks > 1) {
    // DP communication (inter-node)
    uint32_t _temp_gpus_per_server = gpus_per_server / tp_size;
    float _temp_nics_per_server = nics_per_server / tp_size;
    result = cal_busbw(gpu_type, nvlink_bw, bw_per_nic, _temp_nics_per_server, 
                       nranks, coll_type, _temp_gpus_per_server, nic_type);
}
// ... [similar for EP, DP_EP]
```

#### 4. Apply Empirical Correction
```cpp
bw_ratio = cal_ratio(data_size, nranks, tp_size, gpus_per_server, 
                     group_type, coll_type, result.is_nvlink);
```
**Purpose:** Adjust ideal bandwidth for real-world inefficiencies (contention, protocol overhead).

#### 5. Calculate Final Time
```cpp
comp_time = (data_size * GBps / (result.busbw * bw_ratio) * 1e9) / FREQ;
return comp_time;
```
**Formula:** time = data_size / (bandwidth × correction_factor)

---

## Utility Functions

### `void Layer::print_involved_dimensions(std::vector<bool>& involved_dimensions)`

```cpp
std::cout << " involved dimensions: ";
for (int i = 0; i < involved_dimensions.size(); i++) {
    if (involved_dimensions[i] == true) {
        std::cout << " 1,";
    } else {
        std::cout << " 0,";
    }
}
std::cout << std::endl;
```
**Output Example:** ` involved dimensions:  1, 1, 0, 0, 0, 0, 0, 0, 0, 0,`  
**Interpretation:** Dimensions 0 and 1 active (e.g., TP communication on 2D torus first two dims).

### `void Layer::increment_waiting_for_wg()`
```cpp
total_waiting_for_wg_comm++;
```
**Purpose:** Manual increment (rarely used, mostly automatic via blocking checks).

### `void Layer::increment_waiting_for_ig()`
### `void Layer::increment_waiting_for_fwd()`
Similar single-cycle increment functions.

---

## Key Design Patterns

### 1. Event-Driven Callback Architecture

```
Layer::issue_forward_pass_comm()
    ↓
Sys::generate_all_reduce()
    ↓
MockNccl::getFlowModels()
    ↓
NS-3 network simulation
    ↓ (time passes)
DataSet completion detected
    ↓
Layer::call(Fwd_Comm_Finished)
    ↓
Schedule delayed callback (update_time)
    ↓
Layer::call(Fwd_Comm_Finished_After_Delay)
    ↓
workload->call(EventType::General)
```

**Benefits:**
- **Concurrency:** Layer doesn't block during communication
- **Accuracy:** Models realistic asynchronous GPU behavior
- **Scalability:** Simulator handles thousands of concurrent events

### 2. Blocking vs Non-Blocking Semantics

| Barrier Type | Workload Behavior | Waiting Time Tracked | Use Case |
|--------------|-------------------|---------------------|----------|
| **Blocking** | Calls `is_*_finished_blocking()` repeatedly | Yes, from first check | TP collectives with data dependencies |
| **Non-Blocking** | May not check, or checks non-blocking variant | Only if checked while pending | DP AllReduce overlapped with compute |

**Example Blocking:**
```cpp
// Workload FSM
if (!layers[i]->is_fwd_pass_comm_finished_blocking()) {
    return;  // Pause FSM, wait for completion
}
// Comm finished, continue
```

**Example Non-Blocking:**
```cpp
// Workload issues WG comm
layers[i]->issue_weight_grad_comm(FIFO, Non_Blocking);
// Continues immediately to next layer's WG compute (overlap)
```

### 3. Two-Stage Completion

**Stage 1:** Network completes data transfer → `Comm_Finished` event  
**Stage 2:** After update_time delay → `Comm_Finished_After_Delay` event

**Why two stages?**
- Models **processing delay** after data arrives (e.g., reduction operation, optimizer update)
- Separates **network time** from **GPU compute time**
- More accurate than single-stage model

### 4. Deferred Waiting Time Calculation

**Problem:** Don't know if Workload will wait for collective when issuing it.

**Solution:**
- `issue_*_comm()`: Creates DataSet, but doesn't assume waiting
- `is_*_finished_blocking()`: Records waiting start time on first check
- `call(Comm_Finished_After_Delay)`: Calculates waiting time = finish - start

**Result:** Only counts waiting time if Workload actually blocked.

---

## Integration Points

### With Workload
```cpp
// Workload → Layer
Tick compute = layer->get_fwd_pass_compute();
layer->issue_forward_pass_comm(SchedulingPolicy::None, Blocking);
bool done = layer->is_fwd_pass_comm_finished_blocking();

// Layer → Workload
workload->call(EventType::General, NULL);  // Resume FSM
```

### With Sys
```cpp
// Layer → Sys
DataSet* ds = generator->generate_all_reduce(...);
generator->register_event(this, EventType, data, delay);
generator->increase_finished_streams(count);

// Sys → Layer
layer->call(EventType::Fwd_Comm_Finished, mdata);
```

### With DataSet
```cpp
// Layer → DataSet
fwd_pass_datasets[ds->my_id] = ds;
ds->set_notifier(this, EventType::Fwd_Comm_Finished);

// DataSet → Layer (via Sys callback)
// When ds completes, invokes layer->call()
```

### With MockNccl
```cpp
// Sys → MockNccl (via generator)
ncclInfo* info = generator->get_nccl_Info(group_type, comm_size, comm_type);
// Returns flow models for collective
```

---

## Statistics Collected

### Compute Time
- `total_forward_pass_compute`: Total FP compute cycles
- `total_input_grad_compute`: Total IG compute cycles
- `total_weight_grad_compute`: Total WG compute cycles

### Communication Time
- `total_fwd_comm`: Total FP communication cycles (including overlapped)
- `total_input_grad_comm`: Total IG communication cycles
- `total_weight_grad_comm`: Total WG communication cycles

### Exposed (Waiting) Time
- `total_waiting_for_fwd_comm`: Cycles Workload blocked waiting for FP comm
- `total_waiting_for_ig_comm`: Cycles blocked waiting for IG comm
- `total_waiting_for_wg_comm`: Cycles blocked waiting for WG comm

### Network Statistics
- `queuing_delay`: Per-dimension queuing delay in network
- `net_message_latency`: Per-dimension message propagation latency

### Miscellaneous
- `collective_counter`: Total collectives issued
- `last_fwd_finished`, `last_ig_finished`, `last_wg_finished`: Completion timestamps

---

## Usage Example

### Workload File Entry
```
attention_layer  -1  1820000  ALLGATHER  2097152  1820000  REDUCESCATTER  2097152  1820000  ALLREDUCE  8388608  100
```

### Layer Creation
```cpp
Layer* attn = new Layer(
    "attention_layer",          // id
    6,                           // layer_num
    sys_ptr,                     // generator
    workload_ptr,                // workload
    1820000,                     // fwd_pass_compute_time
    ComType::All_Gather,         // fwd_pass_comm_type
    MockNccl::GroupType::TP,     // fwd_pass_group_type
    2097152,                     // fwd_pass_comm_size (2 MB)
    {true, true, false, ...},    // fwd_pass_comm_involved_dimensions
    1820000,                     // input_grad_compute_time
    ComType::Reduce_Scatter,     // input_grad_comm_type
    MockNccl::GroupType::TP,     // input_grad_group_type
    2097152,                     // input_grad_comm_size
    {true, true, false, ...},    // input_grad_comm_involved_dimensions
    1820000,                     // weight_grad_compute_time
    ComType::All_Reduce,         // weight_grad_comm_type
    MockNccl::GroupType::DP,     // weight_grad_group_type
    8388608,                     // weight_grad_comm_size (8 MB)
    {false, false, true, ...},   // weight_grad_comm_involved_dimensions
    100,                         // weight_grad_update_time
    ParallelismPolicy::Transformer
);
```

### Workload FSM Usage
```cpp
// Forward Pass
Tick compute = attn->get_fwd_pass_compute();  // Returns 1820000
generator->register_event(this, Workload_Wait, NULL, compute);
// ... (after delay) ...
attn->issue_forward_pass_comm(None, Blocking);  // Issues AllGather
// ... (returns, waits for completion) ...
if (!attn->is_fwd_pass_comm_finished_blocking()) {
    return;  // Still waiting
}
// Collective finished, proceed

// Input Gradient
compute = attn->get_input_grad_compute();
// ... (compute delay) ...
attn->issue_input_grad_comm(LIFO, Blocking);  // Issues ReduceScatter
// ... (wait) ...

// Weight Gradient
compute = attn->get_weight_grad_compute();
// ... (compute delay) ...
attn->issue_weight_grad_comm(FIFO, Non_Blocking);  // Issues AllReduce (DP)
// Continues immediately (non-blocking)
```

### Statistics Output
```
Layer id: attention_layer
Total collectives issued for this layer: 3
id: attention_layer ,Total cycles spent on fwd pass compute: 1820000
id: attention_layer ,Total cycles spent idle waiting for fwd finish: 42150
id: attention_layer ,Total cycles spent on fwd pass comm: 45200
id: attention_layer ,Total cycles spent on weight grad compute: 1820000
id: attention_layer ,Total cycles spent idle waiting for weight grad finish: 128450
id: attention_layer ,Total cycles spent on weight grad comm: 131500
```

---

## Common Pitfalls

### 1. Forgetting to Check Completion
**Symptom:** Workload FSM advances before collective finishes, uses stale data  
**Solution:** Always call `is_*_comm_finished_blocking()` before proceeding

### 2. Mixing Blocking and Non-Blocking
**Symptom:** Incorrect waiting time statistics  
**Solution:** Match barrier type to Workload checking behavior:
- If Workload waits → `Blocking`
- If Workload continues → `Non_Blocking`

### 3. DataSet Memory Leak
**Symptom:** Memory grows unbounded  
**Cause:** Forgetting to delete DataSet in completion callback  
**Solution:** Always `delete datasets[data]` and `erase(data)` in `call()`

### 4. Null Generator Pointer
**Symptom:** Segfault when issuing collectives  
**Cause:** Constructor `assert(generator != NULL)` failed  
**Solution:** Ensure Sys* passed to constructor is valid

---

## Future Extensions for Pipeline Parallelism

### Required Additions

1. **Pipeline Stage Assignment**
```cpp
int pipeline_stage;  // Which PP stage owns this layer (-1 if no PP)
bool needs_pp_send;  // Send activations to next stage?
bool needs_pp_recv;  // Receive activations from prev stage?
```

2. **PP Communication Methods**
```cpp
void issue_pp_send_forward(int microbatch_id);
void issue_pp_recv_forward(int microbatch_id);
void issue_pp_send_backward(int microbatch_id);
void issue_pp_recv_backward(int microbatch_id);
```

3. **PP Completion Checks**
```cpp
bool is_pp_send_finished();
bool is_pp_recv_finished();
```

4. **PP-Specific Communication Type**
```cpp
case ComType::PP_Send:
    fp = generator->generate_pp_send(peer_rank, activation_size, layer_num);
    break;
case ComType::PP_Recv:
    fp = generator->generate_pp_recv(peer_rank, activation_size, layer_num);
    break;
```

---

*Documentation generated for SimAI/ASTRA-SIM Layer class - Version: Comprehensive Implementation Guide*
