# Layer Class Modifications for Pipeline Parallelism - Implementation Guide

## Overview

This document provides **complete code modifications** to add Pipeline Parallelism (PP) support to the Layer class. All changes include file names, line locations, and exact code to add.

---

## Part 1: Header File Modifications

### File: `Layer.hh`

#### 1.1 Add PP Member Variables (in class private/public section)

**Location:** After existing member variables (around line 60-80)

```cpp
// [ADD] Pipeline Parallelism Configuration
int pipeline_stage;              // Which PP stage owns this layer (-1 if no PP)
bool needs_pp_send_forward;      // Send forward activations to next stage?
bool needs_pp_recv_forward;      // Receive forward activations from prev stage?
bool needs_pp_send_backward;     // Send backward gradients to prev stage?
bool needs_pp_recv_backward;     // Receive backward gradients from next stage?

// [ADD] PP Communication Storage
std::map<int, DataSet*> pp_send_forward_datasets;    // Microbatch ID → DataSet
std::map<int, DataSet*> pp_recv_forward_datasets;
std::map<int, DataSet*> pp_send_backward_datasets;
std::map<int, DataSet*> pp_recv_backward_datasets;

// [ADD] PP Waiting Time Tracking
std::deque<Tick> started_waiting_for_pp_send_fwd;
std::deque<Tick> started_waiting_for_pp_recv_fwd;
std::deque<Tick> started_waiting_for_pp_send_bwd;
std::deque<Tick> started_waiting_for_pp_recv_bwd;

// [ADD] PP Statistics
Tick total_pp_send_fwd_comm;
Tick total_pp_recv_fwd_comm;
Tick total_pp_send_bwd_comm;
Tick total_pp_recv_bwd_comm;
Tick total_waiting_for_pp_send_fwd;
Tick total_waiting_for_pp_recv_fwd;
Tick total_waiting_for_pp_send_bwd;
Tick total_waiting_for_pp_recv_bwd;
```

#### 1.2 Add PP Method Declarations (in public section)

**Location:** After existing communication methods (around line 120-140)

```cpp
// [ADD] Pipeline Parallelism Communication Methods
void issue_pp_send_forward(int microbatch_id);
void issue_pp_recv_forward(int microbatch_id);
void issue_pp_send_backward(int microbatch_id);
void issue_pp_recv_backward(int microbatch_id);

// [ADD] PP Completion Checks
bool is_pp_send_forward_finished(int microbatch_id);
bool is_pp_recv_forward_finished(int microbatch_id);
bool is_pp_send_backward_finished(int microbatch_id);
bool is_pp_recv_backward_finished(int microbatch_id);

// [ADD] PP Blocking Completion Checks
bool is_pp_send_forward_finished_blocking(int microbatch_id);
bool is_pp_recv_forward_finished_blocking(int microbatch_id);
bool is_pp_send_backward_finished_blocking(int microbatch_id);
bool is_pp_recv_backward_finished_blocking(int microbatch_id);
```

---

## Part 2: Constructor Modifications

### File: `Layer.cc`

#### 2.1 Initialize PP Variables in Constructor

**Location:** In `Layer::Layer(...)` constructor, after existing initialization (around line 80)

```cpp
// [ADD] Initialize Pipeline Parallelism Variables
this->pipeline_stage = -1;  // Default: no PP
this->needs_pp_send_forward = false;
this->needs_pp_recv_forward = false;
this->needs_pp_send_backward = false;
this->needs_pp_recv_backward = false;

// [ADD] Initialize PP Statistics
this->total_pp_send_fwd_comm = 0;
this->total_pp_recv_fwd_comm = 0;
this->total_pp_send_bwd_comm = 0;
this->total_pp_recv_bwd_comm = 0;
this->total_waiting_for_pp_send_fwd = 0;
this->total_waiting_for_pp_recv_fwd = 0;
this->total_waiting_for_pp_send_bwd = 0;
this->total_waiting_for_pp_recv_bwd = 0;

// [ADD] Log PP configuration if enabled
if (generator->workload->pipeline_model_parallelism > 1) {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::DEBUG, 
        "Layer %s: PP stage assignment pending (will be set by Workload parser)",
        id.c_str());
}
```

---

## Part 3: Event Callback Modifications

### File: `Layer.cc`

#### 3.1 Add PP Event Cases in `call()` Method

**Location:** In `Layer::call(EventType event, CallData* mdata)` switch statement (around line 150-200)

```cpp
// [ADD] After existing Fwd_Comm_Finished case
else if (event == EventType::PP_Send_Forward_Finished) {
    int microbatch_id = ((IntData*)mdata)->data;
    generator->register_event(
        this,
        EventType::PP_Send_Forward_Finished_After_Delay,
        mdata,
        fwd_update_time);  // Use forward update time
    return;
}
else if (event == EventType::PP_Recv_Forward_Finished) {
    int microbatch_id = ((IntData*)mdata)->data;
    generator->register_event(
        this,
        EventType::PP_Recv_Forward_Finished_After_Delay,
        mdata,
        fwd_update_time);
    return;
}
else if (event == EventType::PP_Send_Backward_Finished) {
    int microbatch_id = ((IntData*)mdata)->data;
    generator->register_event(
        this,
        EventType::PP_Send_Backward_Finished_After_Delay,
        mdata,
        input_grad_update_time);  // Use input grad update time
    return;
}
else if (event == EventType::PP_Recv_Backward_Finished) {
    int microbatch_id = ((IntData*)mdata)->data;
    generator->register_event(
        this,
        EventType::PP_Recv_Backward_Finished_After_Delay,
        mdata,
        input_grad_update_time);
    return;
}
```

#### 3.2 Add PP Delayed Event Handlers

**Location:** After existing Fwd_Comm_Finished_After_Delay case (around line 300)

```cpp
// [ADD] PP Send Forward Delayed Handler
else if (event == EventType::PP_Send_Forward_Finished_After_Delay) {
    int microbatch_id = ((IntData*)mdata)->data;
    IntData* intData = ((IntData*)mdata);

    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::INFO,
        "[PP_FWD_SEND] Layer %s (stage %d): Send forward complete for microbatch %d",
        id.c_str(), pipeline_stage, microbatch_id);

    // Update finish tick
    pp_send_forward_datasets[microbatch_id]->finish_tick += fwd_update_time;

    // Accumulate communication time
    total_pp_send_fwd_comm += pp_send_forward_datasets[microbatch_id]->finish_tick -
                               pp_send_forward_datasets[microbatch_id]->creation_tick;

    // Handle waiting time if Workload was blocked
    if (started_waiting_for_pp_send_fwd.size() > 0) {
        total_waiting_for_pp_send_fwd += pp_send_forward_datasets[microbatch_id]->finish_tick -
                                          started_waiting_for_pp_send_fwd.front();
        started_waiting_for_pp_send_fwd.pop_front();
    }

    // Cleanup
    update_stream_stats(pp_send_forward_datasets[microbatch_id]);
    int dataset_streams = pp_send_forward_datasets[microbatch_id]->total_streams;
    delete pp_send_forward_datasets[microbatch_id];
    pp_send_forward_datasets.erase(microbatch_id);
    generator->increase_finished_streams(dataset_streams);
    delete intData;

    // Resume Workload FSM
    workload->call(EventType::General, NULL);
    return;
}

// [ADD] PP Recv Forward Delayed Handler
else if (event == EventType::PP_Recv_Forward_Finished_After_Delay) {
    int microbatch_id = ((IntData*)mdata)->data;
    IntData* intData = ((IntData*)mdata);

    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::INFO,
        "[PP_FWD_RECV] Layer %s (stage %d): Recv forward complete for microbatch %d",
        id.c_str(), pipeline_stage, microbatch_id);

    pp_recv_forward_datasets[microbatch_id]->finish_tick += fwd_update_time;

    total_pp_recv_fwd_comm += pp_recv_forward_datasets[microbatch_id]->finish_tick -
                               pp_recv_forward_datasets[microbatch_id]->creation_tick;

    if (started_waiting_for_pp_recv_fwd.size() > 0) {
        total_waiting_for_pp_recv_fwd += pp_recv_forward_datasets[microbatch_id]->finish_tick -
                                          started_waiting_for_pp_recv_fwd.front();
        started_waiting_for_pp_recv_fwd.pop_front();
    }

    update_stream_stats(pp_recv_forward_datasets[microbatch_id]);
    int dataset_streams = pp_recv_forward_datasets[microbatch_id]->total_streams;
    delete pp_recv_forward_datasets[microbatch_id];
    pp_recv_forward_datasets.erase(microbatch_id);
    generator->increase_finished_streams(dataset_streams);
    delete intData;

    workload->call(EventType::General, NULL);
    return;
}

// [ADD] PP Send Backward Delayed Handler
else if (event == EventType::PP_Send_Backward_Finished_After_Delay) {
    int microbatch_id = ((IntData*)mdata)->data;
    IntData* intData = ((IntData*)mdata);

    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::INFO,
        "[PP_BWD_SEND] Layer %s (stage %d): Send backward complete for microbatch %d",
        id.c_str(), pipeline_stage, microbatch_id);

    pp_send_backward_datasets[microbatch_id]->finish_tick += input_grad_update_time;

    total_pp_send_bwd_comm += pp_send_backward_datasets[microbatch_id]->finish_tick -
                               pp_send_backward_datasets[microbatch_id]->creation_tick;

    if (started_waiting_for_pp_send_bwd.size() > 0) {
        total_waiting_for_pp_send_bwd += pp_send_backward_datasets[microbatch_id]->finish_tick -
                                          started_waiting_for_pp_send_bwd.front();
        started_waiting_for_pp_send_bwd.pop_front();
    }

    update_stream_stats(pp_send_backward_datasets[microbatch_id]);
    int dataset_streams = pp_send_backward_datasets[microbatch_id]->total_streams;
    delete pp_send_backward_datasets[microbatch_id];
    pp_send_backward_datasets.erase(microbatch_id);
    generator->increase_finished_streams(dataset_streams);
    delete intData;

    workload->call(EventType::General, NULL);
    return;
}

// [ADD] PP Recv Backward Delayed Handler
else if (event == EventType::PP_Recv_Backward_Finished_After_Delay) {
    int microbatch_id = ((IntData*)mdata)->data;
    IntData* intData = ((IntData*)mdata);

    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::INFO,
        "[PP_BWD_RECV] Layer %s (stage %d): Recv backward complete for microbatch %d",
        id.c_str(), pipeline_stage, microbatch_id);

    pp_recv_backward_datasets[microbatch_id]->finish_tick += input_grad_update_time;

    total_pp_recv_bwd_comm += pp_recv_backward_datasets[microbatch_id]->finish_tick -
                               pp_recv_backward_datasets[microbatch_id]->creation_tick;

    if (started_waiting_for_pp_recv_bwd.size() > 0) {
        total_waiting_for_pp_recv_bwd += pp_recv_backward_datasets[microbatch_id]->finish_tick -
                                          started_waiting_for_pp_recv_bwd.front();
        started_waiting_for_pp_recv_bwd.pop_front();
    }

    update_stream_stats(pp_recv_backward_datasets[microbatch_id]);
    int dataset_streams = pp_recv_backward_datasets[microbatch_id]->total_streams;
    delete pp_recv_backward_datasets[microbatch_id];
    pp_recv_backward_datasets.erase(microbatch_id);
    generator->increase_finished_streams(dataset_streams);
    delete intData;

    workload->call(EventType::General, NULL);
    return;
}
```

---

## Part 4: PP Communication Implementation

### File: `Layer.cc`

#### 4.1 Implement PP Send Forward

**Location:** After existing `issue_forward_pass_comm()` method (around line 500)

```cpp
void Layer::issue_pp_send_forward(int microbatch_id) {
    // Check if PP is enabled and this layer needs send
    if (generator->workload->pipeline_model_parallelism <= 1 || !needs_pp_send_forward) {
        return;  // No PP send needed
    }

    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    int my_stage = pipeline_stage;
    int next_stage = my_stage + 1;
    int PP_size = generator->workload->pipeline_model_parallelism;

    // Validate stage
    if (my_stage < 0 || my_stage >= PP_size - 1) {
        NcclLog->writeLog(NcclLogLevel::ERROR,
            "Layer %s: Invalid PP send forward from stage %d (PP_size=%d)",
            id.c_str(), my_stage, PP_size);
        return;
    }

    // Get activation size from workload
    uint64_t activation_size = generator->workload->pp_comm_size;

    NcclLog->writeLog(NcclLogLevel::INFO,
        "[PP_FWD_SEND] Layer %s (stage %d) sending %llu bytes to stage %d, microbatch %d",
        id.c_str(), my_stage, activation_size, next_stage, microbatch_id);

    // Generate PP send collective
    DataSet* pp_ds = generator->generate_collective(
        0,                          // Not used for PP
        ComType::PP_Send,
        activation_size,
        layer_num,
        LoopState::Forward_Pass,
        nullptr,
        true);                      // Issue immediately

    // Check if collective is active
    if (pp_ds == nullptr || !pp_ds->active) {
        NcclLog->writeLog(NcclLogLevel::WARNING,
            "PP send forward collective not active for layer %s", id.c_str());
        if (pp_ds != nullptr) {
            delete pp_ds;
        }
        return;
    }

    // Store DataSet and set callback
    pp_send_forward_datasets[microbatch_id] = pp_ds;
    pp_ds->set_notifier(this, EventType::PP_Send_Forward_Finished, microbatch_id);

    collective_counter++;
}
```

#### 4.2 Implement PP Recv Forward

**Location:** After `issue_pp_send_forward()` (around line 550)

```cpp
void Layer::issue_pp_recv_forward(int microbatch_id) {
    if (generator->workload->pipeline_model_parallelism <= 1 || !needs_pp_recv_forward) {
        return;
    }

    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    int my_stage = pipeline_stage;
    int prev_stage = my_stage - 1;
    int PP_size = generator->workload->pipeline_model_parallelism;

    if (my_stage <= 0 || my_stage >= PP_size) {
        NcclLog->writeLog(NcclLogLevel::ERROR,
            "Layer %s: Invalid PP recv forward at stage %d (PP_size=%d)",
            id.c_str(), my_stage, PP_size);
        return;
    }

    uint64_t activation_size = generator->workload->pp_comm_size;

    NcclLog->writeLog(NcclLogLevel::INFO,
        "[PP_FWD_RECV] Layer %s (stage %d) receiving %llu bytes from stage %d, microbatch %d",
        id.c_str(), my_stage, activation_size, prev_stage, microbatch_id);

    DataSet* pp_ds = generator->generate_collective(
        0,
        ComType::PP_Recv,
        activation_size,
        layer_num,
        LoopState::Forward_Pass,
        nullptr,
        true);

    if (pp_ds == nullptr || !pp_ds->active) {
        NcclLog->writeLog(NcclLogLevel::WARNING,
            "PP recv forward collective not active for layer %s", id.c_str());
        if (pp_ds != nullptr) {
            delete pp_ds;
        }
        return;
    }

    pp_recv_forward_datasets[microbatch_id] = pp_ds;
    pp_ds->set_notifier(this, EventType::PP_Recv_Forward_Finished, microbatch_id);

    collective_counter++;
}
```

#### 4.3 Implement PP Send Backward

**Location:** After `issue_pp_recv_forward()` (around line 600)

```cpp
void Layer::issue_pp_send_backward(int microbatch_id) {
    // Backward send goes to PREVIOUS stage (opposite of forward recv)
    if (generator->workload->pipeline_model_parallelism <= 1 || !needs_pp_recv_forward) {
        return;  // needs_pp_recv_forward because backward flows opposite
    }

    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    int my_stage = pipeline_stage;
    int prev_stage = my_stage - 1;
    int PP_size = generator->workload->pipeline_model_parallelism;

    if (my_stage <= 0 || my_stage >= PP_size) {
        NcclLog->writeLog(NcclLogLevel::ERROR,
            "Layer %s: Invalid PP send backward at stage %d", id.c_str(), my_stage);
        return;
    }

    uint64_t gradient_size = generator->workload->pp_comm_size;

    NcclLog->writeLog(NcclLogLevel::INFO,
        "[PP_BWD_SEND] Layer %s (stage %d) sending gradients %llu bytes to stage %d, microbatch %d",
        id.c_str(), my_stage, gradient_size, prev_stage, microbatch_id);

    DataSet* pp_ds = generator->generate_collective(
        0,
        ComType::PP_Send,
        gradient_size,
        layer_num,
        LoopState::Input_Gradient,
        nullptr,
        true);

    if (pp_ds == nullptr || !pp_ds->active) {
        NcclLog->writeLog(NcclLogLevel::WARNING,
            "PP send backward collective not active for layer %s", id.c_str());
        if (pp_ds != nullptr) {
            delete pp_ds;
        }
        return;
    }

    pp_send_backward_datasets[microbatch_id] = pp_ds;
    pp_ds->set_notifier(this, EventType::PP_Send_Backward_Finished, microbatch_id);

    collective_counter++;
}
```

#### 4.4 Implement PP Recv Backward

**Location:** After `issue_pp_send_backward()` (around line 650)

```cpp
void Layer::issue_pp_recv_backward(int microbatch_id) {
    // Backward recv comes from NEXT stage (opposite of forward send)
    if (generator->workload->pipeline_model_parallelism <= 1 || !needs_pp_send_forward) {
        return;  // needs_pp_send_forward because backward flows opposite
    }

    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    int my_stage = pipeline_stage;
    int next_stage = my_stage + 1;
    int PP_size = generator->workload->pipeline_model_parallelism;

    if (my_stage < 0 || my_stage >= PP_size - 1) {
        NcclLog->writeLog(NcclLogLevel::ERROR,
            "Layer %s: Invalid PP recv backward at stage %d", id.c_str(), my_stage);
        return;
    }

    uint64_t gradient_size = generator->workload->pp_comm_size;

    NcclLog->writeLog(NcclLogLevel::INFO,
        "[PP_BWD_RECV] Layer %s (stage %d) receiving gradients %llu bytes from stage %d, microbatch %d",
        id.c_str(), my_stage, gradient_size, next_stage, microbatch_id);

    DataSet* pp_ds = generator->generate_collective(
        0,
        ComType::PP_Recv,
        gradient_size,
        layer_num,
        LoopState::Input_Gradient,
        nullptr,
        true);

    if (pp_ds == nullptr || !pp_ds->active) {
        NcclLog->writeLog(NcclLogLevel::WARNING,
            "PP recv backward collective not active for layer %s", id.c_str());
        if (pp_ds != nullptr) {
            delete pp_ds;
        }
        return;
    }

    pp_recv_backward_datasets[microbatch_id] = pp_ds;
    pp_ds->set_notifier(this, EventType::PP_Recv_Backward_Finished, microbatch_id);

    collective_counter++;
}
```

---

## Part 5: Completion Check Implementation

### File: `Layer.cc`

#### 5.1 Non-Blocking Completion Checks

**Location:** After existing completion check methods (around line 700)

```cpp
bool Layer::is_pp_send_forward_finished(int microbatch_id) {
    return (pp_send_forward_datasets.count(microbatch_id) == 0);
}

bool Layer::is_pp_recv_forward_finished(int microbatch_id) {
    return (pp_recv_forward_datasets.count(microbatch_id) == 0);
}

bool Layer::is_pp_send_backward_finished(int microbatch_id) {
    return (pp_send_backward_datasets.count(microbatch_id) == 0);
}

bool Layer::is_pp_recv_backward_finished(int microbatch_id) {
    return (pp_recv_backward_datasets.count(microbatch_id) == 0);
}
```

#### 5.2 Blocking Completion Checks

**Location:** After non-blocking checks (around line 720)

```cpp
bool Layer::is_pp_send_forward_finished_blocking(int microbatch_id) {
    if (pp_send_forward_datasets.count(microbatch_id) == 0) {
        return true;  // Already finished
    }

    // First time waiting - record start time
    if (started_waiting_for_pp_send_fwd.size() == 0) {
        started_waiting_for_pp_send_fwd.push_back(Sys::boostedTick());
    }

    return false;
}

bool Layer::is_pp_recv_forward_finished_blocking(int microbatch_id) {
    if (pp_recv_forward_datasets.count(microbatch_id) == 0) {
        return true;
    }

    if (started_waiting_for_pp_recv_fwd.size() == 0) {
        started_waiting_for_pp_recv_fwd.push_back(Sys::boostedTick());
    }

    return false;
}

bool Layer::is_pp_send_backward_finished_blocking(int microbatch_id) {
    if (pp_send_backward_datasets.count(microbatch_id) == 0) {
        return true;
    }

    if (started_waiting_for_pp_send_bwd.size() == 0) {
        started_waiting_for_pp_send_bwd.push_back(Sys::boostedTick());
    }

    return false;
}

bool Layer::is_pp_recv_backward_finished_blocking(int microbatch_id) {
    if (pp_recv_backward_datasets.count(microbatch_id) == 0) {
        return true;
    }

    if (started_waiting_for_pp_recv_bwd.size() == 0) {
        started_waiting_for_pp_recv_bwd.push_back(Sys::boostedTick());
    }

    return false;
}
```

---

## Part 6: Statistics Reporting Updates

### File: `Layer.cc`

#### 6.1 Add PP Statistics to `report()` Method

**Location:** In `Layer::report(...)` method, after existing communication accumulation (around line 1000)

```cpp
// [ADD] Accumulate PP Communication Statistics
if (generator->workload->pipeline_model_parallelism > 1) {
    // PP communication is separate from TP/DP/EP
    double pp_fwd_exposed = (total_waiting_for_pp_send_fwd + 
                             total_waiting_for_pp_recv_fwd) / FREQ;
    double pp_bwd_exposed = (total_waiting_for_pp_send_bwd + 
                             total_waiting_for_pp_recv_bwd) / FREQ;

    // Add to exposed communication (separate from TP/DP)
    total_exposed += pp_fwd_exposed + pp_bwd_exposed;

    // Log PP-specific statistics
    if (seprate_log && generator->id == 0) {
        std::cout << "  PP Send Forward Comm: " 
                  << total_pp_send_fwd_comm / FREQ << " s" << std::endl;
        std::cout << "  PP Recv Forward Comm: " 
                  << total_pp_recv_fwd_comm / FREQ << " s" << std::endl;
        std::cout << "  PP Send Backward Comm: " 
                  << total_pp_send_bwd_comm / FREQ << " s" << std::endl;
        std::cout << "  PP Recv Backward Comm: " 
                  << total_pp_recv_bwd_comm / FREQ << " s" << std::endl;
        std::cout << "  PP Exposed Time: " 
                  << (pp_fwd_exposed + pp_bwd_exposed) << " s" << std::endl;
    }
}

// [ADD] Update LayerData structure with PP stats
layerData.total_pp_send_fwd_comm = total_pp_send_fwd_comm / FREQ;
layerData.total_pp_recv_fwd_comm = total_pp_recv_fwd_comm / FREQ;
layerData.total_pp_send_bwd_comm = total_pp_send_bwd_comm / FREQ;
layerData.total_pp_recv_bwd_comm = total_pp_recv_bwd_comm / FREQ;
layerData.total_waiting_for_pp_send_fwd = total_waiting_for_pp_send_fwd / FREQ;
layerData.total_waiting_for_pp_recv_fwd = total_waiting_for_pp_recv_fwd / FREQ;
layerData.total_waiting_for_pp_send_bwd = total_waiting_for_pp_send_bwd / FREQ;
layerData.total_waiting_for_pp_recv_bwd = total_waiting_for_pp_recv_bwd / FREQ;
```

---

## Part 7: EventType Enum Additions

### File: `Common.hh` (or wherever EventType is defined)

**Location:** In EventType enum (around line 50)

```cpp
enum class EventType {
    // ... existing events ...
    Fwd_Comm_Finished_After_Delay,

    // [ADD] Pipeline Parallelism Events
    PP_Send_Forward_Finished,
    PP_Send_Forward_Finished_After_Delay,
    PP_Recv_Forward_Finished,
    PP_Recv_Forward_Finished_After_Delay,
    PP_Send_Backward_Finished,
    PP_Send_Backward_Finished_After_Delay,
    PP_Recv_Backward_Finished,
    PP_Recv_Backward_Finished_After_Delay,

    // ... rest of events ...
};
```

---

## Part 8: LayerData Structure Extension

### File: `Common.hh` (or wherever LayerData is defined)

**Location:** In LayerData struct (around line 200)

```cpp
struct LayerData {
    // ... existing fields ...

    // [ADD] Pipeline Parallelism Statistics
    double total_pp_send_fwd_comm;
    double total_pp_recv_fwd_comm;
    double total_pp_send_bwd_comm;
    double total_pp_recv_bwd_comm;
    double total_waiting_for_pp_send_fwd;
    double total_waiting_for_pp_recv_fwd;
    double total_waiting_for_pp_send_bwd;
    double total_waiting_for_pp_recv_bwd;
};
```

---

## Part 9: Usage Example in Workload

### File: `Workload.cc`

#### 9.1 Set PP Configuration in `initialize_workload()`

**Location:** After creating Layer object (around line 180)

```cpp
// [ADD] Configure Pipeline Parallelism for this layer
if (pipeline_model_parallelism > 1) {
    // Determine which stage owns this layer
    int layers_per_stage = SIZE / pipeline_model_parallelism;
    int assigned_stage = i / layers_per_stage;

    // Handle remainder layers
    if (assigned_stage >= pipeline_model_parallelism) {
        assigned_stage = pipeline_model_parallelism - 1;
    }

    layers[i]->pipeline_stage = assigned_stage;

    // Determine boundary layers
    bool is_last_layer_of_stage = ((i + 1) / layers_per_stage > assigned_stage);
    bool is_first_layer_of_stage = (i / layers_per_stage == assigned_stage) && 
                                     (i % layers_per_stage == 0);

    // Set PP communication flags
    layers[i]->needs_pp_send_forward = (assigned_stage < pipeline_model_parallelism - 1) && 
                                         is_last_layer_of_stage;
    layers[i]->needs_pp_recv_forward = (assigned_stage > 0) && 
                                         is_first_layer_of_stage;
    layers[i]->needs_pp_send_backward = layers[i]->needs_pp_recv_forward;  // Opposite direction
    layers[i]->needs_pp_recv_backward = layers[i]->needs_pp_send_forward;  // Opposite direction

    if (id == 0) {
        std::cout << "Layer " << i << " (" << layers[i]->id 
                  << ") assigned to stage " << assigned_stage 
                  << " (pp_send_fwd=" << layers[i]->needs_pp_send_forward
                  << ", pp_recv_fwd=" << layers[i]->needs_pp_recv_forward << ")"
                  << std::endl;
    }
}
```

#### 9.2 Use PP in `iterate_hybrid_parallel_Transformer_fwd_in_bckwd()`

**Location:** In Forward_Pass state (around line 350)

```cpp
// [ADD] Before compute delay in Forward_Pass
if (pipeline_model_parallelism > 1) {
    // Check if this stage owns this layer
    int my_stage = generator->id;  // Assuming GPU ID = stage ID for simple mapping

    if (layers[index]->pipeline_stage != my_stage) {
        // Skip layers not owned by this stage
        index++;
        if (index >= SIZE) {
            current_state = LoopState::Input_Gradient;
            index--;
        }
        generator->register_event(this, EventType::General, NULL, 1);
        return;
    }

    // Issue PP recv before processing layer
    if (layers[index]->needs_pp_recv_forward) {
        layers[index]->issue_pp_recv_forward(current_microbatch);
        if (!layers[index]->is_pp_recv_forward_finished_blocking(current_microbatch)) {
            return;  // Wait for activation from previous stage
        }
    }
}

// ... [existing compute and communication logic] ...

// [ADD] After forward pass communication
if (pipeline_model_parallelism > 1 && layers[index]->needs_pp_send_forward) {
    layers[index]->issue_pp_send_forward(current_microbatch);
    // Non-blocking - continue to next layer
}
```

---

## Testing Checklist

After implementing all modifications:

- [ ] Compile successfully (no syntax errors)
- [ ] Test with PP disabled (`pp: 1`) - should behave identically to before
- [ ] Test with PP enabled (`pp: 2`) - verify log messages show PP communication
- [ ] Check memory cleanup - no leaks in PP DataSet maps
- [ ] Verify statistics - PP communication time reported separately
- [ ] Test blocking behavior - Workload waits correctly for PP completion
- [ ] Test microbatch tracking - different microbatches tracked independently
- [ ] Validate stage assignment - layers correctly assigned to stages

---

## Expected Output with PP Enabled

```
Layer 0 (embedding_layer) assigned to stage 0 (pp_send_fwd=0, pp_recv_fwd=0)
...
Layer 8 (attention_layer) assigned to stage 0 (pp_send_fwd=1, pp_recv_fwd=0)
Layer 9 (mlp_layer) assigned to stage 1 (pp_send_fwd=0, pp_recv_fwd=1)
...

[PP_FWD_SEND] Layer attention_layer (stage 0) sending 1048576 bytes to stage 1, microbatch 0
[PP_FWD_RECV] Layer mlp_layer (stage 1) receiving 1048576 bytes from stage 0, microbatch 0
[PP_BWD_SEND] Layer mlp_layer (stage 1) sending gradients 1048576 bytes to stage 0, microbatch 0
[PP_BWD_RECV] Layer attention_layer (stage 0) receiving gradients 1048576 bytes from stage 1, microbatch 0

Layer id: attention_layer
  PP Send Forward Comm: 0.042 s
  PP Recv Backward Comm: 0.043 s
```

---

*Complete modification guide for Pipeline Parallelism support in Layer class - SimAI/ASTRA-SIM*
