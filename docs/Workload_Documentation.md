# Documentazione Classe Workload - SimAI/ASTRA-SIM

## Panoramica

La classe `Workload` gestisce l'esecuzione del carico di lavoro di training AI nel simulatore, orchestrando l'iterazione attraverso i layer del modello, l'emissione di operazioni collettive, e la gestione delle tre fasi principali: Forward Pass, Input Gradient, e Weight Gradient. Rappresenta il "motore di esecuzione" che interpreta il workload generato da AICB e lo traduce in chiamate al sistema di simulazione.

---

## Costruttore

### `Workload::Workload(...)`

**Semantica**: Inizializza il workload caricando la configurazione dal file e preparando le strutture per logging e statistiche.

**Parametri**:
- `std::string run_name`: Nome identificativo della run di simulazione
- `Sys* generator`: Puntatore al nodo Sys proprietario
- `std::string name`: Path al file workload (es. "workload_inputs/gpt3_175B.txt")
- `int TOTAL_PASS`: Numero totale di passaggi (epoch) da eseguire
- `int total_rows`: Numero totale di righe per statistiche aggregate
- `int stat_row`: Indice riga corrente per statistiche
- `std::string path`: Path per output CSV
- `bool seprate_log`: Flag per abilitare logging separato per dimensione

**Processo**:
1. **Inizializzazione variabili di stato**:
   - `initialized = false`: Flag validità
   - `counter = 0`: Delay counter per compute time
   - `delay_loaded = false`: Flag per evitare double-loading di delay
   - `checkpoint_initiated = false`: Traccia inizializzazione checkpoint (per Transformer con recomputation)
   - `collective_issued = false`: Previene doppia emissione di collettive
   - `current_state = LoopState::Forward_Pass`: Stato iniziale FSM
   - `pass_counter = 0`: Contatore epoch completate
   - `index = 0`: Indice layer corrente

2. **Parsing workload**: Chiama `initialize_workload(name)` che legge il file e crea l'array `layers`

3. **Inizializzazione CSV writers** (solo su nodo 0 se `seprate_log == true`):
   - `detailed`: Statistiche dettagliate per layer (`detailed_<num_nodes>.csv`)
   - `end_to_end`: Statistiche end-to-end aggregate (`EndToEnd.csv`)
   - `dimension_utilization`: Utilizzo delle code per dimensione (`<run_name>_dimension_utilization_<offset>.csv`)

4. **Pre-allocazione file**: Se `stat_row == 0`, chiama `initialize_stat_files()` per allocare righe CSV

**Complessità**: O(n) dove n è il numero di layer nel workload

---

## Distruttore

### `Workload::~Workload()`

**Semantica**: Cleanup completo di tutte le risorse allocate.

**Operazioni**:
1. Elimina i CSV writers: `end_to_end`, `detailed`, `dimension_utilization`
2. Elimina tutti gli oggetti `Layer`: `for (i=0; i<SIZE; i++) delete layers[i]`
3. Elimina l'array di puntatori: `delete[] layers`

---

## Parsing e Inizializzazione

### `bool Workload::initialize_workload(std::string name)`

**Semantica**: Cuore del parsing del file workload, legge la configurazione e crea gli oggetti Layer.

**Formato file workload**:

```
<PARALLELISM_POLICY> [model_parallel_NPU_group: N] [ep: E] [pp: P] [vpp: V] [ga: G] [all_gpus: A] [pp_comm: C] [checkpoints: K layer1 layer2...] [checkpoint_initiates: M layer1 layer2...]
<NUM_LAYERS>
<layer_id> <dependency> <fp_compute> <fp_comm_type> <fp_comm_size> <ig_compute> <ig_comm_type> <ig_comm_size> <wg_compute> <wg_comm_type> <wg_comm_size> <wg_update> [<specific_policy>]
...
```

**Prima linea - Parsing configurazione**:

1. **Parallelism Policy**: Identifica la strategia generale (DATA, HYBRID_TRANSFORMER, DLRM, MODEL, ecc.)
   - Convertita tramite `decode_parallelsim()` in enum `ParallelismPolicy`

2. **Parametri di parallelismo** (per Transformer):
   - `model_parallel_NPU_group`: Dimensione gruppo TP (Tensor Parallelism)
   - `ep`: Expert Parallelism size (per MoE models)
   - `pp`: Pipeline Parallelism stages
   - `vpp`: Virtual Pipeline Parallelism (numero virtual stages per GPU)
   - `ga`: Gradient Accumulation microbatches
   - `all_gpus`: Numero totale GPU nel cluster
   - `pp_comm`: Dimensione comunicazione pipeline (se pp > 1)

3. **Checkpoint configuration** (per TransformerFwdInBckwd):
   - `checkpoints: N layer1 layer2 ...`: Lista di N layer che sono checkpoint
   - `checkpoint_initiates: M layer1 layer2 ...`: Lista di M layer che iniziano recomputation

4. **DLRM specific**:
   - `DLRM_LAST_BOTTOM_LAYER: N`: Ultimo layer bottom MLP in DLRM

**Seconda linea - Numero layer**: Intero che specifica quanti layer seguono

**Linee successive - Definizione layer**: Per ogni layer i:

- `id`: Identificatore stringa layer
- `dependency`: (Attualmente non usato, legacy)
- `fp_compute_time`: Cicli computazione forward pass
- `fp_comm_type`: Tipo collettiva FP (ALLREDUCE, ALLGATHER, REDUCESCATTER, ALLTOALL, NONE)
- `fp_comm_size`: Byte da comunicare nel FP
- `ig_compute_time`: Cicli computazione input gradient
- `ig_comm_type`: Tipo collettiva IG
- `ig_comm_size`: Byte da comunicare nell'IG
- `wg_compute_time`: Cicli computazione weight gradient
- `wg_comm_type`: Tipo collettiva WG (tipicamente ALLREDUCE per sincronizzare gradienti)
- `wg_comm_size`: Byte da comunicare nel WG
- `wg_update_time`: Cicli per update dei pesi (optimizer step)
- `specific_policy` (opzionale): Policy specifica per questo layer se HybridCustomized

**Suffissi comunicazione**: I tipi comm possono avere suffissi per specificare il gruppo:
- Nessun suffisso: Gruppo standard (TP per FP/IG, DP per WG)
- `_EP`: Expert Parallelism group
- `_DP_EP`: Data Parallelism dentro Expert Parallelism

**Conversione e creazione Layer**:

1. Converte i tipi comunicazione stringa in enum `ComType` (All_Reduce, All_Gather, etc.)
2. Converte suffissi in `MockNccl::GroupType` (TP, DP, EP, DP_EP)
3. Determina `involved_dimensions` per il layer tramite `decode_involved_dimensions()`:
   - Crea vettori booleani indicanti quali dimensioni della topologia partecipano a ciascuna fase
   - Esempio Transformer: FP/IG usano dimensioni TP, WG usa dimensioni DP
4. Applica scaling factors: `compute_time * compute_scale`, `comm_size * comm_scale`
5. Crea l'oggetto `Layer` con tutti i parametri
6. Marca flag speciali: `is_checkpoint`, `needs_fwd_in_bckwd_initiation`
7. Inserisce in `layers[i]`

**Validazione**:
- Se file non apribile: stampa errore e termina con `exit(1)`
- Se parallelism policy sconosciuta: termina (eccetto in modalità PHY_MTP dove default a TransformerFwdInBckwd)
- Controlla coerenza parametri (warning se mancano parametri obbligatori)

**Return**: `true` se parsing completato con successo, `false` altrimenti

---

### `ParallelismPolicy Workload::decode_parallelsim(std::string parallelism)`

**Semantica**: Converte stringa policy in enumerazione.

**Mapping**:
- `"DATA"` → `ParallelismPolicy::Data`
- `"HYBRID_TRANSFORMER"` → `ParallelismPolicy::Transformer`
- `"HYBRID_TRANSFORMER_FWD_IN_BCKWD"` → `ParallelismPolicy::TransformerFwdInBckwd`
- `"HYBRID_DLRM"` → `ParallelismPolicy::DLRM`
- `"HYBRID_DLRM_ENHANCED"` → `ParallelismPolicy::DLRMEnhanced`
- `"MODEL"` → `ParallelismPolicy::Model`
- `"HYBRID_DATA_MODEL"` → `ParallelismPolicy::HybridDataModel`
- `"HYBRID_MODEL_DATA"` → `ParallelismPolicy::HybridModelData`
- `"HYBRID_CUSTOMIZED"` → `ParallelismPolicy::HybridCustomized`
- `"MICRO"` → `ParallelismPolicy::MicroBenchmark`
- `"DISTRIBUTED_INFERENCE"` → `ParallelismPolicy::DistributedInference`
- Altro → `ParallelismPolicy::None`

---

### `std::map<std::string, std::vector<bool>> Workload::decode_involved_dimensions(...)`

**Semantica**: Determina quali dimensioni della topologia partecipano a ciascuna fase (FP, IG, WG) basandosi sulla parallelism policy.

**Parametri**:
- `ParallelismPolicy policy`: Strategia di parallelismo
- `int model_parallel_npu_group`: Dimensione gruppo model parallelism

**Logica per policy**:

**ParallelismPolicy::Data**:
- `fwd`: Nessuna dimensione (vettore di false) - Nessuna comunicazione in FP
- `ig`: Nessuna dimensione - Nessuna comunicazione in IG
- `wg`: Tutte le dimensioni (vettore di true) - AllReduce su tutte le repliche

**ParallelismPolicy::Model**:
- `fwd`: Tutte le dimensioni - Comunicazione inter-stage nel FP
- `ig`: Tutte le dimensioni - Comunicazione inter-stage nell'IG
- `wg`: Nessuna dimensione - Nessuna sincronizzazione gradienti (ogni GPU ha slice diverso)

**ParallelismPolicy::HybridModelData**:
- `fwd`: Solo dimensione 0 (model parallelism)
- `ig`: Solo dimensione 0
- `wg`: Dimensioni 1+ (data parallelism)

**ParallelismPolicy::HybridDataModel**:
- `fwd`: Dimensioni 1+ (model parallelism)
- `ig`: Dimensioni 1+
- `wg`: Solo dimensione 0 (data parallelism)

**ParallelismPolicy::Transformer / TransformerFwdInBckwd**:
- Chiama `generator->break_dimension(model_parallel_npu_group)` per determinare boundary
- `fwd`: Dimensioni 0 fino a boundary (TP dimensions)
- `ig`: Dimensioni 0 fino a boundary
- `wg`: Dimensioni da boundary+1 in poi (DP dimensions)
- Esempio: boundary=1 → fwd/ig usano dim 0,1 (TP intra-node), wg usa dim 2+ (DP inter-node)

**Return**: Mappa con chiavi `"fwd"`, `"ig"`, `"wg"` contenenti vettori booleani di 10 elementi

---

## Esecuzione e State Machine

### `void Workload::fire()`

**Semantica**: Avvia l'esecuzione del workload chiamando il callback principale.

**Implementazione**: Semplicemente `call(EventType::General, NULL)` per iniziare la FSM.

---

### `void Workload::call(EventType event, CallData* data)`

**Semantica**: Entry point principale della state machine, distribuisce l'esecuzione alle funzioni iterate specifiche.

**Processo**:

1. **Gestione delay**: Se `counter > 0`, re-schedula se stesso dopo `counter` cicli
   - Questo gestisce i delay dovuti a compute time dei layer
   - `generator->try_register_event(this, EventType::Workload_Wait, NULL, counter)`
   - Return per attendere completamento compute

2. **Dispatch basato su policy**: Chiama la funzione iterate appropriata:
   - `ParallelismPolicy::Data` → `iterate_data_parallel()`
   - `ParallelismPolicy::Transformer` → `iterate_hybrid_parallel_Transformer()`
   - `ParallelismPolicy::DLRM` / `DLRMEnhanced` → `iterate_hybrid_parallel_DLRM()`
   - `ParallelismPolicy::MicroBenchmark` → `iterate_micro_benchmark()`
   - `ParallelismPolicy::Model` → `iterate_model_parallel()`
   - `ParallelismPolicy::HybridDataModel` → `iterate_hybrid_parallel_data_model()`
   - `ParallelismPolicy::HybridModelData` → `iterate_hybrid_parallel_model_data()`
   - `ParallelismPolicy::DistributedInference` → `iterate_distributed_inference()`
   - `ParallelismPolicy::TransformerFwdInBckwd` → `iterate_hybrid_parallel_Transformer_fwd_in_bckwd()`
   - `ParallelismPolicy::HybridCustomized` → `iterate_hybrid_parallel_customized()`
   - Altro → `Sys::sys_panic("No known parallelism!")`

**Pattern Callback**: Registrato come `Callable` in Sys, viene richiamato quando eventi schedulati si verificano.

---

## Iterazione Workload - Funzioni Specifiche

### `void Workload::iterate_data_parallel()`

**Semantica**: Implementa la FSM per Data Parallelism puro (es. ResNet, AlexNet standard).

**Stati FSM**: Forward_Pass → Weight_Gradient → Input_Gradient → Forward_Pass (loop)

**Logica per stato**:

**LoopState::Forward_Pass**:
1. **Verifica WG precedente completato**: `layers[index]->is_weight_grad_comm_finished_blocking()`
   - Se false, return (attende completamento AllReduce gradienti layer precedente)
2. **Carica compute time**: Se `delay_loaded == false`, legge `layers[index]->get_fwd_pass_compute()` in `counter`
   - Setta `delay_loaded = true` per evitare rilettura
3. **Applica compute delay**: Se `counter > 0`, schedula evento futuro e return
4. **Avanza layer**: Incrementa `index`, resetta `delay_loaded = false`
5. **Fine forward**: Se `index >= SIZE`, transisce a `Weight_Gradient`, decrementa `index` per partire dall'ultimo layer
6. **Continua**: Schedula prossimo evento con `generator->register_event(this, EventType::General, NULL, 1)`

**LoopState::Weight_Gradient**:
1. **Carica compute time**: `layers[index]->get_weight_grad_compute()`
2. **Applica delay**: Come FP
3. **Emette collettiva WG**: `layers[index]->issue_weight_grad_comm(SchedulingPolicy::None, CollectiveBarrier::Non_Blocking)`
   - `Non_Blocking`: Non attende completamento, continua immediatamente
4. **Fine epoch**: Se `index == 0`:
   - Stampa "pass N finished at time T"
   - Incrementa `pass_counter`
   - Resetta a `Forward_Pass`, `index` rimane 0
5. **Continua backward**: Altrimenti transisce a `Input_Gradient`
6. **Schedula**: `register_event()`

**LoopState::Input_Gradient**:
1. **Carica compute time**: `layers[index]->get_input_grad_compute()`
2. **Applica delay**: Come FP
3. **Decrementa layer**: `index--` (backpropagation procede all'indietro)
4. **Torna a WG**: Transisce a `Weight_Gradient` per processare gradienti del layer precedente
5. **Schedula**: `register_event()`

**Peculiarità Data Parallel**:
- Nessuna comunicazione in FP o IG
- Solo AllReduce in WG per sincronizzare gradienti tra repliche
- Layer processati sequenzialmente in FP, all'indietro in backward

---

### `void Workload::iterate_hybrid_parallel_Transformer()`

**Semantica**: FSM per Transformer con Tensor+Data Parallelism (Megatron-LM style).

**Differenze da Data Parallel**:

**LoopState::Forward_Pass**:
- Aggiunto flag `collective_issued` per gestire collettive blocking
- **Emette collettiva FP**: `layers[index]->issue_forward_pass_comm(SchedulingPolicy::None, CollectiveBarrier::Blocking)`
  - `Blocking`: Attende completamento prima di procedere (necessario per TP perché layer successivo dipende da output)
- Se collettiva non completata, return per attendere

**LoopState::Weight_Gradient**:
- **Policy FIFO**: `issue_weight_grad_comm(SchedulingPolicy::FIFO, ...)`
  - Ordina AllReduce secondo ordine layer per ottimizzare pipelining
- **Attende IG completamento**: `layers[index]->is_input_grad_comm_finished_blocking()`
  - Necessario perché gradienti devono propagarsi completamente prima di update

**LoopState::Input_Gradient**:
- **Emette collettiva IG**: `issue_input_grad_comm(SchedulingPolicy::LIFO, CollectiveBarrier::Blocking)`
  - `LIFO`: Policy Last-In-First-Out per reverse order rispetto a FP
  - Blocking perché layer precedente (in backprop order) dipende da questi gradienti

**Pattern**: FP (compute + comm) → Avanza → IG (compute + comm) ← Indietro ← WG (compute + comm)

---

### `void Workload::iterate_hybrid_parallel_Transformer_fwd_in_bckwd()`

**Semantica**: Estensione di Transformer con **recomputation checkpoint** per ridurre memoria.

**Stato aggiuntivo**: `LoopState::Forward_In_BackPass`

**Logica checkpoint**:

**LoopState::Input_Gradient** - Gestione inizializzazione recomputation:
1. **Verifica necessità recomputation**: Se `layers[index]->needs_fwd_in_bckwd_initiation && !checkpoint_initiated`
2. **Trova ultimo checkpoint**: Scansiona all'indietro finché `layers[index]->is_checkpoint == true`
3. **Inizia recomputation**: Transisce a `Forward_In_BackPass` dallo stesso indice
4. **Setta flag**: `checkpoint_initiated = true` per evitare re-inizializzazione
5. **Log**: Stampa "initiating fwd_in_bkwd starting from layer X to layer Y"

**LoopState::Forward_In_BackPass** - Re-esecuzione forward:
1. **Attende WG completamento**: Stesso check di FP normale
2. **Carica compute time FP**: `get_fwd_pass_compute()`
3. **Applica delay**: Come FP
4. **Emette collettiva FP**: `issue_forward_pass_comm()` per ricreare attivazioni
5. **Avanza**: `index++`
6. **Fine recomputation**: Se `layers[index]->needs_fwd_in_bckwd_initiation`:
   - Ritorna a `Input_Gradient` per riprendere backprop normale
   - Resetta `checkpoint_initiated = false`

**Vantaggi**: Riduce picco memoria scambiando compute (re-esegue FP) con storage (non salva tutte le attivazioni)

**Minimum message size adjustment**:
```cpp
if (layers[index]->fwd_pass_comm_size < 4096 && layers[index]->fwd_pass_comm_size > 0)
    layers[index]->fwd_pass_comm_size = 4096;
```
Questo garantisce che messaggi troppo piccoli vengano arrotondati, simulando overhead minimum payload di NCCL.

---

### `void Workload::iterate_hybrid_parallel_DLRM()`

**Semantica**: FSM specializzata per modelli DLRM (Deep Learning Recommendation Models) con AllToAll per embedding exchange.

**Architettura DLRM**:
- **Bottom MLPs** (layer 0 a DLRM_LAST_BOTTOM_LAYER): Processa feature dense e sparse
- **Interaction layer** (DLRM_LAST_BOTTOM_LAYER + 1): Combina embedding (usa AllToAll)
- **Top MLP** (layer rimanenti): Predizione finale

**Peculiarità**:

**LoopState::Forward_Pass**:
- **AllToAll in bottom layer**: Se `layers[index]->fwd_pass_comm_type == ComType::All_to_All`
  - Emette con `SchedulingPolicy::HIGHEST` (priorità massima)
  - `Non_Blocking` per permettere overlap compute-communication
- **Sincronizzazione interaction layer**: Se `index == DLRM_LAST_BOTTOM_LAYER`
  - Attende completamento AllToAll del layer 0: `layers[0]->is_fwd_pass_comm_finished_blocking()`
  - Questo garantisce che tutti embedding siano scambiati prima di interaction

**LoopState::Weight_Gradient**:
- Standard per DLRM, opzionale check su IG completamento se non DLRMEnhanced

**LoopState::Input_Gradient**:
- **AllToAll inverse in interaction layer**: Se `index == DLRM_LAST_BOTTOM_LAYER + 1`
  - Emette `layers[0]->issue_input_grad_comm(HIGHEST, Non_Blocking)`
  - Distribuisce gradienti embedding alle GPU appropriate

**Differenza Enhanced**: `ParallelismPolicy::DLRMEnhanced` rimuove alcuni check di sincronizzazione per maggiore parallelismo.

---

### `void Workload::iterate_model_parallel()`

**Semantica**: Model Parallelism puro, ogni GPU ha uno slice del modello.

**Differenze chiave**:
- **Comunicazione FP/IG**: Necessaria per passare attivazioni/gradienti tra slice
- **Nessuna comunicazione WG**: Ogni GPU aggiorna solo i suoi parametri (nessuna replica)

**Logica**:
- Forward: Emette FP comm blocking per inviare attivazioni al prossimo slice
- Input Gradient: Attende IG comm completion per ricevere gradienti dal slice successivo
- Weight Gradient: Solo compute, nessuna comm

**Variante**: `involved_dimensions{true, true, true}` hardcoded indica uso di tutte le dimensioni.

---

### `void Workload::iterate_distributed_inference()`

**Semantica**: Modalità inference distribuita (forward-only, nessun backward).

**Semplicità**:
- **Solo Forward Pass**: Nessuno stato Weight_Gradient o Input_Gradient
- Loop: FP (compute + comm) → Avanza → Se fine, ricomincia da index=0
- Incrementa `pass_counter` a ogni completamento (conta batch inferiti)

**Uso**: Simulazione deployment inference su cluster GPU per servizi online.

---

### `void Workload::iterate_micro_benchmark()`

**Semantica**: Esegue solo comunicazione Weight Gradient per benchmark puro.

**Implementazione**:
```cpp
for (pass_counter = 0; pass_counter < TOTAL_PASS; pass_counter++)
    layers[index]->issue_weight_grad_comm(SchedulingPolicy::None, CollectiveBarrier::Non_Blocking);
check_for_sim_end();
```

**Scopo**: Misurare throughput/latenza collettive isolate senza compute overhead.

---

## Terminazione e Reporting

### `void Workload::check_for_sim_end()`

**Semantica**: Verifica se la simulazione è completata e triggera reporting.

**Condizioni terminazione**:
1. **Pass completati**: `pass_counter == TOTAL_PASS`
2. **Tutti stream finiti**: `generator->streams_finished == generator->streams_injected`

**Processo**:
1. **Transizione a Wait_For_Sim_Finish**: Setta `current_state = LoopState::Wait_For_Sim_Finish`
2. **Registrazione callback stream**: Se stream ancora pendenti:
   - Chiama `generator->register_for_finished_stream(this)` per essere notificato quando finiscono
   - Setta `registered_for_finished_streams = true`
3. **Attesa WG completion**: Chiama `layers[0]->is_weight_grad_comm_finished_blocking()` per sincronizzazione finale
4. **Verifica stream match**: Controlla che stream iniettati == stream completati
5. **Reporting**: Se tutto completato:
   - Chiama `report()` (solo su nodo 0)
   - Chiama `generator->workload_finished()` per notificare Sys

**Callback pattern**: Se registrato, `call()` verrà richiamato quando `streams_finished` incrementa.

---

### `void Workload::report()`

**Semantica**: Genera statistiche finali aggregate e chiama report su ogni layer.

**Variabili accumulate**:
- `total_compute`: Tempo totale computazione (FP+IG+WG compute)
- `total_exposed`: Tempo totale comunicazione esposta (non overlapped)
- `pre_bubble_time`: Pipeline bubble time
- `DP_comm`: Comunicazione Data Parallelism (AllReduce WG)
- `DP_EP_comm`: Comunicazione DP dentro Expert Parallelism
- `Expose_TP_comm`: Comunicazione Tensor Parallelism esposta
- `Expose_EP_comm`: Comunicazione Expert Parallelism esposta
- `total_fwd_time`, `total_wg_time`, `total_ig_time`: Vettori [compute, comm, exposed] per fase

**Processo**:
1. **Itera layer**: Per i = 0 a SIZE-1:
   - Chiama `layers[i]->report(...)` passando tutti gli accumulatori per reference
   - Ogni layer aggiunge le sue statistiche agli accumulatori
   - Restituisce `LayerStats` che viene aggiunto a `astraSimDataAPI.layers_stats`

2. **Popola AstraSimDataAPI**:
   - `run_name`: Nome run
   - `workload_finished_time`: Tempo completamento in secondi (`Sys::boostedTick() / FREQ`)
   - `total_compute`: Totale compute time accumulato
   - `total_exposed_comm`: Totale exposed communication
   - `avg_chunk_latency_per_logical_dimension`: Media latenza chunk per dimensione (da scheduler)
     - Converte da cicli a secondi dividendo per FREQ

3. **Passa a network interface**: `generator->NI->pass_front_end_report(astraSimDataAPI)`
   - Questo permette a NS-3 backend di ricevere statistiche per post-processing

4. **Dimension utilization report** (se `seprate_log` in modalità NS3_MTP/NS3_MPI):
   - Itera su `generator->scheduler_unit->usage` per ogni dimensione
   - Chiama `usage[i].report_percentage(10000)` per ottenere utilizzo temporale
   - Scrive in `dimension_utilization` CSV via `finalize_csv(dims)`

**Output**: Stampa su stdout:
```
workload stats for the job scheduled at NPU offset: <offset>
*************************
all passes finished at time: <tick>, id of first layer: <id>
```

---

## Utility

### `int Workload::get_layer_numbers(std::string workload_input)`

**Semantica**: Pre-parsing rapido per determinare numero layer nel workload senza parsing completo.

**Processo**:
1. Apre file `"workload_inputs/" + workload_input`
2. Salta prima linea (header con policy)
3. Legge seconda linea come intero (numero layer)
4. Chiude file e return

**Uso**: Potrebbe essere usato per pre-allocazione, attualmente non chiamato nel codice fornito.

---

## Variabili Membro Chiave

### Stati FSM
- `LoopState current_state`: Stato corrente (Forward_Pass, Input_Gradient, Weight_Gradient, Forward_In_BackPass, Wait_For_Sim_Finish)
- `int index`: Indice layer corrente (0 a SIZE-1)
- `int pass_counter`: Numero pass (epoch) completati
- `int TOTAL_PASS`: Numero totale pass da eseguire

### Flags di controllo
- `bool delay_loaded`: Previene double-loading di compute time
- `bool checkpoint_initiated`: Traccia se recomputation è stata iniziata (TransformerFwdInBckwd)
- `bool collective_issued`: Previene doppia emissione collettive blocking
- `bool registered_for_finished_streams`: Callback registrato per stream completion

### Configurazione parallelismo
- `ParallelismPolicy parallelismPolicy`: Strategia generale
- `int model_parallel_npu_group`: Dimensione gruppo TP
- `int expert_parallel_npu_group`: Dimensione gruppo EP
- `int pipeline_model_parallelism`: Numero stage PP
- `int vpp`: Virtual pipeline stages per GPU
- `int GA`: Gradient Accumulation microbatches
- `int all_gpus`: Totale GPU cluster
- `int pp_commsize`: Dimensione comunicazione PP

### DLRM specific
- `int DLRM_LAST_BOTTOM_LAYER`: Ultimo layer bottom MLP

### Layer e workload
- `Layer** layers`: Array dinamico di puntatori Layer
- `int SIZE`: Numero totale layer nel workload
- `std::string run_type`: Tipo run (dalla prima linea workload)

### Statistics e logging
- `CSVWriter* end_to_end`: Writer per statistiche aggregate
- `CSVWriter* detailed`: Writer per statistiche dettagliate layer
- `CSVWriter* dimension_utilization`: Writer per utilizzo dimensioni
- `std::string path`: Path base per output files
- `int stat_row`: Indice riga per statistiche multi-run
- `int total_rows`: Totale righe statistiche
- `bool seprate_log`: Flag per abilitare logging separato

### System interface
- `Sys* generator`: Puntatore al nodo Sys proprietario
- `Tick counter`: Delay counter per simulate compute time
- `int waiting_for_comm`: (Potenzialmente unused, per tracking comunicazioni pendenti)

---

## Pattern di Esecuzione Tipico

1. **Costruzione**: `new Workload(...)` carica file e crea layer
2. **Avvio**: `workload->fire()` inizia esecuzione chiamando `call()`
3. **Loop FSM**: 
   - `call()` dispatcha a `iterate_<policy>()`
   - Funzione iterate gestisce stato FSM
   - Emette collettive via `layer->issue_*_comm()`
   - Schedula prossima iterazione via `register_event()`
4. **Callback compute**: Quando compute delay scade, `call()` richiamato automaticamente
5. **Callback comm**: Quando collettive completano, layer notifica workload che può procedere
6. **Terminazione**: Dopo TOTAL_PASS pass, `check_for_sim_end()` triggera `report()`
7. **Cleanup**: Distruttore elimina layer e CSV writers

---

## Interazione con Altri Componenti

### Con Sys
- Registra callback: `generator->register_event()`, `try_register_event()`
- Registra listener stream finished: `generator->register_for_finished_stream()`
- Accede scheduler: `generator->scheduler_unit`
- Notifica completamento: `generator->workload_finished()`

### Con Layer
- Legge compute time: `layer->get_fwd_pass_compute()`, ecc.
- Emette collettive: `layer->issue_forward_pass_comm()`, ecc.
- Verifica completion: `layer->is_weight_grad_comm_finished_blocking()`, ecc.

### Con NS-3 Backend
- Passa statistiche: `generator->NI->pass_front_end_report()`

---

## Considerazioni di Design

**State Machine vs Event-Driven**: Workload usa FSM esplicita invece che callback puri per chiarezza logica di training flow

**Blocking vs Non-Blocking collettive**: Scelte diverse per ottimizzare overlap:
- TP/PP collettive: Blocking (dipendenze forti)
- DP AllReduce: Non-Blocking (può overlap con compute layer successivo)

**Checkpoint recomputation**: TransformerFwdInBckwd implementa gradient checkpointing di Megatron-LM per scalare a modelli grandi

**DLRM specialization**: Gestione speciale per AllToAll embedding exchange che domina comunicazione

**Policy-specific iterate**: Separare funzioni invece di mega-if-else migliora leggibilità e manutenibilità

---

*Documentazione generata per SimAI/ASTRA-SIM - Versione file analizzato: Workload.cc*
