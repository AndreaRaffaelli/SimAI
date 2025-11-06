# Documentazione Classe Sys - SimAI/ASTRA-SIM

## Panoramica

La classe `Sys` è il cuore del simulatore SimAI, responsabile della gestione e orchestrazione di tutte le operazioni di training distribuito su cluster GPU. Rappresenta un singolo nodo NPU/GPU nel sistema e coordina la comunicazione di rete, la gestione della memoria, lo scheduling degli stream e l'esecuzione del workload.

---

## Costruttore

### `Sys::Sys(...)`

**Semantica**: Inizializza un nodo GPU/NPU nel simulatore creando tutte le strutture dati e componenti necessari per la simulazione.

**Parametri**:
- `AstraNetworkAPI* NI`: Interfaccia di rete (NS-3) per gestire le comunicazioni
- `AstraMemoryAPI* MEM`: Interfaccia per modellare accessi alla memoria
- `int id`: Identificatore univoco del nodo (rank)
- `int npu_offset`: Offset per l'indicizzazione globale dei nodi
- `int num_passes`: Numero di passaggi attraverso il workload
- `std::vector<int> physical_dims`: Dimensioni fisiche della topologia (es. [8, 16] per 128 GPU)
- `std::vector<int> queues_per_dim`: Numero di code per dimensione
- `std::string my_sys`: Path al file di configurazione sistema
- `std::string my_workload`: Path al file workload
- `float comm_scale, compute_scale, injection_scale`: Fattori di scala per tempi
- Altri parametri per logging, rendezvous, tipo GPU, ecc.

**Operazioni principali**:
1. **Parsing configurazione**: Legge parametri da file di sistema e li processa tramite `post_process_inputs()`
2. **Calcolo topologia**: Determina il numero totale di nodi moltiplicando le dimensioni fisiche
3. **Creazione code**: Alloca code di comunicazione (`active_Streams`) per ogni dimensione secondo `queues_per_dim`
4. **Inizializzazione componenti**:
   - `SchedulerUnit`: Gestisce lo scheduling degli stream
   - `QueueLevels`: Organizza le code gerarchicamente
   - `MemBus`: Modella il bus memoria-NPU
   - `Workload`: Carica il carico di lavoro
5. **Topologie logiche**: Crea topologie separate per AllReduce, ReduceScatter, AllGather, AllToAll
6. **Inizializzazione NCCL mock**: Se in modalità NS3, inizializza gruppi NCCL simulati

**Complessità**: O(n) dove n è il numero totale di dimensioni

---

## Gestione Dimensioni

### `int Sys::break_dimension(int model_parallel_npu_group)`

**Semantica**: Divide una dimensione fisica in due sottodimensioni logiche quando il numero di NPU richiesti per il model parallelism non corrisponde esattamente al prodotto delle dimensioni.

**Esempio**: 
- Dimensioni fisiche: [8, 8] (64 GPU)
- Richiesto: 16 GPU per gruppo
- Risultato: [2, 4, 8] dove 2×4 = 8 diventa la prima dimensione

**Processo**:
1. Identifica quale dimensione dividere iterando finché `all_npus × physical_dims[dim]` supera il target
2. Calcola le due sottodimensioni: `first_subdim = target / all_npus`, `second_subdim = dim_size / first_subdim`
3. **Ricostruisce tutte le strutture**:
   - Elimina e ricrea `scheduler_unit` e `vLevels`
   - Inserisce una nuova coda in `queues_per_dim`
   - Clona le implementazioni collettive per la nuova dimensione
   - Ricrea tutte le topologie logiche con le nuove dimensioni
4. Salva le dimensioni logiche in `logical_broken_dims`

**Return**: Indice della dimensione divisa, o -1 se non necessario

---

## Comunicazione di Rete

### `int Sys::sim_send(...)`

**Semantica**: Invia un messaggio sulla rete simulata, gestendo la serializzazione delle send per evitare conflitti sullo stesso tag.

**Parametri**:
- `Tick delay`: Ritardo prima dell'invio (in cicli)
- `void* buffer`: Buffer dati (spesso dummy)
- `uint64_t count`: Dimensione messaggio
- `int dst`: Nodo destinazione
- `int tag`: Tag per identificare il messaggio
- `sim_request* request`: Richiesta contenente metadati
- Callback e argomenti

**Logica**:
1. Se `delay == 0` e non ci sono send pendenti sullo stesso `(dst, tag)`:
   - Marca il canale come occupato in `is_there_pending_sends`
   - Invia immediatamente tramite `NI->sim_send()`
2. Altrimenti:
   - Accoda la send in `pending_sends[make_pair(dst, tag)]`
   - Quando il canale si libera (in `handleEvent::PacketSent`), riprende la prima send in coda

**Thread Safety**: Usa `sysCriticalSection` per proteggere accessi concorrenti alle mappe

### `int Sys::sim_recv(...)`

**Semantica**: Registra una receive sulla rete, delegando immediatamente a NS-3 o schedulandola con delay.

**Differenza con send**: Le receive non richiedono serializzazione perché NS-3 gestisce il matching dei messaggi internamente.

### `int Sys::rendezvous_sim_send(...)` e `rendezvous_sim_recv(...)`

**Semantica**: Implementano un protocollo rendezvous two-sided per messaggi grandi, riducendo l'occupazione di buffer.

**Flusso rendezvous send**:
1. Receiver invia prima un piccolo messaggio di controllo (8192 byte) con `tag + 500000000`
2. Quando il sender riceve questo controllo, inizia la send vera e propria
3. Evita che il sender occupi buffer di rete prima che il receiver sia pronto

---

## Gestione Memoria

### `Tick Sys::mem_read(uint64_t bytes)` e `mem_write(uint64_t bytes)`

**Semantica**: Modellano la latenza di accesso alla memoria convertendo i nanosecondi restituiti da `MEM->npu_mem_read/write()` in cicli di simulazione.

**Formula**: `delay_cycles = delay_ns / CLOCK_PERIOD`

**Uso**: Chiamati dagli algoritmi collettivi per contabilizzare i tempi di lettura/scrittura di gradienti e attivazioni.

---

## Operazioni Collettive

### `DataSet* Sys::generate_all_reduce(...)`, `generate_all_gather(...)`, ecc.

**Semantica**: Factory methods che creano operazioni collettive complete scomponendole in fasi attraverso le dimensioni.

**Delega a**: `generate_collective()` che implementa la logica generale.

### `DataSet* Sys::generate_collective(...)`

**Semantica**: Cuore della generazione collettive, crea uno stream di fase collettive attraversando le dimensioni della topologia.

**Processo**:
1. **Chunking**: Divide i dati in chunk secondo `determine_chunk_size()`
2. **Dimensioni mapping**: Crea `dim_mapper` per attraversare le dimensioni nell'ordine corretto
   - AllGather: ordine inverso
   - RoundRobin: rotazione delle dimensioni
   - OfflineGreedy: ordine ottimizzato
3. **Generazione fasi per ogni chunk**:
   - **Baseline**: Attraversa ogni dimensione applicando l'operazione collettiva specificata
   - **LocalBWAware**: Ottimizza AllReduce come ReduceScatter → AllGather
   - **Hierarchical AllReduce**: Riduce prime dimensioni, AllReduce sulla dimensione centrale, AllGather sulle precedenti
4. **Creazione stream**: Ogni chunk diventa uno `StreamBaseline` con la lista di fasi
5. **Inserimento in ready_list**: Gli stream vengono accodati per l'esecuzione

**Return**: `DataSet` contenente tutti gli stream generati

### `CollectivePhase Sys::generate_collective_phase(...)`

**Semantica**: Crea una singola fase collettiva (un passo di comunicazione su una dimensione) scegliendo l'algoritmo appropriato.

**Algoritmi supportati**:
- **Ring**: Comunicazione ad anello per bandwidth aggregata massima
- **DoubleBinaryTree**: Albero binario per ridurre latenza
- **HalvingDoubling**: Algoritmo ricorsivo per potenze di 2
- **Direct/AllToAll**: Comunicazione diretta punto-a-punto
- **NcclFlowModel**: Simula esattamente i protocolli NCCL (Ring, Tree, NVLS)

**Selezione NcclFlowModel**:
1. Determina lo stato corrente del workload (Forward, Input Gradient, Weight Gradient)
2. Recupera `nccl_info` per identificare l'algoritmo NCCL (RING, TREE, NVLS)
3. Genera i flow models tramite `generate_flow_model()` che contengono:
   - Mapping dei flussi tra GPU
   - Dipendenze parent/child tra flussi
   - Chunking interno di NCCL
4. Crea un `NcclTreeFlowModel` con i flussi e i canali NCCL

**Return**: `CollectivePhase` contenente l'algoritmo istanziato e metadati

---

## NCCL Mock

### `bool Sys::mock_nccl_grobal_group_init()`

**Semantica**: Inizializza il gruppo NCCL globale calcolando i gruppi di parallelismo dal workload.

**Calcoli**:
- `TP_size`: Tensor Parallelism = `model_parallel_npu_group`
- `PP_size`: Pipeline Parallelism = 1 (fisso nella versione attuale)
- `DP_size`: Data Parallelism = `total_gpus / (TP × PP)`
- `EP_size`: Expert Parallelism = `expert_parallel_npu_group`
- `DP_EP_size`: Data Parallelism per Expert = `DP / EP`

**Creazione**: Istanzia `MockNcclGroup` che rappresenta l'intera topologia NCCL globale

### `bool Sys::mock_nccl_comms_init()`

**Semantica**: Crea comunicatori NCCL separati per ogni tipo di parallelismo attivo nel workload.

**Comunicatori creati**:
- Se TP_size > 1: `mock_nccl_comms[TP]`
- Se DP_size > 1: `mock_nccl_comms[DP]`
- Se EP_size > 1: `mock_nccl_comms[EP]`
- Se DP_EP_size > 1: `mock_nccl_comms[DP_EP]`

**Uso**: Ogni stream collettivo interroga il comunicatore appropriato per ottenere i flow models.

### `ncclInfo* Sys::get_nccl_Info(ParallelStrategy comm_ps, uint64_t data_size, ComType collective_type)`

**Semantica**: Interroga il comunicatore NCCL per determinare quale algoritmo (RING, TREE, NVLS) usare per una data dimensione messaggio.

**Decision tree interno a NCCL**:
- Messaggi piccoli (<64KB): LL o LL128 su Tree
- Messaggi medi (64KB-1MB): Simple su Ring
- Messaggi grandi (>1MB): Simple su Ring o NVLS se disponibile

### `std::shared_ptr Sys::generate_flow_model(...)`

**Semantica**: Genera i flussi punto-a-punto dettagliati che implementano l'operazione collettiva, replicando esattamente la logica NCCL.

**Output**: 
- `RingFlowModel`: Mappa di `<rank, flow_id>` → `SingleFlow` contenente src, dest, size, dipendenze
- `TreeFlowModel`: Simile ma con struttura ad albero invece che ring
- Contiene tutti i chunk e step necessari per completare la collettiva

---

## Scheduling

### `SchedulerUnit`

Sottoclasse interna che gestisce lo scheduling degli stream sulle code.

#### `void SchedulerUnit::notify_stream_added(int vnet)`

**Semantica**: Chiamata quando uno stream viene aggiunto a una coda, inizializza fino a `queue_threshold` stream in parallelo sulla coda.

**Processo**:
1. Incrementa `total_active_chunks_per_dimension` per tracciare utilizzo
2. Posiziona `stream_pointer` all'inizio degli stream non inizializzati
3. Chiama `init()` su fino a `queue_threshold` stream per avviarne l'esecuzione
4. Incrementa `running_streams[vnet]`

#### `void SchedulerUnit::notify_stream_removed(int vnet, Tick running_time)`

**Semantica**: Chiamata quando uno stream completa una fase, aggiorna statistiche e schedula nuovi stream.

**Operazioni**:
1. Decrementa `running_streams[vnet]`
2. Aggiorna `latency_per_dimension` e `total_chunks_per_dimension` per statistiche
3. Se la ready list ha stream pendenti, schedula fino al limite (`max_running_streams`)
4. Avanza `stream_pointer` e inizializza nuovi stream disponibili

#### `void SchedulerUnit::notify_stream_added_into_ready_list()`

**Semantica**: Quando uno stream entra nella ready list, verifica se ci sono slot disponibili per eseguirlo immediatamente.

**Condizioni**: Se `first_phase_streams < ready_list_threshold` e `total_running_streams < max_running_streams`, chiama `sys->schedule(max)`.

### `void Sys::schedule(int num)`

**Semantica**: Preleva `num` stream dalla ready list e li promuove alla fase successiva chiamando `proceed_to_next_vnet_baseline()`.

**Limitazioni**: Il numero di stream schedulati è il minimo tra `num` richiesto e `ready_list.size()`.

### `void Sys::proceed_to_next_vnet_baseline(StreamBaseline* stream)`

**Semantica**: Transizione di stato fondamentale che muove uno stream da una fase alla successiva o lo completa.

**Flusso**:
1. **Cleanup fase precedente**:
   - Decrementa `first_phase_streams` dopo la prima fase
   - Calcola media latenza messaggi in `net_message_latency`
   - Elimina l'algoritmo collettivo della fase completata
   - Rimuove stream dalla coda attiva `active_Streams[old_queue_id]`

2. **Completamento stream**:
   - Se `phases_to_go.size() == 0`:
     - Calcola statistiche finali (`take_bus_stats_average()`)
     - Notifica il dataset (`notify_stream_finished()`)
     - Decrementa `total_running_streams`
     - Elimina lo stream
     - Return

3. **Avanzamento alla fase successiva**:
   - Incrementa `steps_finished`
   - Estrae la prossima fase da `phases_to_go`
   - Aggiorna `current_queue_id` e `current_com_type`
   - Resetta contatori (`test`, `test2`, `initialized`)
   - Inserisce stream nella nuova coda attiva
   - Cambia stato a `StreamState::Ready`
   - Chiama `scheduler_unit->notify_stream_removed()` per la vecchia coda
   - Chiama `scheduler_unit->notify_stream_added()` per la nuova coda

### `void Sys::insert_stream(std::list<BaseStream*>* queue, BaseStream* baseStream)`

**Semantica**: Inserisce uno stream in una coda rispettando la politica di scheduling intra-dimensione.

**Politiche**:

- **FIFO**: Inserisce dopo tutti gli stream con priorità >= baseStream.priority
- **RG (Reduce-Gather)**: Come FIFO ma mantiene adiacenti coppie ReduceScatter-AllGather
- **SmallestFirst**: Ordina per `initial_data_size` crescente
- **LessRemainingPhaseFirst**: Ordina per `phases_to_go.size()` crescente

**Regola comune**: Gli stream già inizializzati (`initialized == true`) restano sempre in testa.

---

## Event Handling

### `void Sys::try_register_event(Callable* callable, EventType event, CallData* callData, Tick& cycles)`

**Semantica**: Registra un evento da eseguire dopo `cycles` cicli di simulazione, schedulandolo su NS-3 se necessario.

**Processo**:
1. Calcola il timestamp assoluto: `Sys::boostedTick() + cycles`
2. Protegge con critical section (NS3_MTP)
3. Se non esiste una lista eventi per quel tick:
   - Crea `event_queue[tick]`
   - Setta `should_schedule = true`
4. Aggiunge `(callable, event, callData)` alla lista
5. Incrementa `pending_events`
6. Se `should_schedule`, chiama `NI->sim_schedule()` per registrare l'evento su NS-3

**Output**: Resetta `cycles` a 0 per evitare double-scheduling

### `void Sys::call_events()`

**Semantica**: Esegue tutti gli eventi schedulati per il tick corrente, chiamata periodicamente da NS-3.

**Processo**:
1. Verifica se esiste `event_queue[Sys::boostedTick()]`
2. Itera su tutti gli eventi schedulati per questo tick
3. Chiama `callable->call(event, callData)` per ciascuno
4. Gestisce eccezioni se un callable è stato eliminato
5. Pulisce la entry dall'event_queue
6. **Controllo terminazione**: Se `finished_workloads == 1` e non ci sono eventi pendenti, elimina il nodo

### `void Sys::handleEvent(void* arg)`

**Semantica**: Handler statico per diversi tipi di eventi, distribuisce agli handler specifici.

**Eventi gestiti**:

- **CallEvents**: Chiama `iterate()` che a sua volta chiama `call_events()`
- **RendezvousSend**: Completa protocollo rendezvous, chiama callback originale
- **RendezvousRecv**: Simile per receive
- **PacketReceived**: Passa i dati allo stream owner tramite `owner->consume()`
- **PacketSent**: Gestisce completamento send:
  - Verifica se ci sono send pendenti sulla stessa coppia `(dst, tag)`
  - Se sì, preleva la prossima send dalla coda e la esegue
  - Se no, marca il canale come libero
  - Verifica condizioni di terminazione
- **PacketSentFinished**: Callback per stream che ha completato l'invio

---

## Parsing Configurazione

### `bool Sys::parse_var(std::string var, std::string value)`

**Semantica**: Interpreta una coppia variabile-valore dal file di configurazione sistema.

**Variabili riconosciute**:
- `scheduling-policy`: LIFO o FIFO
- `all-reduce-implementation`, `reduce-scatter-implementation`, ecc.: Algoritmi per dimensione (es. "ring_doubleBinaryTree_direct")
- `collective-optimization`: baseline o localBWAware
- `endpoint-delay`: Latenza comunicazione base
- `local-reduction-delay`: Latenza operazioni di riduzione locali
- `active-chunks-per-dimension`: Chunking granularity
- `L`, `o`, `g`, `G`: Parametri bus memoria
- `intra-dimension-scheduling`: FIFO, RG, SmallestFirst, LessRemainingPhaseFirst
- `inter-dimension-scheduling`: Ascending, RoundRobin, OfflineGreedy, OfflineGreedyFlex

**Conversioni**: Applica scaling factors (`injection_scale`) alle latenze.

### `bool Sys::post_process_inputs()`

**Semantica**: Valida e converte gli input testuali in enumerazioni e strutture dati interne.

**Operazioni**:
1. Chiama `generate_collective_implementation_from_input()` per ogni tipo di collettiva
2. Verifica che ogni stringa di implementazione sia valida
3. Converte `inp_collective_optimization` in enum `CollectiveOptimization`
4. Converte `inp_scheduling_policy` in enum `SchedulingPolicy`
5. Converte flag booleani (boost_mode, model_shared_bus)

**Panic**: Se trova valori non riconosciuti, chiama `sys_panic()` e termina.

### `std::vector<CollectiveImplementation*> Sys::generate_collective_implementation_from_input(std::string input)`

**Semantica**: Parsifica una stringa come "ring_doubleBinaryTree_direct" in vettore di implementazioni, una per dimensione.

**Formato input**: Algoritmi separati da underscore, uno per ogni dimensione.

**Algoritmi riconosciuti**:
- `ring`, `oneRing`: Ring standard o singolo ring
- `doubleBinaryTree`: Albero binario doppio
- `direct`, `oneDirect`: Comunicazione diretta (con window opzionale)
- `halvingDoubling`, `oneHalvingDoubling`: Recursive halving/doubling
- `NcclFlowModel`: Usa i protocolli NCCL reali
- `ncclRingTreeModel`: Variante Ring/Tree di NCCL

**Output**: Vettore di puntatori `CollectiveImplementation`, uno per dimensione topologica.

---

## Utility

### `uint64_t Sys::determine_chunk_size(uint64_t size, ComType type)`

**Semantica**: Calcola la dimensione dei chunk in cui dividere un messaggio collettivo.

**Formula attuale**: `chunk_size = size / preferred_dataset_splits`

**Motivazione**: Chunking permette pipelining e riduce la memoria richiesta, migliorando efficienza e overlapping computation-communication.

### `Tick Sys::boostedTick()`

**Semantica**: Restituisce il tempo simulazione corrente come numero di cicli, sincronizzato tra tutti i nodi.

**Implementazione**:
1. Trova un nodo valido in `all_generators` (tipicamente nodo 0)
2. Legge il tempo NS-3 tramite `NI->sim_get_time()`
3. Converte nanosecondi in cicli: `tick = time_val / CLOCK_PERIOD`
4. Aggiunge offset globale `Sys::offset`

**Uso**: Tutti i timestamp sono basati su questo metodo per garantire coerenza.

### `std::vector<std::string> Sys::split_string(std::string str, std::string sep)`

**Semantica**: Divide una stringa usando un separatore, utile per parsing di input multi-valore.

**Implementazione**: Usa `strtok` C-style per tokenizzazione.

### `int Sys::nextPowerOf2(int n)`

**Semantica**: Calcola la prossima potenza di 2 >= n.

**Algoritmo**: Conta gli shift necessari per azzerare n, poi restituisce 1 << count.

### `void Sys::sys_panic(std::string msg)`

**Semantica**: Termina la simulazione stampando un messaggio di errore fatale.

**Uso**: Chiamato per errori di configurazione non recuperabili.

### `int Sys::get_priority(SchedulingPolicy pref_scheduling)`

**Semantica**: Assegna una priorità a uno stream secondo la politica di scheduling.

**Logica**:
- Se `pref_scheduling == HIGHEST`: Return 100000000 (priorità massima)
- Altrimenti usa `priority_counter`:
  - Se `scheduling_policy == LIFO`: Incrementa counter (priorità crescente = ultimi arrivati eseguiti prima)
  - Se `scheduling_policy == FIFO`: Decrementa counter (priorità decrescente = primi arrivati eseguiti prima)

---

## Gestione Workload

### `void Sys::register_for_finished_stream(Callable* callable)`

**Semantica**: Registra un callback da chiamare quando uno stream finisce, usato dal Workload per tracciare progresso.

### `void Sys::increase_finished_streams(int amount)`

**Semantica**: Incrementa il contatore degli stream completati e notifica tutti i listener registrati.

**Processo**:
1. Incrementa `streams_finished`
2. Itera su `registered_for_finished_stream_event`
3. Chiama `c->call(EventType::StreamsFinishedIncrease, nullptr)` per ogni listener

---

## Thread Safety

### `sysCriticalSection`

**Semantica**: RAII wrapper per sezioni critiche che garantisce mutua esclusione in modalità multi-threaded (NS3_MTP, PHY_MTP).

**Uso**:
```cpp
{
    Sys::sysCriticalSection cs;
    // Accesso a strutture dati condivise
    cs.ExitSection(); // Opzionale, distruttore lo fa automaticamente
}
```

**Implementazione**: Usa `g_sys_inCriticalSection` atomic per spin-lock.

**Protezione**: Usata per accedere a:
- `event_queue`
- `pending_sends`
- `is_there_pending_sends`
- `mock_nccl_comms`
- `waiting_to_notify_receiver` (nella tua domanda originale sul segfault)

---

## Distruttore

### `Sys::~Sys()`

**Semantica**: Cleanup completo di tutte le risorse allocate e calcolo statistiche finali.

**Processo**:
1. Calcola durata simulazione: `end_sim_time - start_sim_time`
2. Se nodo 0, stampa statistiche:
   - Implementazioni collettive usate
   - Durata totale simulazione
   - Stream iniettati vs completati
   - Percentuale completamento
3. **Deallocazione**:
   - Rimuove questo nodo da `all_generators`
   - Elimina tutte le topologie logiche
   - Elimina tutte le implementazioni collettive per dimensione
   - Elimina scheduler_unit, vLevels, memBus, workload, offline_greedy
4. **Terminazione simulazione**:
   - Verifica se tutti i nodi hanno terminato controllando `all_generators`
   - Se sì, chiama `exitSimLoop("Exiting")` per terminare NS-3

---

## Variabili Globali Statiche

- `std::vector<Sys*> all_generators`: Array di tutti i nodi, indicizzato per ID
- `std::atomic<bool> g_sys_inCriticalSection`: Flag per spin-lock delle critical section
- `Tick offset`: Offset tempo globale per sincronizzazione
- `uint8_t* dummy_data`: Buffer dummy condiviso per send/recv che non trasportano dati reali

---

## Pattern di Utilizzo Tipico

1. **Costruzione**: Crea un nodo Sys per ogni GPU/NPU nel cluster
2. **Inizializzazione Workload**: `workload->fire()` per avviare il primo layer
3. **Loop Eventi**: NS-3 chiama ripetutamente `handleEvent(CallEvents)` che esegue `call_events()`
4. **Generazione Collettive**: Il workload chiama `generate_all_reduce()` e simili per ogni layer
5. **Scheduling Stream**: Gli stream vengono accodati, schedulati, e eseguiti fase per fase
6. **Completamento**: Quando tutti gli stream finiscono, il distruttore viene chiamato e termina la simulazione

---

## Ottimizzazioni e Considerazioni

**Chunking**: Dividere messaggi grandi in chunk permette pipelining e riduce memoria
**Multi-threading**: Modalità NS3_MTP usa critical section per accessi sicuri
**Rendezvous**: Riduce occupazione buffer per messaggi grandi
**Caching**: `is_there_pending_sends` evita lookup ripetuti in `pending_sends`
**Lazy Scheduling**: Stream non vengono inizializzati finché non c'è capacità disponibile

---

## Debugging

**MockNcclLog**: Sistema di logging centralizzato per tracciare flow collettivi
**Statistiche per dimensione**: `SchedulerUnit` traccia latenza media per dimensione
**Panic on error**: Configurazioni invalide terminano immediatamente con messaggio chiaro

---

*Documentazione generata per SimAI/ASTRA-SIM - Versione file analizzato: Sys.cc*
