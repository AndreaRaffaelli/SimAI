# How RDMA gets simulated in NS3

## Software Stack

- **RDMA Client**: Provides user-friendly methods to initialize a RDMA communication hiding `RDMA Driver`
  - [Header](./../ns-3-alibabacloud/simulation/src/applications/model/rdma-client.h)
  - [Implementation](./../ns-3-alibabacloud/simulation/src/applications/model/rdma-client.cc`)
- **RDMA Driver**: Incapsulate the RDMA Hardare `ns3::RdmaHw` and Node `ns3::Node` and provides access to private fields. It lets to create a RDMA Queue pair and adds trace method to the queue pair. One queue pair per driver?
  - [Header](../ns-3-alibabacloud/simulation/src/point-to-point/model/rdma-driver.h)
  - [Implementation](../ns-3-alibabacloud/simulation/src/point-to-point/model/rdma-driver.cc)
- **RDMA Hardware**: models the hardware abstraction for RDMA-enabled NICs in ns-3, managing queue pairs, routing, and congestion control mechanisms. It enables simulation of various RDMA transport protocols and congestion control algorithms, such as DCQCN, TIMELY, DCTCP, and HPCC-PINT.
  - [Header](../ns-3-alibabacloud/simulation/build/include/ns3/rdma-hw.h)
  - [Implementation](../ns-3-alibabacloud/simulation/src/point-to-point/model/rdma-hw.cc)
- **RDMA Queue Pair**: models an RDMA (Remote Direct Memory Access) channel in ns-3, representing the state and parameters of a Queue Pair for RDMA communication. It characterizes the RDMA channel by tracking connection endpoints, window sizes, rates, sequence numbers, and protocol-specific runtime states for congestion control and flow management.
  - [Header](../ns-3-alibabacloud/simulation/src/point-to-point/model/rdma-queue-pair.h)
  - [Implementation](../ns-3-alibabacloud/simulation/src/point-to-point/model/rdma-queue-pair.cc)

- **Qbb Net Device**: 