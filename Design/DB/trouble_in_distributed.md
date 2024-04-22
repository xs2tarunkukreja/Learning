# Distributed
A good software either work or fail completely on single system. Predictable.
Checkpoint. Fail to start from last checkpoint.

Distributed - Partial Failure.
If we want to make distributed systems work, we must accept the possibility of partial failure and build fault-tolerance mechanisms into the software. In other words, we need to build a reliable system from unreliable components.


# Node Outage or Replica Failover

# Replication Lag

# Concurrency Control - Transactions

# Unreliable Network
## Request and Response
1. Request lost
2. Request waiting in queue
3. Remote Node failed.
4. Remote Node is temp unavailable.
5. Processed request by remote node but response lost.
5. Processed request by remote node but response delayed.

Impossible to tell why? If not get reply.

## Detecting Faults
Automatic Detect Fault Node
    Load Balancer

## Synchronous V/S Asynchronous Network.
Synchronous On telephone line, each call is assigned a fix close network connection/bandwidth. So, it was reliable. Statically Division. So, resource utilization is not proper.

Async can't use same as it is for bursty traffic. It bandwidth requirement vary. Dynamic Division. It cause queue and delay. So, you can't predict maximum delay.


# Unreliable Clock
Duration or Point in Time.
Each machine have a clock - quartz crystal oscillator - Not Perfectly Accurate

Sync clocks - Network Time Protocol (NTP).

GPS - More Accurate.

Time of day clocks - Actual date and time. NTP - Backward or Forward. Backward can cause confusion.
Monotonic clocks - Suitable to measure a duration. Absolute value has no meaning. NTP may for frequency to change.

VM share CPU. So, each VM is paused for 10 of millisec.. so, from application point of view.. clock jump forward.

User machine clock is completly non-trustable.


#  