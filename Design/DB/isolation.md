# Isolation
It talk about concurrent operations. Race condition when 2 clients are performing action on same record.

Initial counter = 42
counter++
counter++
Ideally counter = 44

Concurrently running transactions should not interfere with each other. If one transaction makes several writes then another transaction should see either all or none of the writes. (No Subset.)

Can be implemented using locks

Hide concurrency issues from users.

## Read Uncommitted
Prevent dirty writes, but doesn't prevent dirty read.

## Read Committed
Prevent Dirty read and write.

## Snapshot Isolation & Repeatable Read
Weaker than serializable.

Each transaction read from a consistent snapshot of DB. Transaction see all the data that was committed in DB at the start of transaction.

It is boon for long running, read only queries such as backup and analytics.

Write Skew and Phantom are not handled by Snapshot Isolation.

Implementation -
Write lock + DB keep several different committed version of an object.
It is known as 
#### Multi Version Concurrency Control

## Serializable Isolation
2 transactions results in such way as they run in serial order. (Even if they are actually running in parallel.)

It prevent all race conditions.

### Solutions or Implementation - 3 ways 

#### Literally Execute the statement in serial order
Stored Procedure - Complete transaction queries in one go to DB. Now flow between application and db.
It use single CPU Core as write in happening on single thread.
VoltDB is use this
##### Partition - So, use N Cores.

It is better for small and fast transaction. Data in RAM.

#### Two Phase Lock - 2PL
Several reads are allowed to same object.
In 2PL, writers don’t just block other writers; they also block readers and vice versa.

• If transaction A has read an object and transaction B wants to write to that
object, B must wait until A commits or aborts before it can continue.

• If transaction A has written an object and transaction B wants to read that object,
B must wait until A commits or aborts before it can continue.

[Snapshot - Reader never block writes and write never block read.]

##### Implementation
Each object have a lock on DB. Lock is shared mode or exclusive mode.
Reader will acquire lock in shared mode. It will be blocked in some transaction have exclusive mode lock (Writer).
Writer will acquire Exclusive mode.
Any transaction first read and then write so, it can upgrade its lock from shared to exclusive.

Two Phase - 1st phase) Lock is acquired. 2) When lock is released (in end of transaction.)
It cause deadlock. DB detect it and kill one transaction.

###### Predicate Lock
It is like shared or exclusive lock but belong to all objects that match some search condition.
Room Book - Search for all booking for a room from a time range.

Select statement try to acquired shared predict lock if any object has exclusive lock on any object on condition matching, select lock will wait.
Also insert/update will check if the object is part of any predict lock before updating.

In short, predict lock is on not existing objects.

###### Index Range Lock
Predict Lock performance is not good as check all active locks in not good option.
Index Range Lock means bigger range. Example room 123 between 12 to 1. So room 123 for anytime, any room between 12 to 1.
So if you have an index on room then select query will add shared lock on room 123 index.

#### Optimistic Concurrency Control Technique such as Serializable Snapshot Isolation
It has good performance.
2 Phase is Pessimistic Concurrency Control - It is better to wait until situation is safe again before doing anything.

Instead of blocking if something potentially dangerous happens, transactions continue anyway, in the hope that everything will turn out all right. When a transaction wants to commit, the database checks
whether anything bad happened (i.e., whether isolation was violated); if so, the transaction is aborted or retried.

Perform bad if there is lot of contention. Contention can be reduced with cummulative atomic operation.

It work on Snapshot Isolation.

##### Premise
A fact that is true in starting of transaction. There are more than 1 Doctor in call.
DB assume that any change in query result means that write in transaction is invalid.
How DB know that there is a change in query result:-
• Detecting reads of a stale MVCC object version (uncommitted write occurred before the read)  
    DB needs to track when a transaction ignore another transaction write due to MVCC's visibility rules. When a transaction want to commit, it check whether any of the ignored write have now be commited.
• Detecting writes that affect prior reads (the write occurs after the read)
    If 2 transactions are reading a data shift_id=1234, index data is used to record the information that 2 transaction 42 and 43 have read it.
    While writing, it check for indexes if the same data is read by someone else. It notify the other transaction that data is updated.
