Concurrently running transactions should not interfere with each other. If one transaction makes several writes then another transaction should see either all or none of the writes. (No Subset.)

Can be implemented using locks

Hide concurrency issues from users.

# Dirty Read
One client reads another client’s writes before they have been committed.

Any write by transaction only become visible to others when that transaction commits.

When read from a DB, you see only committed read. (no dirty read)

Issues - See data that may be rollback; Partial data read like email unread counter and actual new email. See counter increase but no email.

Dirty reads prevent -
    Using same lock. (But costly in term of read response.)
    For every object, DB remember both the old committed value and new value set by transaction that hold write lock.

# Dirty Write
One client overwrites data that another client has written, but not yet committed.

When write, you will only overwrite data that has been committed (no dirty write). With dirty write, conflict writes from different transactions can be mixed up.

Dirty write prevented using locks at row level.


# Read Skew
A client sees different parts of the database at different points in time.

Alice have 2 accounts 500 each.
He check his accounts. In parallel some one initiate 100 transfer from A to B.
Actions -
    Alice read account B = 500
    Account A deducted 100 = 400
    Account B credit 100 = 600..
    Alice read account A = 400.
Result - Alice see $900 in his account. Yeah, problem will fix after some second.

But some places the issue may be permanent.
    Backup in || when write is happening on actual DB. So, backup may have some updated data and some non-updated data. So, if we retore such stage then it is permanent.


# Lost Updates
Two clients concurrently perform a read-modify-write cycle. One overwrites the
other’s write without incorporating its changes, so data is lost.

Issues when 2 concurrent write on same record. It occur when read, moify and the write updated value.
## Solutions
### Atomic Write/Update Operation
counter increment is supported as atomic operation.
It is not feasible for all type of operation.
Implemented the same using lock.

### Explicit Locking
Select * from figures where name = 'robot' and game_id = 222 For Update;
"For Update" - It ask DB to lock explicitly.

### Automatic Detecting Lost Updates
If there is any lost, abort transaction and force retry.

### Compare and Set
Allow update to happen only if the value is same as at the time of read.

### Conflict resolution and replication
Multi-leader (in replication) - Compare/Set; lock etc. don't work.
Concurrent write create multiple versions of value (siblings) & use application code or special DS to merge those.

# Write Skew
A transaction reads something, makes a decision based on the value it saw, and writes the decision to the database. However, by the time the write is made, the premise of the decision is no longer true. Only serializable isolation prevents this anomaly.

One more issue - 2 doctors in a call and both drop at a time. Ideally there should be one doctor in call. This is Write Skew.
As both operation updating two different objects.
It occurs when 2 transaction read same object and update same/different object.

Issue occur on Snapshot Isolaton

Solution -
1. Serializable Isolation
2. Explicitly lock on row on which transaction depends on.

Another examples -
1. Book Meeting rooms.
2. Multiplayer Game - 2 figures move to same position.
3. Claiming User Email Id

## Phantoms Causing Write Skew
Pattern - Select; Based on result, decide path; Update that effect 1st select query result.
Phantom - Effect "When a write of one transaction change the result of search query of other transaction".

## Materialize Conflict
Sometimes there is no object for lock in case of phantom effect. MC is approach to introduce rows on which apply lock. For e.g. meeting rooms with time slots in a table can be used for lock.



