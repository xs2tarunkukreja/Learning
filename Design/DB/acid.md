# Atomicity
Multiple operations behave as a single operation. Either all success or all fails.

Implemented through LOG for crash recovery.

Leaderless doesn't support this.

# Consistency
Application rules must be satisfied and data should be in good state. Example - Account system. Money debit and credit across all accounts must be balanced.

Some constraints are supported by DB. Other must be taken care by application.
## Consistency is property of Application.

# Isolation
It talk about concurrent operations. Race condition when 2 clients are performing action on same record.

Initial counter = 42
counter++
counter++
Ideally counter = 44

Concurrently running transactions should not interfere with each other. If one transaction makes several writes then another transaction should see either all or none of the writes. (No Subset.)

Can be implemented using locks

Hide concurrency issues from users.

## Serializable Isolation
2 transactions results in such way as they run in serial order. (Even if they are actually running in parallel.)

## Preventing Lost Updates
Issues when 2 concurrent write on same record. It occur when read, moify and the write updated value.
### Solutions
#### Atomic Write/Update Operation
counter increment is supported as atomic operation.
It is not feasible for all type of operation.
Implemented the same using lock.

#### Explicit Locking
Select * from figures where name = 'robot' and game_id = 222 For Update;
"For Update" - It ask DB to lock explicitly.

#### Automatic Detecting Lost Updates
If there is any lost, abort transaction and force retry.

#### Compare and Set
Allow update to happen only if the value is same as at the time of read.

#### Conflict resolution and replication


## Snapshot Isolation & Repeatable Read
Weaker than serializable.

Each transaction read from a consistent snapshot of DB. Transaction see all the data that was committed in DB at the start of transaction.

It is boon for long running, read only queries such as backup and analytics.

Implementation -
Write lock + DB keep several different committed version of an object.
It is known as 
#### Multi Version Concurrency Control

### Read Skew in Read Commited
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

## Read Committed
1. When read from a DB, you see only committed read. (no dirty read)
    Any write by transaction only become visible to others when that transaction commits.
    Issues - See data that may be rollback; Partial data read like email unread counter and actual new email. See counter increase but no email.
2. When write, you will only overwrite data that has been committed (no dirty write).
    With dirty write, conflict writes from different transactions can be mixed up.

Dirty write prevented using locks at row level.
Dirty reads prevent -
    Using same lock. (But costly in term of read response.)
    For every object, DB remember both the old committed value and new value set by transaction that hold write lock.

## Read Uncommitted
Prevent dirty writes, but doesn't prevent dirty read.

# Durability
It is a promise that once a transaction has committed successfully, then there will be no data loss even if DB crash or hardware failure.

Nothing is perfect durable.