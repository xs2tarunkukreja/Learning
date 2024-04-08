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
## Snapshot Isolation & Repeatable Read
## Read Commited
## Read Committed
## Read Uncommitted

# Durability
It is a promise that once a transaction has committed successfully, then there will be no data loss even if DB crash or hardware failure.

Nothing is perfect durable.