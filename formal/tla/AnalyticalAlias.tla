-------------------------- MODULE AnalyticalAlias --------------------------
EXTENDS Integers, FiniteSets

CONSTANT Mutation
VARIABLES kind, outputBytes, pc, reads, writes, memory, corruptRead

vars == <<kind, outputBytes, pc, reads, writes, memory, corruptRead>>
Kinds == {"projection", "angular", "filter", "geometry", "angles"}
Reread == kind \in {"projection", "angular", "filter"}
InputBytes == {0, 2}
Addresses == 0..5
UnsafeAlias == Reread /\ InputBytes \cap outputBytes # {}

Init == /\ kind \in Kinds
        \* Overlap, same-allocation interleaving, and distinct allocation.
        /\ outputBytes \in {{0, 1}, {1, 3}, {4, 5}}
        /\ pc = "admit" /\ reads = 0 /\ writes = 0
        /\ memory = [a \in Addresses |-> 0] /\ corruptRead = FALSE

Admit == /\ pc = "admit"
         /\ pc' = IF UnsafeAlias /\ Mutation # "legacy" THEN "rejected" ELSE "read"
         /\ UNCHANGED <<kind, outputBytes, reads, writes, memory, corruptRead>>

Read == /\ pc = "read" /\ reads < 2
        /\ reads' = reads + 1 /\ pc' = "write"
        /\ corruptRead' = (corruptRead \/ (Reread /\ \E a \in InputBytes : memory[a] # 0))
        /\ UNCHANGED <<kind, outputBytes, writes, memory>>

Write == /\ pc = "write" /\ writes < 2 /\ Mutation # "stall"
         /\ writes' = writes + 1
         /\ pc' = IF writes = 1 THEN "done" ELSE "read"
         /\ memory' = [a \in Addresses |-> IF a \in outputBytes THEN 1 ELSE memory[a]]
         /\ UNCHANGED <<kind, outputBytes, reads, corruptRead>>

\* The negative progress control omits the synchronous sink write entirely.
Stall == pc = "write" /\ Mutation = "stall" /\ UNCHANGED vars
Terminal == pc \in {"done", "rejected"} /\ UNCHANGED vars
Next == Admit \/ Read \/ Write \/ Stall \/ Terminal
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

TypeOK == /\ kind \in Kinds /\ outputBytes \subseteq Addresses
          /\ pc \in {"admit", "read", "write", "done", "rejected"}
          /\ reads \in 0..2 /\ writes \in 0..2
          /\ memory \in [Addresses -> 0..1] /\ corruptRead \in BOOLEAN
AdmissionSafety == pc \in {"read", "write", "done"} => ~UnsafeAlias
RejectionBeforeIO == pc = "rejected" => reads = 0 /\ writes = 0
NoCorruptedRead == ~corruptRead
Termination == <>(pc \in {"done", "rejected"})

\* Reachability controls deliberately violate the named invariant.
RejectedNotReached == pc # "rejected"
InterleavedCompletionNotReached == ~(Reread /\ outputBytes = {1, 3} /\ pc = "done")
SnapshotCompletionNotReached == ~(~Reread /\ outputBytes = {0, 1} /\ pc = "done")
=============================================================================
