--------------------------- MODULE ThreadSafety ---------------------------
EXTENDS Integers, FiniteSets
CONSTANTS Threads, Mutation
VARIABLES pc, stats, lock, groups, cached, badStats
vars == <<pc, stats, lock, groups, cached, badStats>>
\* Calls are not serialized. Each thread publishes into its own statistics
\* view; "sharedStats" restores one shared report. Only the group cache check,
\* creation and store hold a lock; "unlockedCache" removes it.
Slot(t) == IF Mutation = "sharedStats" THEN CHOOSE s \in Threads : TRUE ELSE t
Init == /\ pc = [t \in Threads |-> 0] /\ stats = [t \in Threads |-> 0] /\ lock = 0
        /\ groups = {} /\ cached = FALSE /\ badStats = FALSE
Enter(t) == /\ pc[t] = 0 /\ pc' = [pc EXCEPT ![t] = 1]
            /\ stats' = [stats EXCEPT ![Slot(t)] = 0]
            /\ UNCHANGED <<lock, groups, cached, badStats>>
Acquire(t) == /\ pc[t] = 1 /\ (Mutation = "unlockedCache" \/ lock = 0)
              /\ pc' = [pc EXCEPT ![t] = 2]
              /\ lock' = IF Mutation = "unlockedCache" THEN lock ELSE t
              /\ UNCHANGED <<stats, groups, cached, badStats>>
CheckCache(t) == /\ pc[t] = 2
                 /\ pc' = [pc EXCEPT ![t] = IF cached THEN 4 ELSE 3]
                 /\ UNCHANGED <<stats, lock, groups, cached, badStats>>
Create(t) == /\ pc[t] = 3 /\ pc' = [pc EXCEPT ![t] = 4]
             /\ groups' = groups \cup {t} /\ cached' = TRUE
             /\ UNCHANGED <<stats, lock, badStats>>
Release(t) == /\ pc[t] = 4 /\ pc' = [pc EXCEPT ![t] = 5]
              /\ lock' = IF Mutation = "unlockedCache" THEN lock ELSE 0
              /\ UNCHANGED <<stats, groups, cached, badStats>>
Publish(t) == /\ pc[t] = 5 /\ pc' = [pc EXCEPT ![t] = 6]
              /\ stats' = [stats EXCEPT ![Slot(t)] = t]
              /\ UNCHANGED <<lock, groups, cached, badStats>>
Read(t) == /\ pc[t] = 6 /\ pc' = [pc EXCEPT ![t] = 7]
           /\ badStats' = (badStats \/ stats[Slot(t)] # t)
           /\ UNCHANGED <<stats, lock, groups, cached>>
Done == \A t \in Threads : pc[t] = 7
Next == (\E t \in Threads : Enter(t) \/ Acquire(t) \/ CheckCache(t) \/ Create(t)
                            \/ Release(t) \/ Publish(t) \/ Read(t))
        \/ (Done /\ UNCHANGED vars)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
StatsIsolation == ~badStats
GroupReuse == Cardinality(groups) <= 1
Termination == <>Done
=============================================================================
