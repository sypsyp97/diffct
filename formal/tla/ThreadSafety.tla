--------------------------- MODULE ThreadSafety ---------------------------
EXTENDS Integers, FiniteSets
CONSTANTS Threads, Serialized
VARIABLES pc, stats, lock, groups, cached, badStats
vars == <<pc, stats, lock, groups, cached, badStats>>
Init == /\ pc = [t \in Threads |-> 0] /\ stats = 0 /\ lock = 0
        /\ groups = {} /\ cached = FALSE /\ badStats = FALSE
Enter(t) == /\ pc[t] = 0 /\ (~Serialized \/ lock = 0)
            /\ pc' = [pc EXCEPT ![t] = 1] /\ stats' = 0
            /\ lock' = IF Serialized THEN t ELSE lock
            /\ UNCHANGED <<groups, cached, badStats>>
CheckCache(t) == /\ pc[t] = 1
                 /\ pc' = [pc EXCEPT ![t] = IF cached THEN 3 ELSE 2]
                 /\ UNCHANGED <<stats, lock, groups, cached, badStats>>
Create(t) == /\ pc[t] = 2 /\ pc' = [pc EXCEPT ![t] = 3]
             /\ groups' = groups \cup {t} /\ cached' = TRUE
             /\ UNCHANGED <<stats, lock, badStats>>
Publish(t) == /\ pc[t] = 3 /\ pc' = [pc EXCEPT ![t] = 4] /\ stats' = t
              /\ UNCHANGED <<lock, groups, cached, badStats>>
Read(t) == /\ pc[t] = 4 /\ pc' = [pc EXCEPT ![t] = 5]
           /\ badStats' = (badStats \/ stats # t) /\ lock' = IF Serialized THEN 0 ELSE lock
           /\ UNCHANGED <<stats, groups, cached>>
Done == \A t \in Threads : pc[t] = 5
Next == (\E t \in Threads : Enter(t) \/ CheckCache(t) \/ Create(t) \/ Publish(t) \/ Read(t))
        \/ (Done /\ UNCHANGED vars)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
StatsIsolation == ~badStats
GroupReuse == Cardinality(groups) <= 1
Termination == <>Done
=============================================================================
