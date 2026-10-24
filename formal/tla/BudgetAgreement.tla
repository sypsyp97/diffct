-------------------------- MODULE BudgetAgreement --------------------------
EXTENDS Integers, FiniteSets
CONSTANTS Ranks, Mutation
VARIABLES free, graph, tiles, cuda, hostData, hostFree, devicePlan, hostPlan, done
vars == <<free, graph, tiles, cuda, hostData, hostFree, devicePlan, hostPlan, done>>
Min(S) == CHOOSE x \in S : \A y \in S : x <= y
Max(a, b) == IF a >= b THEN a ELSE b
LocalDevice(r) == IF cuda[r] THEN Max(8, (free[r] - graph[r]) \div 4) ELSE 0
LocalHost(r) == IF hostData[r] THEN Max(8, Min({hostFree[r] \div 4, free[r] \div 4})) ELSE 0
SharedDevice == IF \E r \in Ranks : ~cuda[r] THEN 0 ELSE Min({LocalDevice(r) : r \in Ranks})
WithHost == {r \in Ranks : hostData[r]}
SharedHost == IF SharedDevice = 0 \/ WithHost = {} THEN 0
              ELSE Min({LocalHost(r) : r \in WithHost})
Init == /\ free \in [Ranks -> {32, 64}] /\ graph \in [Ranks -> {0, 8}]
        /\ tiles \in [Ranks -> {0, 1, 2}] /\ cuda \in [Ranks -> BOOLEAN]
        /\ hostData \in [Ranks -> BOOLEAN] /\ hostFree \in [Ranks -> {32, 64}]
        /\ devicePlan = [r \in Ranks |-> 0] /\ hostPlan = [r \in Ranks |-> 0] /\ done = FALSE
Agree == /\ ~done /\ done' = TRUE
         /\ devicePlan' = [r \in Ranks |-> IF Mutation = "device" THEN LocalDevice(r) ELSE SharedDevice]
         /\ hostPlan' = [r \in Ranks |-> IF Mutation = "host" THEN LocalHost(r) ELSE SharedHost]
         /\ UNCHANGED <<free, graph, tiles, cuda, hostData, hostFree>>
Next == Agree \/ (done /\ UNCHANGED vars)
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
DeviceAgreement == Cardinality({devicePlan[r] : r \in Ranks}) = 1
HostAgreement == Cardinality({hostPlan[r] : r \in Ranks}) = 1
Termination == <>done
=============================================================================
